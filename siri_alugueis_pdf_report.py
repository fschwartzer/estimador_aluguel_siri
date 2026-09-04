from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html import escape
from io import BytesIO
import gzip
import math
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as ReportImage,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


INK = colors.HexColor("#2B2020")
MUTED = colors.HexColor("#7A6161")
LINE = colors.HexColor("#DDE4EA")
NAVY = colors.HexColor("#7F1D1D")
TEAL = colors.HexColor("#B91C1C")
SOFT_TEAL = colors.HexColor("#E9F5F4")
SOFT_BLUE = colors.HexColor("#EEF4F8")
RED = colors.HexColor("#B42318")
WHITE = colors.white

MAP_WIDTH_PX = 1_400
MAP_HEIGHT_PX = 660
MAP_HORIZONTAL_PADDING_PX = 100
MAP_VERTICAL_PADDING_PX = 80
TILE_SIZE_PX = 256
MAX_TILE_REQUESTS = 48
MAX_VECTOR_TILE_ZOOM = 14
MIN_MAP_ZOOM = 3.0
MAX_MAP_ZOOM = 20.0
CARTO_ATTRIBUTION = "© OpenStreetMap contributors · © CARTO"
CARTO_VECTOR_TILE_URL = (
    "https://tiles-{subdomain}.basemaps.cartocdn.com/"
    "vectortiles/carto.streets/v1/{zoom}/{x}/{y}.mvt"
)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _first_series(data: pd.DataFrame, column: str | None) -> pd.Series:
    if not column or column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    selected = data.loc[:, column]
    if isinstance(selected, pd.DataFrame):
        selected = selected.iloc[:, 0]
    return selected


def _money_br(value: Any, decimals: int = 2) -> str:
    number = _finite_number(value)
    if number is None:
        return "-"
    text = f"{number:,.{decimals}f}"
    return "R$ " + text.replace(",", "X").replace(".", ",").replace("X", ".")


def _number_br(value: Any, decimals: int = 2) -> str:
    number = _finite_number(value)
    if number is None:
        return "-"
    text = f"{number:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def _percent_br(value: Any, decimals: int = 1) -> str:
    number = _finite_number(value)
    if number is None:
        return "-"
    return f"{number:.{decimals}f}%".replace(".", ",")


def _measurement_br(value: Any, unit: str, decimals: int = 2) -> str:
    number = _finite_number(value)
    if number is None:
        return "-"
    return f"{_number_br(number, decimals)} {unit}"


def _paragraph_text(value: Any, default: str = "-") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return escape(text) if text else default


def calculate_comparable_cod(
    neighbors: pd.DataFrame,
    value_column: str = "_valor_unitario_ajustado",
) -> float:
    """
    Calcula a dispersão percentual dos VUs comparáveis em torno da mediana.

    É um diagnóstico descritivo da seleção final. Não é o COD de um estudo de
    razões avaliação/preço, que exige previsões fora da amostra e valores
    observados independentes.
    """
    values = pd.to_numeric(
        _first_series(neighbors, value_column),
        errors="coerce",
    ).to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    if median <= 0:
        return float("nan")
    return float(np.mean(np.abs(values - median)) / median * 100.0)


def prepare_report_comparables(
    neighbors: pd.DataFrame,
    *,
    latitude_column: str | None,
    longitude_column: str | None,
    type_column: str | None,
    reference_area_column: str | None,
) -> pd.DataFrame:
    """Cria uma visão enxuta e estável dos comparáveis usados no relatório."""
    report = pd.DataFrame(index=neighbors.index)
    report["tipo"] = (
        _first_series(neighbors, type_column).astype("string").fillna("-")
    )
    report["area"] = pd.to_numeric(
        _first_series(neighbors, reference_area_column), errors="coerce"
    )
    report["valor_unitario"] = pd.to_numeric(
        _first_series(neighbors, "_valor_unitario_ajustado"), errors="coerce"
    )
    report["valor_unitario_robusto"] = pd.to_numeric(
        _first_series(neighbors, "_valor_unitario_robusto"), errors="coerce"
    )
    report["peso"] = pd.to_numeric(
        _first_series(neighbors, "_peso_knn"), errors="coerce"
    )
    report["distancia_km"] = pd.to_numeric(
        _first_series(neighbors, "_distancia_geografica_km"), errors="coerce"
    )
    report["linha_excel"] = pd.to_numeric(
        _first_series(neighbors, "_row_excel"), errors="coerce"
    )
    report["latitude"] = pd.to_numeric(
        _first_series(neighbors, latitude_column), errors="coerce"
    )
    report["longitude"] = pd.to_numeric(
        _first_series(neighbors, longitude_column), errors="coerce"
    )
    report = report.sort_values("peso", ascending=False, na_position="last")
    report = report.reset_index(drop=True)
    report.insert(0, "ponto", np.arange(1, len(report) + 1))
    return report


def _mercator_xy(latitude: float, longitude: float, zoom: float) -> tuple[float, float]:
    latitude = float(np.clip(latitude, -85.05112878, 85.05112878))
    scale = TILE_SIZE_PX * (2**zoom)
    x = (longitude + 180.0) / 360.0 * scale
    sin_latitude = math.sin(math.radians(latitude))
    y = (
        0.5
        - math.log((1.0 + sin_latitude) / (1.0 - sin_latitude))
        / (4.0 * math.pi)
    ) * scale
    return x, y


def _choose_zoom(points: list[tuple[float, float]]) -> float:
    """Enquadra os pontos periféricos com uma margem fixa para os marcadores."""
    if len(points) <= 1:
        return MAX_MAP_ZOOM

    # No zoom zero, dobrar o zoom dobra exatamente os vãos em pixels. Isso
    # permite calcular um zoom contínuo que usa a área disponível sem excluir
    # os pontos extremos por arredondamento para um nível inteiro.
    pixels = [_mercator_xy(lat, lon, 0.0) for lat, lon in points]
    span_x = max(value[0] for value in pixels) - min(value[0] for value in pixels)
    span_y = max(value[1] for value in pixels) - min(value[1] for value in pixels)
    usable_width = MAP_WIDTH_PX - 2 * MAP_HORIZONTAL_PADDING_PX
    usable_height = MAP_HEIGHT_PX - 2 * MAP_VERTICAL_PADDING_PX
    zoom_limits = []
    if span_x > 0:
        zoom_limits.append(math.log2(usable_width / span_x))
    if span_y > 0:
        zoom_limits.append(math.log2(usable_height / span_y))
    if not zoom_limits:
        return MAX_MAP_ZOOM
    return float(np.clip(min(zoom_limits), MIN_MAP_ZOOM, MAX_MAP_ZOOM))


def _fetch_vector_tile(
    zoom: int,
    tile_x: int,
    tile_y: int,
) -> tuple[int, int, bytes | None]:
    max_index = 2**zoom
    wrapped_x = tile_x % max_index
    if tile_y < 0 or tile_y >= max_index:
        return tile_x, tile_y, None
    subdomain = "abcd"[(wrapped_x + tile_y) % 4]
    url = CARTO_VECTOR_TILE_URL.format(
        subdomain=subdomain,
        zoom=zoom,
        x=wrapped_x,
        y=tile_y,
    )
    request = Request(url, headers={"User-Agent": "SIRI-Alugueis/1.0 PDF report"})
    try:
        with urlopen(request, timeout=4.0) as response:
            tile = response.read()
            encoding = response.headers.get("Content-Encoding", "").casefold()
        if encoding == "gzip" or tile.startswith(b"\x1f\x8b"):
            tile = gzip.decompress(tile)
        return tile_x, tile_y, tile
    except Exception:
        return tile_x, tile_y, None


def _fallback_map(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 80):
        draw.line((x, 0, x, height), fill="#ECEEED", width=1)
    for y in range(0, height, 80):
        draw.line((0, y, width, y), fill="#ECEEED", width=1)
    return image


def _tile_point(
    coordinate: list[float] | tuple[float, float],
    *,
    extent: int,
    destination_x: float,
    destination_y: float,
    tile_size_px: float = TILE_SIZE_PX,
) -> tuple[float, float]:
    return (
        destination_x + float(coordinate[0]) / extent * tile_size_px,
        destination_y + float(coordinate[1]) / extent * tile_size_px,
    )


def _draw_polygon_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: Mapping[str, Any],
    *,
    extent: int,
    destination_x: float,
    destination_y: float,
    fill: str,
    tile_size_px: float = TILE_SIZE_PX,
) -> None:
    coordinates = geometry.get("coordinates", [])
    polygons = (
        [coordinates]
        if geometry.get("type") == "Polygon"
        else coordinates
        if geometry.get("type") == "MultiPolygon"
        else []
    )
    for polygon in polygons:
        for ring_index, ring in enumerate(polygon):
            points = [
                _tile_point(
                    coordinate,
                    extent=extent,
                    destination_x=destination_x,
                    destination_y=destination_y,
                    tile_size_px=tile_size_px,
                )
                for coordinate in ring
            ]
            if len(points) >= 3:
                draw.polygon(
                    points,
                    fill=fill if ring_index == 0 else "#FAFAF8",
                )


def _draw_line_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: Mapping[str, Any],
    *,
    extent: int,
    destination_x: float,
    destination_y: float,
    fill: str,
    width: int,
    tile_size_px: float = TILE_SIZE_PX,
) -> None:
    coordinates = geometry.get("coordinates", [])
    lines = (
        [coordinates]
        if geometry.get("type") == "LineString"
        else coordinates
        if geometry.get("type") == "MultiLineString"
        else []
    )
    for line in lines:
        points = [
            _tile_point(
                coordinate,
                extent=extent,
                destination_x=destination_x,
                destination_y=destination_y,
                tile_size_px=tile_size_px,
            )
            for coordinate in line
        ]
        if len(points) >= 2:
            draw.line(points, fill=fill, width=max(int(width), 1), joint="curve")


def _road_width(road_class: str) -> int:
    if road_class in {"motorway", "trunk"}:
        return 5
    if road_class == "primary":
        return 4
    if road_class in {"secondary", "tertiary"}:
        return 3
    if road_class == "minor":
        return 2
    return 1


def _render_vector_tile(
    image: Image.Image,
    tile_bytes: bytes,
    *,
    destination_x: float,
    destination_y: float,
    tile_size_px: float = TILE_SIZE_PX,
) -> None:
    import mapbox_vector_tile

    decoded = mapbox_vector_tile.decode(
        tile_bytes,
        default_options={"y_coord_down": True},
    )
    draw = ImageDraw.Draw(image)

    fill_layers = (
        ("landcover", "#EEF3EB"),
        ("landuse", "#F0F0ED"),
        ("park", "#EAF1E9"),
        ("water", "#D4DADC"),
        ("building", "#E8E8E5"),
    )
    for layer_name, fill in fill_layers:
        layer = decoded.get(layer_name, {})
        extent = int(layer.get("extent", 4096))
        for feature in layer.get("features", []):
            _draw_polygon_geometry(
                draw,
                feature.get("geometry", {}),
                extent=extent,
                destination_x=destination_x,
                destination_y=destination_y,
                fill=fill,
                tile_size_px=tile_size_px,
            )

    # O tile-pai é ampliado geometricamente acima do zoom vetorial máximo,
    # mas a espessura das vias deve continuar em pixels de tela. Escalá-la
    # junto com o tile transformaria ruas em faixas excessivamente largas.
    stroke_scale = 1.0

    boundary_layer = decoded.get("boundary", {})
    boundary_extent = int(boundary_layer.get("extent", 4096))
    for feature in boundary_layer.get("features", []):
        _draw_line_geometry(
            draw,
            feature.get("geometry", {}),
            extent=boundary_extent,
            destination_x=destination_x,
            destination_y=destination_y,
            fill="#E1C5C7",
            width=round(stroke_scale),
            tile_size_px=tile_size_px,
        )

    road_layer = decoded.get("transportation", {})
    road_extent = int(road_layer.get("extent", 4096))
    road_features = road_layer.get("features", [])
    for case_pass in (True, False):
        for feature in road_features:
            road_class = str(feature.get("properties", {}).get("class", ""))
            width = _road_width(road_class)
            if case_pass:
                width += 2
                fill = "#D8D8D6"
            else:
                fill = "#FFFFFF" if width >= 2 else "#E3E3E0"
            geometry = feature.get("geometry", {})
            if geometry.get("type") in {"Polygon", "MultiPolygon"}:
                _draw_polygon_geometry(
                    draw,
                    geometry,
                    extent=road_extent,
                    destination_x=destination_x,
                    destination_y=destination_y,
                    fill=fill,
                    tile_size_px=tile_size_px,
                )
            else:
                _draw_line_geometry(
                    draw,
                    geometry,
                    extent=road_extent,
                    destination_x=destination_x,
                    destination_y=destination_y,
                    fill=fill,
                    width=round(width * stroke_scale),
                    tile_size_px=tile_size_px,
                )

    waterway_layer = decoded.get("waterway", {})
    waterway_extent = int(waterway_layer.get("extent", 4096))
    for feature in waterway_layer.get("features", []):
        _draw_line_geometry(
            draw,
            feature.get("geometry", {}),
            extent=waterway_extent,
            destination_x=destination_x,
            destination_y=destination_y,
            fill="#D1DBDF",
            width=round(2 * stroke_scale),
            tile_size_px=tile_size_px,
        )


def _draw_centered_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    fill: str,
    font: ImageFont.ImageFont,
) -> None:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        width = box[2] - box[0]
        height = box[3] - box[1]
    except AttributeError:
        width, height = draw.textsize(text, font=font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2 - 1), text, fill=fill, font=font)


def _map_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ("DejaVuSans-Bold.ttf", "arialbd.ttf")
        if bold
        else ("DejaVuSans.ttf", "arial.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_comparables_map_image(
    comparables: pd.DataFrame,
    *,
    target_latitude: float,
    target_longitude: float,
    fetch_tiles: bool = True,
) -> tuple[bytes, str]:
    """Renderiza o avaliando e os comparáveis sobre o CARTO Positron."""
    target_latitude = float(target_latitude)
    target_longitude = float(target_longitude)
    valid = (
        comparables["latitude"].between(-90, 90)
        & comparables["longitude"].between(-180, 180)
        & comparables["latitude"].notna()
        & comparables["longitude"].notna()
    )
    mapped = comparables.loc[valid].copy()
    points = [(target_latitude, target_longitude)] + list(
        zip(mapped["latitude"].astype(float), mapped["longitude"].astype(float))
    )
    display_zoom = _choose_zoom(points)
    # O tileset vetorial oficial do CARTO é publicado até o zoom 14. Em níveis
    # maiores, o tile-pai é ampliado como vetor para preservar o enquadramento.
    tile_zoom = min(math.floor(display_zoom), MAX_VECTOR_TILE_ZOOM)
    tile_scale = 2 ** (display_zoom - tile_zoom)
    rendered_tile_size = TILE_SIZE_PX * tile_scale
    global_pixels = [
        _mercator_xy(lat, lon, display_zoom) for lat, lon in points
    ]
    center_x = (min(x for x, _ in global_pixels) + max(x for x, _ in global_pixels)) / 2
    center_y = (min(y for _, y in global_pixels) + max(y for _, y in global_pixels)) / 2
    viewport_left = center_x - MAP_WIDTH_PX / 2
    viewport_top = center_y - MAP_HEIGHT_PX / 2
    first_tile_x = math.floor(viewport_left / rendered_tile_size)
    last_tile_x = math.floor(
        (viewport_left + MAP_WIDTH_PX) / rendered_tile_size
    )
    first_tile_y = math.floor(viewport_top / rendered_tile_size)
    last_tile_y = math.floor(
        (viewport_top + MAP_HEIGHT_PX) / rendered_tile_size
    )
    tile_coordinates = [
        (x, y)
        for x in range(first_tile_x, last_tile_x + 1)
        for y in range(first_tile_y, last_tile_y + 1)
    ]

    image = _fallback_map(MAP_WIDTH_PX, MAP_HEIGHT_PX)
    loaded_tiles = 0
    if fetch_tiles and len(tile_coordinates) <= MAX_TILE_REQUESTS:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_fetch_vector_tile, tile_zoom, x, y)
                for x, y in tile_coordinates
            ]
            for future in as_completed(futures):
                tile_x, tile_y, tile_bytes = future.result()
                if tile_bytes is None:
                    continue
                destination_x = int(
                    tile_x * rendered_tile_size - viewport_left
                )
                destination_y = int(
                    tile_y * rendered_tile_size - viewport_top
                )
                try:
                    _render_vector_tile(
                        image,
                        tile_bytes,
                        destination_x=destination_x,
                        destination_y=destination_y,
                        tile_size_px=rendered_tile_size,
                    )
                    loaded_tiles += 1
                except Exception:
                    continue

    draw = ImageDraw.Draw(image, "RGBA")
    marker_font = _map_font(21, bold=True)
    legend_font = _map_font(17)
    attribution_font = _map_font(14)

    target_x, target_y = global_pixels[0]
    target_xy = (target_x - viewport_left, target_y - viewport_top)
    for row in mapped.itertuples(index=False):
        comp_x, comp_y = _mercator_xy(
            float(row.latitude), float(row.longitude), display_zoom
        )
        comp_xy = (comp_x - viewport_left, comp_y - viewport_top)
        draw.line((*target_xy, *comp_xy), fill=(23, 59, 87, 70), width=2)

    weights = pd.to_numeric(mapped["peso"], errors="coerce").fillna(0.0)
    mapped["peso_mapa"] = weights
    minimum_weight = float(weights.min()) if not weights.empty else 0.0
    maximum_weight = float(weights.max()) if not weights.empty else 0.0
    for row in mapped.itertuples(index=False):
        comp_x, comp_y = _mercator_xy(
            float(row.latitude), float(row.longitude), display_zoom
        )
        px = comp_x - viewport_left
        py = comp_y - viewport_top
        if maximum_weight > minimum_weight:
            radius = 10 + 8 * (float(row.peso_mapa) - minimum_weight) / (
                maximum_weight - minimum_weight
            )
        else:
            radius = 13
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(14, 124, 123, 225),
            outline=(255, 255, 255, 255),
            width=3,
        )
        _draw_centered_label(
            draw,
            (px, py),
            str(int(row.ponto)),
            fill="#FFFFFF",
            font=marker_font,
        )

    target_radius = 16
    draw.ellipse(
        (
            target_xy[0] - target_radius,
            target_xy[1] - target_radius,
            target_xy[0] + target_radius,
            target_xy[1] + target_radius,
        ),
        fill=(180, 35, 24, 245),
        outline=(255, 255, 255, 255),
        width=4,
    )
    _draw_centered_label(
        draw,
        target_xy,
        "A",
        fill="#FFFFFF",
        font=marker_font,
    )

    draw.rounded_rectangle(
        (16, 14, 372, 59),
        radius=9,
        fill=(255, 255, 255, 225),
        outline=(210, 219, 227, 255),
        width=1,
    )
    draw.ellipse((31, 26, 51, 46), fill=(180, 35, 24, 245))
    draw.text((61, 24), "Avaliando", fill="#172033", font=legend_font)
    draw.ellipse((176, 26, 196, 46), fill=(14, 124, 123, 225))
    draw.text((206, 24), "Comparáveis", fill="#172033", font=legend_font)

    attribution_box = draw.textbbox(
        (0, 0), CARTO_ATTRIBUTION, font=attribution_font
    )
    attribution_width = attribution_box[2] - attribution_box[0]
    draw.rectangle(
        (
            MAP_WIDTH_PX - attribution_width - 20,
            MAP_HEIGHT_PX - 24,
            MAP_WIDTH_PX,
            MAP_HEIGHT_PX,
        ),
        fill=(255, 255, 255, 205),
    )
    draw.text(
        (MAP_WIDTH_PX - attribution_width - 10, MAP_HEIGHT_PX - 19),
        CARTO_ATTRIBUTION,
        fill="#344054",
        font=attribution_font,
    )

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    if loaded_tiles == len(tile_coordinates) and loaded_tiles > 0:
        source_note = "Fundo: Claro - CARTO Positron (tiles vetoriais oficiais)."
    elif loaded_tiles > 0:
        source_note = "Fundo: Claro - CARTO Positron (tiles parcialmente disponíveis)."
    else:
        source_note = "Fundo CARTO Positron indisponível; grade neutra usada como contingência."
    return output.getvalue(), source_note


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            spaceAfter=3 * mm,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=NAVY,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=INK,
        ),
        "note": ParagraphStyle(
            "ReportNote",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=6.7,
            leading=8.5,
            textColor=MUTED,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.6,
            leading=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.5,
            leading=8,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=6.4,
            leading=8,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


def _metric_card(label: str, value: str, styles: Mapping[str, ParagraphStyle]) -> Table:
    card = Table(
        [
            [Paragraph(label, styles["metric_label"])],
            [Paragraph(value, styles["metric_value"])],
        ],
        colWidths=[42 * mm],
        rowHeights=[7 * mm, 10 * mm],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1 * mm),
            ]
        )
    )
    return card


def _page_footer(canvas: Any, document: BaseDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 6.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "SIRI Aluguéis · Relatório sintético de estimativa")
    canvas.drawRightString(
        A4[0] - 18 * mm,
        8.5 * mm,
        f"Página {document.page}",
    )
    canvas.restoreState()


def build_inference_report_pdf(
    *,
    estimated_unit_value: float,
    estimated_total_value: float,
    confidence_score: float,
    neighbors: pd.DataFrame,
    purpose: str,
    area_regime_label: str,
    target: Mapping[str, Any],
    latitude_column: str | None,
    longitude_column: str | None,
    type_column: str | None,
    reference_area_column: str | None,
    logo_path: str | Path | None = None,
    generated_at: datetime | None = None,
    fetch_map_tiles: bool = True,
) -> bytes:
    """Gera o relatório sintético do SIRI Aluguéis em memória."""
    if neighbors.empty:
        raise ValueError("O relatório exige ao menos um comparável.")

    comparables = prepare_report_comparables(
        neighbors,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        type_column=type_column,
        reference_area_column=reference_area_column,
    )
    cod = calculate_comparable_cod(neighbors)
    styles = _styles()
    output = BytesIO()
    document = BaseDocTemplate(
        output,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title="SIRI Aluguéis - Relatório sintético de estimativa",
        author="SIRI Aluguéis",
        subject="Estimativa imobiliária por referências amostrais",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="report",
    )
    document.addPageTemplates(
        [PageTemplate(id="report", frames=[frame], onPage=_page_footer)]
    )

    story: list[Any] = []
    header_cells: list[Any] = []
    resolved_logo = Path(logo_path) if logo_path else None
    if resolved_logo and resolved_logo.is_file():
        header_cells.append(
            ReportImage(str(resolved_logo), width=50 * mm, height=14 * mm)
        )
    else:
        header_cells.append(
            Paragraph(
                "<b>SIRI ALUGUÉIS</b><br/><font size='7'>"
                "Valor Estimado por Referências Amostrais</font>",
                styles["body"],
            )
        )
    header_cells.append(
        Paragraph("Relatório sintético de inferência", styles["title"])
    )
    header = Table([header_cells], colWidths=[56 * mm, 115 * mm])
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.8, TEAL),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.extend([header, Spacer(1, 3 * mm)])

    timestamp = generated_at or datetime.now().astimezone()
    context = (
        f"Finalidade: <b>{_paragraph_text(purpose)}</b> · Base de área: "
        f"<b>{_paragraph_text(area_regime_label)}</b> · Emitido em "
        f"{timestamp.strftime('%d/%m/%Y %H:%M')}"
    )
    story.append(Paragraph(context, styles["subtitle"]))

    metrics = [
        _metric_card(
            "VALOR UNITÁRIO",
            _money_br(estimated_unit_value) + "/m²",
            styles,
        ),
        _metric_card("VALOR TOTAL", _money_br(estimated_total_value), styles),
        _metric_card("COD DOS COMPARÁVEIS", _percent_br(cod), styles),
        _metric_card(
            "SCORE DE CONFIANÇA",
            _percent_br(confidence_score, 0),
            styles,
        ),
    ]
    metric_grid = Table([metrics], colWidths=[43 * mm] * 4, hAlign="LEFT")
    metric_grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1 * mm),
            ]
        )
    )
    story.extend([metric_grid, Spacer(1, 3 * mm)])

    target_latitude = _finite_number(target.get("latitude"))
    target_longitude = _finite_number(target.get("longitude"))
    valid_target = (
        target_latitude is not None
        and target_longitude is not None
        and -90 <= target_latitude <= 90
        and -180 <= target_longitude <= 180
    )
    valid_comparables = comparables.loc[
        comparables["latitude"].between(-90, 90)
        & comparables["longitude"].between(-180, 180)
        & comparables["latitude"].notna()
        & comparables["longitude"].notna()
    ]

    story.append(Paragraph("Localização dos comparáveis", styles["section"]))
    if valid_target and not valid_comparables.empty:
        map_bytes, map_note = render_comparables_map_image(
            comparables,
            target_latitude=target_latitude,
            target_longitude=target_longitude,
            fetch_tiles=fetch_map_tiles,
        )
        story.append(
            ReportImage(
                BytesIO(map_bytes),
                width=document.width,
                height=document.width * MAP_HEIGHT_PX / MAP_WIDTH_PX,
            )
        )
        story.append(Spacer(1, 1.2 * mm))
        story.append(
            Paragraph(
                f"{map_note} A = imóvel avaliando; números = posição na tabela. ",
                styles["note"],
            )
        )
    else:
        missing_map = Table(
            [[Paragraph(
                "Mapa não gerado: a estimativa não dispõe de coordenadas "
                "válidas do avaliando e dos comparáveis.",
                styles["body"],
            )]],
            colWidths=[document.width],
            rowHeights=[24 * mm],
        )
        missing_map.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
                    ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ]
            )
        )
        story.append(missing_map)

    story.append(PageBreak())
    story.append(Paragraph("Comparáveis utilizados", styles["section"]))
    table_data: list[list[Any]] = [
        [
            Paragraph("Ponto", styles["table_header"]),
            Paragraph("Tipo", styles["table_header"]),
            Paragraph("Área", styles["table_header"]),
            Paragraph("VU ajustado", styles["table_header"]),
            Paragraph("VU robusto", styles["table_header"]),
            Paragraph("Peso", styles["table_header"]),
            Paragraph("Distância", styles["table_header"]),
            Paragraph("Linha", styles["table_header"]),
        ]
    ]
    for row in comparables.itertuples(index=False):
        line_number = _finite_number(row.linha_excel)
        table_data.append(
            [
                Paragraph(str(int(row.ponto)), styles["table_cell"]),
                Paragraph(_paragraph_text(row.tipo), styles["table_cell"]),
                Paragraph(_measurement_br(row.area, "m²"), styles["table_cell"]),
                Paragraph(_money_br(row.valor_unitario), styles["table_cell"]),
                Paragraph(
                    _money_br(row.valor_unitario_robusto), styles["table_cell"]
                ),
                Paragraph(
                    _percent_br(
                        float(row.peso) * 100 if _finite_number(row.peso) is not None else np.nan
                    ),
                    styles["table_cell"],
                ),
                Paragraph(
                    _measurement_br(row.distancia_km, "km", 3),
                    styles["table_cell"],
                ),
                Paragraph(
                    str(int(line_number)) if line_number is not None else "-",
                    styles["table_cell"],
                ),
            ]
        )

    comparable_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            12 * mm,
            23 * mm,
            20 * mm,
            29 * mm,
            29 * mm,
            18 * mm,
            23 * mm,
            13 * mm,
        ],
        hAlign="LEFT",
    )
    comparable_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SOFT_TEAL]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4 * mm),
            ]
        )
    )
    story.append(comparable_table)
    story.append(Spacer(1, 2.5 * mm))
    story.append(
        Paragraph(
            "VU robusto: valor unitário ajustado após winsorização por MAD "
            "ponderada, que limita valores extremos sem excluir o comparável. "
            "Os pesos são aplicados aos VUs robustos; a soma de peso × VU "
            "robusto produz o valor unitário estimado. "
            "COD dos comparáveis: desvio absoluto médio dos valores unitários "
            "ajustados em torno da mediana, em percentual. É um diagnóstico "
            "local de dispersão; não substitui o COD de backtesting por razões "
            "avaliação/preço. O score de confiança é heurístico e não "
            "representa intervalo de confiança estatístico. Revise os "
            "comparáveis antes do uso operacional.",
            styles["note"],
        )
    )

    document.build(story)
    return output.getvalue()

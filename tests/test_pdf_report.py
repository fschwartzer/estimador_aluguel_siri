from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from siri_alugueis_pdf_report import (
    MAP_HORIZONTAL_PADDING_PX,
    MAP_HEIGHT_PX,
    MAP_VERTICAL_PADDING_PX,
    MAP_WIDTH_PX,
    MAX_MAP_ZOOM,
    MAX_VECTOR_TILE_ZOOM,
    TILE_SIZE_PX,
    _choose_zoom,
    _mercator_xy,
    build_inference_report_pdf,
    calculate_comparable_cod,
    prepare_report_comparables,
    render_comparables_map_image,
)


ROOT = Path(__file__).resolve().parents[1]


def sample_neighbors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tipo": ["Guia ITBI", "Oferta", "Guia ITBI"],
            "area": [100.0, 110.0, 95.0],
            "lat": [-30.0300, -30.0350, -30.0250],
            "lon": [-51.2300, -51.2250, -51.2350],
            "_row_excel": [12, 18, 25],
            "_valor_unitario_ajustado": [4_000.0, 5_000.0, 6_000.0],
            "_valor_unitario_robusto": [4_000.0, 4_800.0, 6_000.0],
            "_peso_knn": [0.2, 0.5, 0.3],
            "_distancia_geografica_km": [0.8, 0.4, 1.2],
        }
    )


class PdfReportTests(unittest.TestCase):
    def test_peripheral_comparables_drive_dynamic_zoom(self) -> None:
        points = [
            (-30.0300, -51.2300),
            (-30.0305, -51.2304),
            (-30.0296, -51.2295),
            (-30.0302, -51.2301),
        ]

        zoom = _choose_zoom(points)
        pixels = [_mercator_xy(lat, lon, zoom) for lat, lon in points]
        span_x = max(x for x, _ in pixels) - min(x for x, _ in pixels)
        span_y = max(y for _, y in pixels) - min(y for _, y in pixels)
        usable_width = MAP_WIDTH_PX - 2 * MAP_HORIZONTAL_PADDING_PX
        usable_height = MAP_HEIGHT_PX - 2 * MAP_VERTICAL_PADDING_PX

        self.assertGreater(zoom, 17)
        self.assertLessEqual(span_x, usable_width + 1e-6)
        self.assertLessEqual(span_y, usable_height + 1e-6)
        self.assertAlmostEqual(
            max(span_x / usable_width, span_y / usable_height),
            1.0,
            places=6,
        )

    def test_comparable_cod_uses_median_adjusted_unit_values(self) -> None:
        cod = calculate_comparable_cod(sample_neighbors())
        self.assertAlmostEqual(cod, 13.3333333333)

    def test_comparables_are_numbered_by_descending_weight(self) -> None:
        report = prepare_report_comparables(
            sample_neighbors(),
            latitude_column="lat",
            longitude_column="lon",
            type_column="tipo",
            reference_area_column="area",
        )
        self.assertEqual(report["ponto"].tolist(), [1, 2, 3])
        self.assertEqual(report["linha_excel"].tolist(), [18, 25, 12])
        self.assertTrue(np.allclose(report["peso"], [0.5, 0.3, 0.2]))
        self.assertTrue(
            np.allclose(report["valor_unitario_robusto"], [4_800.0, 6_000.0, 4_000.0])
        )

    @patch("siri_alugueis_pdf_report._render_vector_tile")
    @patch("siri_alugueis_pdf_report._fetch_vector_tile")
    def test_close_comparables_use_overzoomed_vector_tiles(
        self,
        fetch_tile,
        render_tile,
    ) -> None:
        report = prepare_report_comparables(
            sample_neighbors().assign(
                lat=[-30.03001, -30.02999, -30.03002],
                lon=[-51.23001, -51.22999, -51.23002],
            ),
            latitude_column="lat",
            longitude_column="lon",
            type_column="tipo",
            reference_area_column="area",
        )
        fetch_tile.side_effect = lambda zoom, x, y: (x, y, b"tile")

        render_comparables_map_image(
            report,
            target_latitude=-30.03,
            target_longitude=-51.23,
            fetch_tiles=True,
        )

        self.assertTrue(fetch_tile.called)
        self.assertEqual({call.args[0] for call in fetch_tile.call_args_list}, {14})
        self.assertTrue(render_tile.called)
        self.assertEqual(
            {call.kwargs["tile_size_px"] for call in render_tile.call_args_list},
            {TILE_SIZE_PX * 2 ** (MAX_MAP_ZOOM - MAX_VECTOR_TILE_ZOOM)},
        )

    def test_map_has_deterministic_offline_fallback(self) -> None:
        report = prepare_report_comparables(
            sample_neighbors(),
            latitude_column="lat",
            longitude_column="lon",
            type_column="tipo",
            reference_area_column="area",
        )
        image_bytes, note = render_comparables_map_image(
            report,
            target_latitude=-30.03,
            target_longitude=-51.23,
            fetch_tiles=False,
        )
        image = Image.open(BytesIO(image_bytes))
        self.assertEqual(image.size, (MAP_WIDTH_PX, MAP_HEIGHT_PX))
        self.assertIn("indisponível", note)

    def test_pdf_is_generated_with_map_and_comparables(self) -> None:
        pdf_bytes = build_inference_report_pdf(
            estimated_unit_value=5_100.0,
            estimated_total_value=510_000.0,
            confidence_score=88,
            neighbors=sample_neighbors(),
            purpose="apartamento",
            area_regime_label="Área privativa",
            target={"latitude": -30.03, "longitude": -51.23},
            latitude_column="lat",
            longitude_column="lon",
            type_column="tipo",
            reference_area_column="area",
            logo_path=ROOT / "static" / "siri_alugueis_header.png",
            generated_at=datetime(2026, 8, 26, 15, 0, tzinfo=timezone.utc),
            fetch_map_tiles=False,
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 20_000)


if __name__ == "__main__":
    unittest.main()

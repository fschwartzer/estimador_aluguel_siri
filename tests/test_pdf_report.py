from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from vera_pdf_report import (
    MAP_HEIGHT_PX,
    MAP_WIDTH_PX,
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
            "_peso_knn": [0.2, 0.5, 0.3],
            "_distancia_geografica_km": [0.8, 0.4, 1.2],
        }
    )


class PdfReportTests(unittest.TestCase):
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
            logo_path=ROOT / "static" / "vera_header.png",
            generated_at=datetime(2026, 8, 26, 15, 0, tzinfo=timezone.utc),
            fetch_map_tiles=False,
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 20_000)


if __name__ == "__main__":
    unittest.main()

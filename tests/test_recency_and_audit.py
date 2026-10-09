from datetime import datetime
from io import BytesIO
import unittest

import numpy as np
import pandas as pd
from pypdf import PdfReader

import estimador_knn_core_v6120 as core
from rental_audit import comparable_addresses, prepare_audit_comparables
from siri_alugueis_pdf_report import build_inference_report_pdf


def mapping():
    return core.ColumnMapping("tipo", "finalidade", "valor", None, "area", "lat", "lon", None, None, data_observacao="data_coleta")


def sample():
    return pd.DataFrame(dict(tipo=["Oferta Aluguel"]*12, finalidade=["SALA COMERCIAL"]*12, valor=np.linspace(800, 1400, 12), area=np.linspace(25, 45, 12), lat=np.linspace(-30.03, -30.032, 12), lon=np.linspace(-51.23, -51.232, 12), data_coleta=["01/10/2026", "01/01/2026"]*6, siat_logradouro=["Rua dos Andradas"]*12, siat_numero=[123.]*12, siat_complemento=["sala 404"]*12, siat_bairro=["Centro Histórico"]*12))


class RecencyTests(unittest.TestCase):
    def test_recency_has_documented_half_life_and_missing_date_floor(self):
        data = pd.DataFrame({"date":["01/10/2026", "01/04/2026", None]})
        factors, diag = core._temporal_weight_factor(data, "date", 183, .1, reference_date="2026-10-01")
        np.testing.assert_allclose(factors, [1., .5, .1])
        self.assertEqual(diag["temporal_weight_missing_dates"], 1)
        disabled, diag = core._temporal_weight_factor(data, "date", enabled=False)
        np.testing.assert_array_equal(disabled, np.ones(3))
        self.assertFalse(diag["temporal_weight_used"])

    def test_dates_include_excel_serials_and_mixed_text(self):
        dates = core._parse_registration_dates(pd.Series(["01/10/2026", 46296., None, "2026-10-01"]))
        self.assertEqual(dates.iloc[0], pd.Timestamp("2026-10-01"))
        self.assertEqual(dates.iloc[1], pd.Timestamp("1899-12-30") + pd.Timedelta(days=46296))
        self.assertTrue(pd.isna(dates.iloc[2]))
        self.assertEqual(dates.iloc[3], dates.iloc[0])

    def test_invalid_parameters_are_rejected(self):
        for options in ({"half_life_days":0}, {"half_life_days":np.inf}, {"min_factor":0}, {"min_factor":1.1}):
            with self.assertRaises(ValueError):
                core._temporal_weight_factor(sample(), "data_coleta", **options)

    def test_future_rows_are_excluded_before_preprocessing(self):
        source = sample()
        source.loc[0, "data_coleta"] = "01/12/2026"
        source.loc[0, "valor"] = 999999
        result = core.prepare_data(source, mapping(), "SALA COMERCIAL", "Valor total", "area", remove_offer_duplicates=False, observation_reference_date="2026-10-09")
        self.assertNotIn(0, result.data.index)
        self.assertEqual(result.diagnostics["preparation_future_excluded"], 1)
        self.assertIn("posterior", result.excluded_data.iloc[0]["_motivo_exclusao"])
        without = core.prepare_data(source.iloc[1:], mapping(), "SALA COMERCIAL", "Valor total", "area", remove_offer_duplicates=False)
        pd.testing.assert_series_equal(result.data._valor_unitario_ajustado, without.data._valor_unitario_ajustado)

    def test_weight_components_reproduce_estimate_with_cap(self):
        preparation = core.prepare_data(sample(), mapping(), "SALA COMERCIAL", "Valor total", "area", remove_offer_duplicates=False)
        result = core.estimate_knn(preparation, mapping(), {"area_privativa":32., "latitude":-30.0305, "longitude":-51.2305}, "area", min_k=6, max_k=12, min_effective_neighbors=5, temporal_half_life_days=90, temporal_min_factor=.1, temporal_reference_date="2026-10-09")
        neighbors = result.neighbors
        raw = neighbors._peso_distancia_bruto * neighbors._fator_temporal * neighbors._fator_edificio
        weights, _ = core._cap_and_normalize_weights(raw.to_numpy(), .25)
        np.testing.assert_allclose(weights, neighbors._peso_knn)
        self.assertAlmostEqual(neighbors._peso_knn.sum(), 1.)
        self.assertAlmostEqual((neighbors._peso_knn*neighbors._valor_unitario_robusto).sum(), result.estimated_unit_value)
        self.assertEqual(result.diagnostics["temporal_weight_date_column"], "data_coleta")
        self.assertEqual(neighbors.attrs["temporal_date_column"], "data_coleta")
        self.assertTrue(result.diagnostics["temporal_weight_used"])
        audit = prepare_audit_comparables(neighbors)
        self.assertIn("Rua dos Andradas, 123, sala 404, Centro Histórico", audit.endereco_completo.iloc[0])
        for column in ("fator_recencia", "fator_edificio", "peso_distancia_bruto", "peso_composto_bruto", "idade_observacao_dias", "peso_knn"):
            self.assertIn(column, audit.columns)
        self.assertNotIn("valor_unitario_ajustado", audit.columns)
        pdf = build_inference_report_pdf(estimated_unit_value=result.estimated_unit_value, estimated_total_value=result.estimated_total_value, confidence_score=result.diagnostics["confidence_score"], neighbors=neighbors, purpose="SALA COMERCIAL", area_regime_label="Área privativa", target={"latitude":-30.0305, "longitude":-51.2305}, latitude_column="lat", longitude_column="lon", type_column="tipo", reference_area_column="area", diagnostics=result.diagnostics, fetch_map_tiles=False, generated_at=datetime(2026,10,9,12))
        text = " ".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
        self.assertIn("Rua dos Andradas", text)
        self.assertIn("Fator recência", text)
        self.assertNotIn("VU ajustado", text)

    def test_missing_addresses_are_explicit(self):
        self.assertEqual(comparable_addresses(pd.DataFrame({"area":[10]})).iloc[0], "Endereço não informado")


if __name__ == "__main__":
    unittest.main()

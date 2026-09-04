import unittest

import pandas as pd

import estimador_knn_core_v6120 as core


class RentalMarketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = core.ColumnMapping(
            tipo_informacao="tipo",
            finalidade_oferta="finalidade",
            valor="valor",
            area_construida="area",
            area_privativa=None,
            latitude="latitude",
            longitude="longitude",
            siat_area_total_lote=None,
            testada=None,
        )

    def test_only_rental_offers_are_kept_without_discount(self) -> None:
        source = pd.DataFrame(
            {
                "tipo": [
                    "Oferta Aluguel",
                    "Oferta Aluguel",
                    "Oferta",
                    "Guia ITBI",
                ],
                "finalidade": ["Apartamento"] * 4,
                "valor": [2_000.0, 2_200.0, 500_000.0, 450_000.0],
                "area": [50.0, 55.0, 50.0, 50.0],
                "latitude": [-30.03, -30.04, -30.03, -30.03],
                "longitude": [-51.23, -51.24, -51.23, -51.23],
            }
        )

        result = core.prepare_data(
            df=source,
            mapping=self.mapping,
            selected_purpose="Apartamento",
            value_kind="Valor total",
            reference_area_column="area",
            remove_offer_duplicates=False,
        )

        self.assertEqual(len(result.data), 2)
        self.assertTrue(result.data["_tipo_norm"].eq("oferta aluguel").all())
        self.assertEqual(result.discount, 0.0)
        self.assertTrue(result.data["_fator_ajuste"].eq(1.0).all())
        self.assertTrue(
            result.data["_valor_unitario_original"].equals(
                result.data["_valor_unitario_ajustado"]
            )
        )

    def test_apartment_floor_and_sale_url_are_audited(self) -> None:
        source = pd.DataFrame(
            {
                "tipo": ["Oferta Aluguel"] * 3,
                "finalidade": ["Apartamento"] * 3,
                "valor": [300.0, 2_000.0, 2_100.0],
                "area": [100.0, 50.0, 50.0],
                "latitude": [-30.03, -30.04, -30.05],
                "longitude": [-51.23, -51.24, -51.25],
                "anuncio_website": [
                    "https://portal.example/imovel/aluguel/1",
                    "https://portal.example/imovel/venda/2",
                    "https://portal.example/imovel/aluguel/3",
                ],
            }
        )

        result = core.prepare_data(
            df=source,
            mapping=self.mapping,
            selected_purpose="Apartamento",
            value_kind="Valor total",
            reference_area_column="area",
            remove_offer_duplicates=False,
        )

        self.assertEqual(len(result.data), 1)
        reasons = " | ".join(result.excluded_data["_motivo_exclusao"])
        self.assertIn("piso da finalidade", reasons)
        self.assertIn("URL identifica oferta de venda", reasons)
        self.assertEqual(result.diagnostics["prefilter_sale_url_excluded"], 1)

    def test_calibrated_profile_is_specific_only_when_confirmed(self) -> None:
        apartment = core.calibrated_parameters_for_purpose("Apartamento")
        warehouse = core.calibrated_parameters_for_purpose(
            "Galpão / Depósito"
        )

        self.assertEqual(apartment["profile"], "aluguel_global_temporal_2026_09")
        self.assertEqual(apartment["similarity_weight"], 0.35)
        self.assertEqual(warehouse["profile"], "aluguel_galpao_temporal_2026_09")
        self.assertEqual(warehouse["similarity_weight"], 0.65)
        self.assertEqual(warehouse["location_weight"], 0.35)


if __name__ == "__main__":
    unittest.main()

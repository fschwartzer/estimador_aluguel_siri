import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import estimador_knn_core_v6120 as core
import estimador_knn_schema_v6120 as schema
import geocodificador_porto_alegre as poa_geocoder


class GenericImportTests(unittest.TestCase):
    def test_generic_sale_label_is_canonicalized_as_offer(self) -> None:
        source = pd.DataFrame(
            {
                "Tipo da informação": ["Venda"],
                "Finalidade": ["Casa"],
                "Valor do imóvel (R$)": ["R$ 450.000,00"],
                "Área construída (m²)": ["120 m²"],
            }
        )

        enriched, _ = schema.enrich_known_schemas(source)

        self.assertEqual(
            enriched[schema.DERIVED_TIPO_INFORMACAO].tolist(),
            ["Oferta"],
        )

    def test_scraped_listing_columns_are_recognized(self) -> None:
        source = pd.DataFrame(
            {
                "Tipo": ["Casa", "Casa"],
                "Preço (R$)": ["R$ 450.000,00", "R$ 510.000,00"],
                "Área construída (m²)": ["120 m²", "135 m²"],
                "Área do terreno (m²)": ["300 m²", "360 m²"],
                "Endereço": ["Rua A, 10", "Rua B, 20"],
            }
        )

        enriched, info = schema.enrich_known_schemas(source)

        self.assertFalse(info.siri_detected)
        self.assertTrue(
            enriched[schema.DERIVED_TIPO_INFORMACAO]
            .eq("Oferta Aluguel")
            .all()
        )
        self.assertTrue(
            enriched[schema.DERIVED_FINALIDADE_CRAWLER_NORMALIZADA]
            .eq(schema.FINALIDADE_CASA)
            .all()
        )
        self.assertEqual(
            enriched[schema.DERIVED_AREA_CONSTRUIDA].tolist(),
            [120.0, 135.0],
        )
        self.assertEqual(
            enriched[schema.DERIVED_AREA_LOTE].tolist(),
            [300.0, 360.0],
        )
        self.assertEqual(
            schema.first_existing(source.columns, ["preco"]),
            "Preço (R$)",
        )

        mapping = core.ColumnMapping(
            tipo_informacao=schema.DERIVED_TIPO_INFORMACAO,
            finalidade_oferta=(
                schema.DERIVED_FINALIDADE_CRAWLER_NORMALIZADA
            ),
            valor="Preço (R$)",
            area_construida=schema.DERIVED_AREA_CONSTRUIDA,
            area_privativa=None,
            latitude=None,
            longitude=None,
            siat_area_total_lote=schema.DERIVED_AREA_LOTE,
            testada=None,
        )
        prepared = core.prepare_data(
            df=enriched,
            mapping=mapping,
            selected_purpose=schema.FINALIDADE_CASA,
            value_kind="Valor total",
            reference_area_column=schema.DERIVED_AREA_CONSTRUIDA,
            remove_offer_duplicates=False,
        )
        self.assertEqual(len(prepared.data), 2)
        self.assertEqual(prepared.diagnostics["n_offer"], 2)
        self.assertEqual(
            core.to_numeric(enriched["Preço (R$)"]).tolist(),
            [450_000.0, 510_000.0],
        )

    def test_vivareal_scrape_schema_runs_without_coordinates(self) -> None:
        source = pd.DataFrame(
            {
                "id_vivareal": range(1, 8),
                "tipo_imovel": [
                    "Casa",
                    "Casa de condomínio",
                    "Casa",
                    "Casa",
                    "Sobrado",
                    "Casa",
                    "Casa",
                ],
                "finalidade": ["Venda"] * 7,
                "logradouro": ["Rua Exemplo"] * 7,
                "numero": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, np.nan],
                "bairro_portal": ["Praia de Belas"] * 7,
                "municipio": ["Porto Alegre"] * 7,
                "uf": ["RS"] * 7,
                "area_anunciada_m2": [130, 250, 547, 241, 450, 580, 70],
                "area_privativa_descrita_m2": [
                    130,
                    np.nan,
                    np.nan,
                    241,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "area_construida_descrita_m2": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    580,
                    np.nan,
                ],
                "area_terreno_m2": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    525,
                    np.nan,
                ],
                "area_total_descrita_m2": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    125.11,
                ],
                "valor_oferta_rs": [
                    1_500_000,
                    1_800_000,
                    3_800_000,
                    2_700_000,
                    1_600_000,
                    3_000_000,
                    320_000,
                ],
                "data_extracao": pd.to_datetime(["2026-08-20"] * 7),
                "url_anuncio": [f"https://example.test/{i}" for i in range(7)],
            }
        )

        enriched, info = schema.enrich_known_schemas(source)

        self.assertFalse(info.siri_detected)
        self.assertEqual(
            enriched[schema.DERIVED_AREA_CONSTRUIDA].tolist(),
            [130.0, 250.0, 547.0, 241.0, 450.0, 580.0, 70.0],
        )
        self.assertEqual(
            int(enriched[schema.DERIVED_AREA_LOTE].notna().sum()),
            1,
        )

        mapping = core.ColumnMapping(
            tipo_informacao=schema.DERIVED_TIPO_INFORMACAO,
            finalidade_oferta=(
                schema.DERIVED_FINALIDADE_CRAWLER_NORMALIZADA
            ),
            valor="valor_oferta_rs",
            area_construida=schema.DERIVED_AREA_CONSTRUIDA,
            area_privativa=schema.DERIVED_AREA_PRIVATIVA,
            latitude=None,
            longitude=None,
            siat_area_total_lote=schema.DERIVED_AREA_LOTE,
            testada=None,
        )
        prepared = core.prepare_data(
            df=enriched,
            mapping=mapping,
            selected_purpose=schema.FINALIDADE_CASA,
            value_kind="Valor total",
            reference_area_column=schema.DERIVED_AREA_CONSTRUIDA,
            remove_offer_duplicates=True,
            duplicate_date_column="data_extracao",
            duplicate_identifier_columns=("id_vivareal", "url_anuncio"),
            duplicate_value_column="valor_oferta_rs",
            floor_purpose=schema.FINALIDADE_CASA,
        )
        parameters = core.calibrated_parameters_for_purpose(
            schema.FINALIDADE_CASA
        )
        target = {
            "area_construida": 250.0,
            "area_privativa": None,
            "siat_area_total_lote": None,
            "testada": None,
            "ano_construcao": None,
            "latitude": None,
            "longitude": None,
            schema.DERIVED_AREA_CONSTRUIDA: 250.0,
        }
        estimate = core.estimate_knn(
            preparation=prepared,
            mapping=mapping,
            target=target,
            reference_area_column=schema.DERIVED_AREA_CONSTRUIDA,
            min_k=int(parameters["min_k"]),
            max_k=int(parameters["max_k"]),
            min_effective_neighbors=float(
                parameters["min_effective_neighbors"]
            ),
            similarity_weight=float(parameters["similarity_weight"]),
            distance_power=float(parameters["distance_power"]),
            max_individual_weight=float(parameters["max_individual_weight"]),
            robust_mad_threshold=float(parameters["robust_mad_threshold"]),
        )

        self.assertEqual(len(prepared.data), 7)
        self.assertEqual(estimate.diagnostics["k_used"], 7)
        self.assertFalse(estimate.diagnostics["location_used"])
        self.assertEqual(estimate.diagnostics["similarity_weight"], 1.0)
        self.assertGreater(estimate.estimated_total_value, 0)

    def test_address_parser_accepts_comma_and_plain_number(self) -> None:
        self.assertEqual(
            poa_geocoder.parse_street_and_number(
                "Av. Borges de Medeiros, 2244, Porto Alegre"
            ),
            ("Av. Borges de Medeiros", 2244),
        )
        self.assertEqual(
            poa_geocoder.parse_street_and_number("Rua dos Andradas 1000"),
            ("Rua dos Andradas", 1000),
        )

    def test_axis_interpolation_respects_address_fraction(self) -> None:
        shape = SimpleNamespace(
            points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
            parts=[0],
        )
        self.assertEqual(
            poa_geocoder._interpolate_shape(shape, 0.75),
            (10.0, 5.0),
        )


class OptionalLocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = core.ColumnMapping(
            tipo_informacao="tipo",
            finalidade_oferta="finalidade",
            valor="valor",
            area_construida="area_construida",
            area_privativa=None,
            latitude=None,
            longitude=None,
            siat_area_total_lote="area_lote",
            testada=None,
            ano_construcao=None,
        )

    def test_built_property_can_use_land_area_as_extra_feature(self) -> None:
        target = {
            "area_construida": 120.0,
            "area_privativa": None,
            "siat_area_total_lote": 300.0,
            "testada": None,
            "ano_construcao": None,
            "latitude": None,
            "longitude": None,
        }

        features, territorial = core._resolve_features(
            self.mapping,
            target,
            territorial=False,
        )

        self.assertFalse(territorial)
        self.assertEqual(
            features,
            [
                ("area_construida", "area_construida"),
                ("siat_area_total_lote", "area_lote"),
            ],
        )

    def test_estimate_without_coordinates_uses_physical_distance_only(self) -> None:
        data = pd.DataFrame(
            {
                "tipo": ["oferta"] * 4,
                "finalidade": ["casa / residencia"] * 4,
                "valor": [400_000.0, 430_000.0, 470_000.0, 520_000.0],
                "area_construida": [100.0, 115.0, 130.0, 150.0],
                "area_lote": [250.0, 290.0, 340.0, 420.0],
                "_valor_unitario_ajustado": [4000.0, 3739.0, 3615.0, 3467.0],
            }
        )
        preparation = core.PreparationResult(
            data=data,
            discount=0.10,
            diagnostics={"purpose": "casa / residencia"},
            excluded_data=data.iloc[0:0].copy(),
            flagged_data=data.iloc[0:0].copy(),
        )
        target = {
            "area_construida": 120.0,
            "area_privativa": None,
            "siat_area_total_lote": 300.0,
            "testada": None,
            "ano_construcao": None,
            "latitude": None,
            "longitude": None,
        }

        result = core.estimate_knn(
            preparation=preparation,
            mapping=self.mapping,
            target=target,
            reference_area_column="area_construida",
            min_k=2,
            max_k=4,
            min_effective_neighbors=1.0,
            similarity_weight=0.85,
            distance_power=1.0,
            max_individual_weight=1.0,
            robust_mad_threshold=2.0,
            territorial=False,
        )

        self.assertFalse(result.diagnostics["location_used"])
        self.assertEqual(result.diagnostics["location_weight"], 0.0)
        self.assertEqual(result.diagnostics["similarity_weight"], 1.0)
        self.assertEqual(len(result.neighbors), 2)
        self.assertTrue(
            np.isnan(result.diagnostics["selected_geo_median_km"])
        )
        self.assertIn("area_lote", result.active_features)

    def test_invalid_coordinates_are_audited_when_location_is_used(self) -> None:
        mapping = core.ColumnMapping(
            **{
                **self.mapping.__dict__,
                "latitude": "latitude",
                "longitude": "longitude",
            }
        )
        data = pd.DataFrame(
            {
                "tipo": ["oferta"] * 4,
                "finalidade": ["casa / residencia"] * 4,
                "valor": [400_000.0, 430_000.0, 470_000.0, 520_000.0],
                "area_construida": [100.0, 115.0, 130.0, 150.0],
                "area_lote": [250.0, 290.0, 340.0, 420.0],
                "latitude": [-30.03, -30.031, -30.032, np.nan],
                "longitude": [-51.23, -51.231, -51.232, np.nan],
                "_valor_unitario_ajustado": [4000.0, 3739.0, 3615.0, 3467.0],
            }
        )
        preparation = core.PreparationResult(
            data=data,
            discount=0.10,
            diagnostics={"purpose": "casa / residencia"},
            excluded_data=data.iloc[0:0].copy(),
            flagged_data=data.iloc[0:0].copy(),
        )
        target = {
            "area_construida": 120.0,
            "area_privativa": None,
            "siat_area_total_lote": 300.0,
            "testada": None,
            "ano_construcao": None,
            "latitude": -30.031,
            "longitude": -51.231,
        }

        result = core.estimate_knn(
            preparation=preparation,
            mapping=mapping,
            target=target,
            reference_area_column="area_construida",
            min_k=2,
            max_k=3,
            min_effective_neighbors=1.0,
            similarity_weight=0.85,
            distance_power=1.0,
            max_individual_weight=1.0,
            robust_mad_threshold=2.0,
            territorial=False,
        )

        self.assertTrue(result.diagnostics["location_used"])
        self.assertEqual(result.diagnostics["location_invalid_excluded"], 1)
        self.assertIn(
            "validação da localização",
            result.local_excluded_data["_etapa_controle"].tolist(),
        )

    def test_available_sample_coordinates_can_be_explicitly_ignored(self) -> None:
        mapping = core.ColumnMapping(
            **{
                **self.mapping.__dict__,
                "latitude": "latitude",
                "longitude": "longitude",
            }
        )
        data = pd.DataFrame(
            {
                "tipo": ["oferta"] * 4,
                "finalidade": ["casa / residencia"] * 4,
                "valor": [400_000.0, 430_000.0, 470_000.0, 520_000.0],
                "area_construida": [100.0, 115.0, 130.0, 150.0],
                "area_lote": [250.0, 290.0, 340.0, 420.0],
                "latitude": [-30.03, np.nan, -30.032, np.nan],
                "longitude": [-51.23, np.nan, -51.232, np.nan],
                "_valor_unitario_ajustado": [4000.0, 3739.0, 3615.0, 3467.0],
            }
        )
        preparation = core.PreparationResult(
            data=data,
            discount=0.10,
            diagnostics={"purpose": "casa / residencia"},
            excluded_data=data.iloc[0:0].copy(),
            flagged_data=data.iloc[0:0].copy(),
        )
        target = {
            "area_construida": 120.0,
            "area_privativa": None,
            "siat_area_total_lote": 300.0,
            "testada": None,
            "ano_construcao": None,
            "latitude": None,
            "longitude": None,
        }

        result = core.estimate_knn(
            preparation=preparation,
            mapping=mapping,
            target=target,
            reference_area_column="area_construida",
            min_k=2,
            max_k=4,
            min_effective_neighbors=1.0,
            similarity_weight=0.85,
            distance_power=1.0,
            max_individual_weight=1.0,
            robust_mad_threshold=2.0,
        )

        self.assertFalse(result.diagnostics["location_used"])
        self.assertEqual(result.diagnostics["location_invalid_excluded"], 0)
        self.assertEqual(
            result.diagnostics["n_candidates_after_physical_validation"],
            4,
        )


if __name__ == "__main__":
    unittest.main()

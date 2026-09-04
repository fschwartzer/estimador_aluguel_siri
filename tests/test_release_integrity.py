import ast
from pathlib import Path
import unittest

import estimador_knn_core_v6120 as core
import estimador_knn_schema_v6120 as schema


ROOT = Path(__file__).resolve().parents[1]


def module_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            result[node.targets[0].id] = node.value.value
    return result


class ReleaseIntegrityTests(unittest.TestCase):
    def test_app_loads_the_published_core_and_schema(self) -> None:
        constants = module_constants(ROOT / "app.py")

        self.assertEqual(constants["APP_EDITION"], "1.0.0")
        self.assertEqual(constants["CORE_VERSION"], "6.13.0")
        self.assertEqual(
            constants["CORE_MODULE_FILE"],
            "estimador_knn_core_v6120.py",
        )
        self.assertEqual(
            constants["SCHEMA_MODULE_FILE"],
            "estimador_knn_schema_v6120.py",
        )
        self.assertTrue((ROOT / constants["CORE_MODULE_FILE"]).is_file())
        self.assertTrue((ROOT / constants["SCHEMA_MODULE_FILE"]).is_file())
        self.assertEqual(core.MODULE_API_VERSION, constants["CORE_VERSION"])
        self.assertEqual(schema.MODULE_API_VERSION, constants["CORE_VERSION"])
        self.assertEqual(core.MODULE_BUILD_ID, constants["MODULE_BUILD_ID"])
        self.assertEqual(schema.MODULE_BUILD_ID, constants["MODULE_BUILD_ID"])

    def test_geocoding_runtime_is_published_with_its_dependencies(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for dependency in ("geopy", "pyproj", "pyshp", "rapidfuzz"):
            self.assertIn(dependency, requirements.casefold())
        self.assertTrue((ROOT / "geocodificador_porto_alegre.py").is_file())
        self.assertTrue(hasattr(schema, "DERIVED_TIPO_INFORMACAO"))

    def test_pdf_runtime_is_published_with_its_dependencies(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for dependency in ("reportlab", "pillow", "mapbox-vector-tile"):
            self.assertIn(dependency, requirements.casefold())
        self.assertTrue((ROOT / "siri_alugueis_pdf_report.py").is_file())

    def test_app_maps_siat_year_to_the_new_mapping_field(self) -> None:
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        build_mapping = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_mapping"
        )
        literals = {
            node.value
            for node in ast.walk(build_mapping)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        mapping_keywords = {
            keyword.arg
            for node in ast.walk(build_mapping)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ColumnMapping"
            for keyword in node.keywords
        }

        self.assertIn("siat_ano", literals)
        self.assertIn("ano_construcao", mapping_keywords)

    def test_app_recognizes_vivareal_scrape_columns(self) -> None:
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

        mapping_literals = {
            node.value
            for node in ast.walk(functions["build_mapping"])
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        address_literals = {
            node.value
            for node in ast.walk(functions["detect_address_columns"])
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }

        self.assertTrue(
            {
                "valor_oferta_rs",
                "area_anunciada_m2",
                "area_privativa_descrita_m2",
                "area_construida_descrita_m2",
                "area_terreno_m2",
            }.issubset(mapping_literals)
        )
        self.assertTrue(
            {
                "logradouro",
                "numero",
                "bairro_portal",
                "municipio",
                "uf",
            }.issubset(address_literals)
        )

    def test_legacy_runtime_modules_are_outside_the_root(self) -> None:
        active = {
            "estimador_knn_core_v6120.py",
            "estimador_knn_schema_v6120.py",
        }
        root_versioned_modules = {
            path.name
            for path in ROOT.glob("estimador_knn_*_v*.py")
        }

        self.assertEqual(root_versioned_modules, active)
        self.assertTrue((ROOT / "archive" / "legacy" / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()

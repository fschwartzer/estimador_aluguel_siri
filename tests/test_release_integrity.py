import ast
import hashlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

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

        self.assertEqual(constants["APP_EDITION"], "1.1.0")
        self.assertEqual(constants["CORE_VERSION"], "6.14.0")
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

    def test_pdf_uses_current_source_when_an_old_import_is_cached(self) -> None:
        # Simula o processo hospedado sobrevivendo à atualização dos arquivos.
        stale = types.ModuleType("siri_alugueis_pdf_report")
        def legacy_pdf(*, estimated_unit_value, estimated_total_value,
                       confidence_score, neighbors, purpose, area_regime_label,
                       target, latitude_column, longitude_column, type_column,
                       reference_area_column, logo_path=None):
            return b"relatorio antigo"
        stale.build_inference_report_pdf = legacy_pdf
        stale.calculate_comparable_cod = lambda neighbors: -1
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        setup = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "siri_alugueis_pdf_report":
                setup.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in ("_load_exact_source_module", "inference_report_pdf"):
                node.decorator_list = []
                setup.append(node)
            elif isinstance(node, ast.Try) and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "_load_exact_source_module" for child in ast.walk(node)):
                setup.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ("CORE_MODULE_FILE", "SCHEMA_MODULE_FILE", "PDF_MODULE_FILE", "build_inference_report_pdf", "calculate_comparable_cod") for target in node.targets):
                setup.append(node)
        scope = {"__file__":str(ROOT / "app.py"), "Path":Path, "hashlib":hashlib, "sys":sys, "types":types, "pd":pd}
        neighbors = pd.DataFrame({"area":[30.,40.], "_valor_unitario_original":[30.,40.], "_valor_unitario_ajustado":[30.,40.], "_valor_unitario_robusto":[30.,40.], "_peso_knn":[.5,.5]})
        with patch.dict(sys.modules, {"siri_alugueis_pdf_report":stale}):
            exec(compile(ast.Module(body=setup, type_ignores=[]), "app.py", "exec"), scope)
            pdf = scope["inference_report_pdf"](
                estimated_unit_value=35., estimated_total_value=1050., confidence_score=70.,
                neighbors=neighbors, purpose="SALA COMERCIAL", area_regime_label="Área privativa",
                target={}, latitude_column=None, longitude_column=None,
                type_column=None, reference_area_column="area", diagnostics={"k_used":2},
            )
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(scope["calculate_comparable_cod"](neighbors), 0)

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

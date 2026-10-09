"""Extração somente leitura da pesquisa SIRI para cache local não versionado."""
from pathlib import Path
import argparse
import sys
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_source(path: Path) -> pd.DataFrame:
    # XML incremental evita carregar as imagens/fotos e estilos da pesquisa.
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(e.itertext()) for e in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rows = []
        for _, element in ET.iterparse(archive.open("xl/worksheets/sheet1.xml"), events=("end",)):
            if element.tag != NS + "row":
                continue
            values = {}
            for cell in element:
                coordinate = cell.attrib.get("r", "")
                letters = "".join(c for c in coordinate if c.isalpha())
                column = 0
                for letter in letters:
                    column = column * 26 + ord(letter) - 64
                node = cell.find(NS + "v")
                value = node.text if node is not None else None
                if value is not None:
                    if cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.attrib.get("t") != "str":
                        value = float(value)
                values[column - 1] = value
            rows.append(values)
            element.clear()
        headers = rows.pop(0)
        result = pd.DataFrame(rows).rename(columns=headers)
        result["_row_excel"] = range(2, len(result) + 2)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--enriched", type=Path, help="Cache normalizado pelo mesmo schema do aplicativo")
    args = parser.parse_args()
    source, destination = args.source, args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    df = read_source(source)
    df.to_pickle(destination)
    if args.enriched:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from estimador_knn_schema_v6120 import enrich_known_schemas
        enriched, schema_info = enrich_known_schemas(df)
        args.enriched.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_pickle(args.enriched)
        print("normalized", len(enriched), "schema", schema_info, flush=True)
    print("rows", len(df), "columns", list(df.columns), flush=True)
    for column in ("crawler_tipo_imovel_normalizado", "siat_finalidade_descricao", "tipo_informacao", "data_encaminhamento", "rdata"):
        print(column, df[column].value_counts(dropna=False).head(30).to_dict(), flush=True)

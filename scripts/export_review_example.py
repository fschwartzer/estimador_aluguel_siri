"""Exemplo real do estimador e das mesmas funções de exportação do aplicativo."""
import argparse
import ast
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import sys

import pandas as pd

from calibrate_rentals import ROOT, PARAM_KEYS, Fold, property_groups, core, schema
sys.path.insert(0, str(ROOT))
from rental_audit import comparable_addresses, prepare_audit_comparables
from siri_alugueis_pdf_report import build_inference_report_pdf


def application_excel_export():
    tree = ast.parse((ROOT/"app.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ("make_unique_column_names","dataframe_to_excel")]
    scope = {"pd":pd,"BytesIO":BytesIO}
    exec(compile(ast.Module(body=selected,type_ignores=[]), "app.py", "exec"),scope)
    return scope["dataframe_to_excel"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    calibration = json.loads((ROOT/"artifacts/calibration/aluguel_parametros_2026_10.json").read_text(encoding="utf-8"))
    data = pd.read_pickle(args.cache).reset_index(drop=True)
    data["_date"] = core._parse_registration_dates(data.data_encaminhamento)
    data["_group"] = property_groups(data)
    purpose = "SALA COMERCIAL"
    fold = Fold(data,purpose,"2026-08-01","2026-09-30",200)
    config = calibration["purpose_parameters"][core.normalize_text(purpose)]
    # Caso de preço mediano, definido sem buscar o menor erro do modelo.
    totals = fold.test.valor_oferta
    index = (totals-totals.median()).abs().idxmin()
    row = fold.test.loc[index]
    target = {**fold.targets[index],"endereco":comparable_addresses(fold.test.loc[[index]]).iloc[0]}
    preparation = fold.preparation(config["unit_value_floor"])
    estimate = core.estimate_knn(preparation,fold.mapping,target,fold.area,temporal_reference_date=row._date,territorial=fold.territorial,**{k:config[k] for k in PARAM_KEYS})
    diagnostics = {**preparation.diagnostics,**estimate.diagnostics,"perfil":config["profile"],"calibration_status":config["calibration_status"],"aluguel_observado_exemplo":float(row.valor_oferta),"endereco_avaliando":target["endereco"],"nota_exemplo":"Exemplo de validação retrospectiva, comparáveis anteriores a agosto de 2026; não representa uma avaliação solicitada."}
    audit = prepare_audit_comparables(estimate.neighbors)
    primary = ["linha_excel","endereco_completo","valor_unitario","valor_unitario_robusto","peso_knn","fator_recencia","fator_edificio","peso_distancia_bruto","peso_composto_bruto","idade_observacao_dias","contribuicao_valor_unitario","distancia_geografica_km","data_encaminhamento","siat_logradouro","siat_numero","siat_complemento","siat_bairro",fold.area]
    audit = audit[[c for c in primary if c in audit.columns]]
    excluded = pd.concat([preparation.excluded_data,estimate.local_excluded_data],ignore_index=True,sort=False)
    control = ["_row_excel","siat_logradouro","siat_numero","siat_bairro","valor_oferta",fold.area,"_valor_unitario_original","_etapa_controle","_motivo_exclusao","_motivo_alerta"]
    excluded = excluded[[c for c in control if c in excluded.columns]]
    flagged = preparation.flagged_data[[c for c in control if c in preparation.flagged_data.columns]]
    excel = application_excel_export()(audit,diagnostics,excluded,flagged)
    (args.output/"auditoria_exemplo.xlsx").write_bytes(excel)
    pdf = build_inference_report_pdf(estimated_unit_value=estimate.estimated_unit_value,estimated_total_value=estimate.estimated_total_value,confidence_score=estimate.diagnostics["confidence_score"],neighbors=estimate.neighbors,purpose=purpose,area_regime_label="Área privativa",target=target,latitude_column=fold.mapping.latitude,longitude_column=fold.mapping.longitude,type_column=fold.mapping.tipo_informacao,reference_area_column=fold.area,logo_path=ROOT/"static/siri_alugueis_header.png",generated_at=datetime(2026,10,9,12),fetch_map_tiles=False,diagnostics=diagnostics)
    (args.output/"relatorio_sintetico_exemplo.pdf").write_bytes(pdf)
    with pd.ExcelFile(BytesIO(excel)) as exported:
        comparisons = pd.read_excel(exported,sheet_name="Comparaveis")
        assert abs(comparisons.peso_knn.sum()-1)<1e-12
        assert abs(comparisons.contribuicao_valor_unitario.sum()-estimate.estimated_unit_value)<1e-10
        assert comparisons.endereco_completo.notna().all()
        assert "valor_unitario_ajustado" not in comparisons.columns
        assert comparisons.fator_recencia.between(0,1).all()
    print(json.dumps({"estimated":estimate.estimated_total_value,"observed":float(row.valor_oferta),"k":len(audit),"output":str(args.output)},ensure_ascii=False))

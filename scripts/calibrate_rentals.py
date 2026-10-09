"""Calibra o estimador de produção, com validação temporal de promoção.

Uso: python scripts/calibrate_rentals.py --cache tmp/enriched.pkl --source arquivo.xlsx
Os dados individuais ficam em tmp/ (ignorado pelo Git); somente parâmetros e
diagnósticos agregados são publicados. Não ajusta o modelo ao preço contratado.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import estimador_knn_core_v6120 as core
import estimador_knn_schema_v6120 as schema

LOGGER = logging.getLogger("calibration")
PARAM_KEYS = ("min_k", "max_k", "min_effective_neighbors", "similarity_weight", "distance_power", "max_individual_weight", "robust_mad_threshold", "temporal_weight_enabled", "temporal_half_life_days", "temporal_min_factor", "building_bonus", "local_filter_enabled")
BASE = dict(min_k=12, max_k=25, min_effective_neighbors=10., similarity_weight=.35, distance_power=.75, max_individual_weight=.25, robust_mad_threshold=1.5, temporal_weight_enabled=True, temporal_half_life_days=180., temporal_min_factor=.35, building_bonus=2., local_filter_enabled=True)


def property_groups(data: pd.DataFrame) -> np.ndarray:
    """Componentes conectados por inscrição, URL ou origem+código, sem preço."""
    parent = np.arange(len(data))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    columns = [data[c].map(core.normalize_text).fillna("").to_numpy() for c in ("siat_inscricao", "anuncio_website", "origem_registro", "imobiliaria_codigo_anuncio")]
    ignored = {"", "0", "0.0", "nan", "none", "<na>"}
    for i, (registration, url, origin, listing) in enumerate(zip(*columns)):
        tokens = []
        if registration not in ignored:
            tokens.append("registration:" + registration)
        if url not in ignored:
            tokens.append("url:" + url)
        if origin not in ignored and listing not in ignored:
            tokens.append("listing:" + origin + ":" + listing)
        for token in tokens:
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    return np.array([find(i) for i in range(len(data))])


def ratio_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}
    actual = frame.actual.to_numpy(float)
    predicted = frame.predicted.to_numpy(float)
    ratios = predicted / actual
    ape = abs(ratios - 1)
    median = float(np.median(ratios))
    total_actual = actual * frame.area.to_numpy(float)
    total_predicted = predicted * frame.area.to_numpy(float)
    return dict(n=len(frame), mdape=float(np.median(ape)), p90_ape=float(np.quantile(ape, .9)), median_ratio=median, cod=float(np.mean(abs(ratios - median)) / median * 100), prd=float(np.mean(ratios) / (total_predicted.sum() / total_actual.sum())), prd_unit=float(np.mean(ratios) / (predicted.sum() / actual.sum())), mean_absolute_error_brl_month=float(np.mean(abs(total_predicted-total_actual))))


def score(metrics: dict) -> float:
    # Métrica de precisão com penalidades explícitas para cauda e viés.
    if metrics.get("n", 0) < 10:
        return float("inf")
    return metrics["mdape"] + .15 * metrics["p90_ape"] + .5 * abs(metrics["median_ratio"] - 1) + .2 * metrics["cod"] / 100 + .3 * abs(metrics["prd"] - 1)


class Fold:
    def __init__(self, data, purpose, start, end, limit, seed=20261009):
        self.purpose = purpose
        self.territorial = purpose in ("TERRENO", "GLEBA")
        preference = schema.reference_area_preference(purpose)
        self.area = schema.DERIVED_AREA_LOTE if self.territorial else (schema.DERIVED_AREA_CONSTRUIDA if preference == "construida" else schema.DERIVED_AREA_PRIVATIVA)
        own = data.loc[data[schema.DERIVED_FINALIDADE_CRAWLER_NORMALIZADA].eq(purpose)].copy()
        if not self.territorial:
            historical = own.loc[own._date.lt(start)]
            private_count = core.to_numeric(historical[schema.DERIVED_AREA_PRIVATIVA]).gt(0).sum()
            if preference == "privativa_ou_construida" and private_count < len(historical)*.5:
                self.area = schema.DERIVED_AREA_CONSTRUIDA
            if core.to_numeric(historical[self.area]).gt(0).sum() < 12:
                alternative = schema.DERIVED_AREA_CONSTRUIDA if self.area == schema.DERIVED_AREA_PRIVATIVA else schema.DERIVED_AREA_PRIVATIVA
                if core.to_numeric(historical[alternative]).gt(0).sum() >= 12:
                    self.area = alternative
        numeric = ["valor_oferta", self.area, "siat_latitude", "siat_longitude", "siat_ano", schema.DERIVED_TESTADA]
        for column in numeric:
            own[column] = core.to_numeric(own[column])
        eligible = own.loc[own._date.ge(start) & own._date.le(end)].sort_values("_date").drop_duplicates("_group", keep="last")
        eligible = eligible.loc[eligible.valor_oferta.gt(core.PREFILTER_SYMBOLIC_VALUE_MAX) & eligible[self.area].gt(0) & eligible.siat_latitude.between(-90, 90) & eligible.siat_longitude.between(-180, 180)]
        if self.territorial:
            eligible = eligible.loc[eligible[schema.DERIVED_TESTADA].gt(0)]
        self.eligible_count = len(eligible)
        self.test = eligible.sample(min(limit, len(eligible)), random_state=seed).sort_index()
        # Retira todos os imóveis elegíveis no período, inclusive não amostrados.
        self.train = own.loc[own._date.lt(start) & ~own._group.isin(eligible._group)].copy()
        if set(self.train._group) & set(eligible._group):
            raise RuntimeError("Vazamento: imóvel da validação presente no treino.")
        if not self.train.empty and not self.train._date.lt(start).all():
            raise RuntimeError("Vazamento: observação futura presente no treino.")
        self.mapping = core.ColumnMapping(schema.DERIVED_TIPO_INFORMACAO, schema.DERIVED_FINALIDADE_CRAWLER_NORMALIZADA, "valor_oferta", None if self.territorial or self.area == schema.DERIVED_AREA_PRIVATIVA else self.area, self.area if not self.territorial and self.area == schema.DERIVED_AREA_PRIVATIVA else None, "siat_latitude", "siat_longitude", self.area if self.territorial else None, schema.DERIVED_TESTADA if self.territorial else None, "siat_ano" if not self.territorial else None, "data_encaminhamento")
        self.preparations = {}
        self.contexts = {}
        self.targets = {}
        for index, row in self.test.iterrows():
            key = "siat_area_total_lote" if self.territorial else ("area_privativa" if self.mapping.area_privativa else "area_construida")
            target = {key: float(row[self.area]), "latitude": float(row.siat_latitude), "longitude": float(row.siat_longitude)}
            if self.territorial:
                target["testada"] = float(row[schema.DERIVED_TESTADA])
            elif core.is_valid_construction_year(row.siat_ano):
                target["ano_construcao"] = float(row.siat_ano)
            self.targets[index] = target
        LOGGER.info("%s %s: treino=%s alvos=%s elegíveis=%s área=%s", purpose, start, len(self.train), len(self.test), len(eligible), self.area)

    def preparation(self, floor):
        if floor not in self.preparations:
            self.preparations[floor] = core.prepare_data(self.train, self.mapping, self.purpose, "Valor total", self.area, remove_offer_duplicates=True, duplicate_date_column="data_encaminhamento", duplicate_identifier_columns=("anuncio_website", "imobiliaria_codigo_anuncio"), duplicate_registration_column="siat_inscricao", duplicate_value_column="valor_oferta", unit_value_floor=floor, conflict_column=schema.DERIVED_CONFLITO_TIPOLOGICO, minimum_without_conflict=12)
        return self.preparations[floor]

    def context(self, index, config):
        key = (index, config["unit_value_floor"], config["similarity_weight"], config["local_filter_enabled"], config["min_k"], config["max_k"])
        if key not in self.contexts:
            preparation = self.preparation(config["unit_value_floor"])
            target = self.targets[index]
            active, _ = core._resolve_features(self.mapping, target, self.territorial)
            candidates, _ = core._valid_candidates(preparation.data, self.mapping, active, use_location=True)
            if config["local_filter_enabled"]:
                candidates, _, _ = core._local_lower_tail_filter(candidates, self.mapping, active, target, config["similarity_weight"], True, config["min_k"], config["max_k"], self.purpose)
            distances, _, geo, _ = core._distance_profile(candidates, self.mapping, active, target, config["similarity_weight"], True)
            positions = np.argsort(distances, kind="mergesort")[:config["max_k"]]
            ordered = candidates.iloc[positions]
            dates = core._parse_registration_dates(ordered.data_encaminhamento)
            self.contexts[key] = (distances[positions], geo[positions], ordered._valor_unitario_ajustado.to_numpy(float), dates)
        return self.contexts[key]

    def evaluate(self, config, verify=False):
        records = []
        for index, row in self.test.iterrows():
            try:
                distances, geo, values, dates = self.context(index, config)
                if len(values) < 2:
                    raise ValueError("candidatos insuficientes")
                reference = row._date
                age = (reference - dates).dt.total_seconds().to_numpy() / 86400
                temporal = np.maximum(np.power(2., -age / config["temporal_half_life_days"]), config["temporal_min_factor"]) if config["temporal_weight_enabled"] else np.ones(len(age))
                temporal = np.nan_to_num(temporal, nan=config["temporal_min_factor"])
                geo_m = geo * 1000.
                bonus = np.ones(len(geo))
                bonus[geo_m <= 30.] = config["building_bonus"]
                transition = (geo_m > 30.) & (geo_m <= 50.)
                bonus[transition] = 1 + (config["building_bonus"]-1) * (50.-geo_m[transition])/(50.-30.)
                raw = (1. / np.power(distances + 1e-9, config["distance_power"])) * temporal * bonus
                for k in range(min(config["min_k"], len(values)), len(values)+1):
                    weights, _ = core._cap_and_normalize_weights(raw[:k], config["max_individual_weight"])
                    if 1 / np.square(weights).sum() >= config["min_effective_neighbors"]:
                        break
                prediction, _, _ = core._robust_weighted_mean(values[:k], weights, config["robust_mad_threshold"])
                if verify:
                    result = core.estimate_knn(self.preparation(config["unit_value_floor"]), self.mapping, self.targets[index], self.area, territorial=self.territorial, temporal_reference_date=reference, **{key:config[key] for key in PARAM_KEYS})
                    try:
                        np.testing.assert_allclose(prediction, result.estimated_unit_value, rtol=1e-10)
                    except AssertionError:
                        LOGGER.error("Divergência produção alvo=%s config=%s dist=%s pesos=%s vs=%s",index,config,distances[:k],weights,result.neighbors[["_distancia_composta","_fator_temporal","_peso_knn"]].to_dict("list"))
                        raise
                records.append(dict(index=int(index), actual=float(row.valor_oferta/row[self.area]), predicted=prediction, area=float(row[self.area]), group=int(row._group), date=str(reference.date()), bairro=str(row.siat_bairro), latitude=float(row.siat_latitude), longitude=float(row.siat_longitude)))
            except ValueError as error:
                LOGGER.warning("Falha %s %s: %s", self.purpose, index, error)
        frame = pd.DataFrame(records)
        metrics = ratio_metrics(frame)
        metrics["failed"] = len(self.test) - len(frame)
        return metrics, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--purposes", nargs="+")
    args = parser.parse_args()
    output = ROOT / "artifacts/calibration"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(ROOT/"tmp/calibration.log", encoding="utf-8")])
    data = pd.read_pickle(args.cache).reset_index(drop=True)
    data["_date"] = core._parse_registration_dates(data.data_encaminhamento)
    data["_group"] = property_groups(data)
    purposes = [p for p in data[schema.DERIVED_FINALIDADE_CRAWLER_NORMALIZADA].unique() if p and schema.natureza_uso_normalizada(p) != "RESIDENCIAL"]
    purposes = sorted(purposes)
    report = dict(schema_version=2, source=args.source.name, source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), source_rows=len(data), seed=20261009, tuning_period=["2026-03-01", "2026-03-31"], development_confirmation_period=["2025-09-01", "2025-09-30"], test_period=["2026-08-01", "2026-09-30"], grouping="componentes conectados: inscrição SIAT, URL, origem+código de anúncio; sem valor", purpose_parameters={}, evaluations={}, candidate_results=[])
    if args.purposes:
        report = json.loads((output/"aluguel_parametros_2026_10.json").read_text(encoding="utf-8"))
        purposes = [p for p in purposes if p in args.purposes]
        report["candidate_results"] = [r for r in report["candidate_results"] if r["purpose"] not in purposes]
    all_predictions = []
    for purpose in purposes:
        tuning = Fold(data, purpose, "2026-03-01", "2026-03-31", args.limit)
        old_floor = core.PURPOSE_UNIT_VALUE_FLOORS.get(core.normalize_text(purpose), 0.)
        legacy = core.CALIBRATED_PURPOSE_PARAMETERS.get(core.normalize_text(purpose),core.CALIBRATED_GLOBAL_PARAMETERS)
        baseline = {**BASE, **{k:v for k,v in legacy.items() if k in PARAM_KEYS}, "unit_value_floor":old_floor}
        if len(tuning.test) < 35 or len(tuning.train) < 80:
            report["evaluations"][purpose] = dict(status="sem suporte para calibração", tuning_n=len(tuning.test), tuning_eligible=tuning.eligible_count, train_rows=len(tuning.train), retained="perfil legado; recência 180 dias/0,35 não validada nesta finalidade")
            continue
        base_metric, _ = tuning.evaluate(baseline)
        candidates = [baseline]
        for pair, sim, power, mad, local in itertools.product([(6,15,5.), (12,25,10.), (20,40,16.)], [.2,.5,.8], [.35,.75], [1.5,3.], [True,False]):
            candidates.append({**baseline, "min_k":pair[0], "max_k":pair[1], "min_effective_neighbors":pair[2], "similarity_weight":sim, "distance_power":power, "robust_mad_threshold":mad, "local_filter_enabled":local})
        ranked = []
        for config in candidates:
            metric, _ = tuning.evaluate(config)
            ranked.append((score(metric), config, metric))
        _, winner, _ = min(ranked, key=lambda x:x[0])
        # Segunda etapa: sensibilidade aos cortes, limite individual e bônus.
        for floor, cap, bonus in itertools.product([0., old_floor/2, old_floor], [.15,.25], [1.,2.]):
            config = {**winner, "unit_value_floor":floor, "max_individual_weight":cap, "building_bonus":bonus}
            metric, _ = tuning.evaluate(config)
            ranked.append((score(metric), config, metric))
        _, structural, _ = min(ranked, key=lambda x:x[0])
        # Confirmação de desenvolvimento anterior, independente do teste final.
        confirmation = Fold(data, purpose, "2025-09-01", "2025-09-30", args.limit)
        confirmation_baseline, _ = confirmation.evaluate(baseline)
        confirmation_selected = None
        if len(confirmation.test) >= 35:
            shortlist = sorted(ranked, key=lambda x:x[0])[:5] + [(score(base_metric), baseline, base_metric)]
            checked = []
            for tuning_score, config, metric in shortlist:
                previous_metric, _ = confirmation.evaluate(config)
                # Normaliza por fold, para um período mais difícil não dominar.
                combined = tuning_score / score(base_metric) + score(previous_metric) / score(confirmation_baseline)
                checked.append((combined, config, previous_metric))
            _, structural, confirmation_selected = min(checked, key=lambda x:x[0])
        # Recência com e sem ponderação no mesmo perfil estrutural.
        recency = []
        structural_pool = [structural, baseline] + [r[1] for r in sorted(ranked,key=lambda x:x[0])[:3]]
        structural_pool = list({json.dumps(c,sort_keys=True):c for c in structural_pool}.values())
        for structure in structural_pool:
            for enabled, half_life, minimum in [(False,180.,.35)] + list(itertools.product([True], [90.,180.,365.,730.], [.1,.35,.65])):
                config = {**structure, "temporal_weight_enabled":enabled, "temporal_half_life_days":half_life, "temporal_min_factor":minimum}
                metric, _ = tuning.evaluate(config)
                selection_score = score(metric) / score(base_metric)
                if len(confirmation.test) >= 35:
                    previous_metric, _ = confirmation.evaluate(config)
                    selection_score += score(previous_metric) / score(confirmation_baseline)
                recency.append((selection_score,config,metric))
        no_recency = min([r for r in recency if not r[1]["temporal_weight_enabled"]],key=lambda x:x[0])
        best_recency = min([r for r in recency if r[1]["temporal_weight_enabled"]],key=lambda x:x[0])
        # Exige ganho no score >=1% para introduzir ponderação temporal.
        selected_recency = best_recency if best_recency[0] < no_recency[0]*.99 else no_recency
        selected = selected_recency[1]
        selected_metric = selected_recency[2]
        # Usa o ganho de ajuste apenas para seleção; nunca usa o teste final.
        selection_baseline = 2. if len(confirmation.test)>=35 else 1.
        if selected_recency[0] >= selection_baseline*.99:
            selected = baseline
            selected_metric = base_metric
        LOGGER.info("Selecionado %s: %s ajuste=%s", purpose, selected, selected_metric)
        tuning.evaluate(selected, verify=True)
        test = Fold(data, purpose, "2026-08-01", "2026-09-30", args.limit*2)
        test_baseline, baseline_frame = test.evaluate(baseline, verify=True)
        test_selected, selected_frame = test.evaluate(selected, verify=True)
        # Ablation is diagnostic only. It cannot change the selected parameters.
        test_no_recency, _ = test.evaluate({**selected,"temporal_weight_enabled":False})
        test_recency, _ = test.evaluate({**selected,"temporal_weight_enabled":True})
        candidate_config = dict(selected)
        candidate_metric = dict(test_selected)
        # Porta de promoção: validação posterior aceita/rejeita, sem retuning.
        # As métricas deste período são de validação, não de teste prospectivo.
        approved = test_selected.get("n",0) >= 30 and not test_selected.get("failed",0) and score(test_selected) < score(test_baseline)*.99 and test_selected["mdape"] <= test_baseline["mdape"]
        if not approved:
            selected = baseline
            test_selected = test_baseline
            selected_frame = baseline_frame.copy()
        value_strata = []
        spatial_strata = []
        if not selected_frame.empty:
            selected_frame["decil"] = pd.qcut(selected_frame.actual*selected_frame.area, 10, labels=False, duplicates="drop")
            for decile, frame in selected_frame.groupby("decil"):
                value_strata.append({"decil":int(decile)+1, **ratio_metrics(frame)})
            for bairro, frame in selected_frame.groupby("bairro"):
                if len(frame)>=5:
                    spatial_strata.append({"bairro":bairro, **ratio_metrics(frame)})
            selected_frame["purpose"] = purpose
            selected_frame["model"] = "selecionado"
            baseline_frame["purpose"] = purpose
            baseline_frame["model"] = "legado"
            all_predictions.extend([selected_frame, baseline_frame])
        config = {**selected, "profile":"aluguel_nao_residencial_2026_10" if approved else "aluguel_legado_mantido_2026_10", "location_weight":1-selected["similarity_weight"], "calibrated_area_column":tuning.area, "calibration_status":"ganho confirmado na validação posterior" if approved else "candidato rejeitado na validação posterior; parâmetros legados mantidos"}
        report["purpose_parameters"][core.normalize_text(purpose)] = config
        confirmation_final, _ = confirmation.evaluate(selected)
        report["evaluations"][purpose] = dict(status=config["calibration_status"], accepted_for_use=bool(approved), candidate_parameters=candidate_config, area_column=tuning.area, tuning_eligible=tuning.eligible_count, tuning_baseline=base_metric, tuning_selected=selected_metric, confirmation_baseline=confirmation_baseline, confirmation_selected=confirmation_final, test_eligible=test.eligible_count, test_baseline=test_baseline, test_candidate=candidate_metric, test_selected=test_selected, test_without_recency=test_no_recency, test_with_recency=test_recency, value_deciles=value_strata, spatial_segments=spatial_strata, recency_tuning=[{"parameters":c,"metrics":m,"score":s} for s,c,m in recency])
        report["candidate_results"].extend([{"purpose":purpose,"parameters":c,"metrics":m,"score":s} for s,c,m in ranked])
        (output/"aluguel_parametros_2026_10.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (output/"aluguel_parametros_2026_10.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame(columns=["purpose", "model", "index", "actual", "predicted", "area"])
    if args.purposes:
        previous = pd.read_csv(ROOT/"tmp/predictions.csv")
        predictions = pd.concat([previous.loc[~previous.purpose.isin(purposes)], predictions],ignore_index=True)
    predictions.to_csv(ROOT/"tmp/predictions.csv", index=False)
    LOGGER.info("Concluído")


if __name__ == "__main__":
    main()

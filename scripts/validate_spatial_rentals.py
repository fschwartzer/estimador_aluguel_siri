"""Diagnóstico espacial pós-seleção; não escolhe parâmetros pelo teste."""
import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_rentals import ROOT, Fold, property_groups, ratio_metrics, core, schema


def spatial_cell(latitude, longitude):
    # Distâncias locais em km, sobre EPSG:4326, mesma aproximação do core.
    x, y = core._local_xy_km(np.asarray(latitude), np.asarray(longitude), -30.03, -51.23)
    return np.column_stack([np.floor(x), np.floor(y)]).astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = ROOT/"artifacts/calibration/aluguel_parametros_2026_10.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    data = pd.read_pickle(args.cache).reset_index(drop=True)
    data["_date"] = core._parse_registration_dates(data.data_encaminhamento)
    data["_group"] = property_groups(data)
    for purpose, evaluation in report["evaluations"].items():
        config = report["purpose_parameters"].get(core.normalize_text(purpose))
        if not config:
            continue
        fold = Fold(data, purpose, "2026-08-01", "2026-09-30", 200)
        train = fold.train
        valid = train.siat_latitude.between(-90,90) & train.siat_longitude.between(-180,180)
        train = train.loc[valid].copy()
        train_cells = spatial_cell(train.siat_latitude, train.siat_longitude)
        records = []
        exclusions = []
        for index, row in fold.test.sample(min(30,len(fold.test)),random_state=20261009).iterrows():
            cell = spatial_cell([row.siat_latitude], [row.siat_longitude])[0]
            spatial = copy.copy(fold)
            mask = np.all(train_cells==cell, axis=1)
            spatial.train = train.loc[~mask].copy()
            spatial.test = fold.test.loc[[index]]
            spatial.preparations = {}
            spatial.contexts = {}
            _, frame = spatial.evaluate(config, verify=True)
            records.append(frame)
            exclusions.append(int(mask.sum()))
        predictions = pd.concat(records,ignore_index=True)
        evaluation["spatial_holdout_1km"] = {**ratio_metrics(predictions), "failed":len(records)-len(predictions), "block_size_km":1., "mean_train_rows_excluded":float(np.mean(exclusions)), "method":"para cada alvo, exclui seu bloco de 1 km do treino temporal, antes do preparo e normalização; sem buffer"}
        logging.info("Espacial %s: %s",purpose,evaluation["spatial_holdout_1km"])
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

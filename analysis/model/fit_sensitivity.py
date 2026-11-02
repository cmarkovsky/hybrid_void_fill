#!/usr/bin/env python3
"""Fit the two regional-transfer models and matched coordinate models from frozen source lists.

All fits use the manuscript settings, seed 42, 172 rounds, and at most 200 targets
per glacier-realization. Coordinates are added only for the coordinates-on arm.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from model_data import REPO, context, read_rows


def fit(data, names, output, params, rounds, threads, coordinates=False):
    values = np.column_stack([data["X"], data["x_coord"].astype(np.float32),
                              data["y_coord"].astype(np.float32)]) if coordinates else data["X"]
    matrix = xgb.DMatrix(values, label=data["y"] - data["h"], weight=data["weight"],
                         feature_names=names, nthread=threads)
    booster = xgb.train({**params, "seed": 42, "nthread": threads}, matrix, num_boost_round=rounds)
    booster.save_model(output)
    print(f"{output.name}: {len(data['y']):,} rows, {len(names)} predictors", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    design = json.loads((REPO / "models/regional_transfer/design.json").read_text())
    params, rounds = design["model"]["params"], design["model"]["rounds"]
    names = context.HYBRID_FEATURES
    outputs = [args.out / f"source_{region}.json" for region in ["whim", "ehim"]]
    outputs += [args.out / "coordinates_off.json", args.out / "coordinates_on.json"]
    if any(path.exists() for path in outputs):
        parser.error("A fitted output exists; choose a fresh --out directory")
    for region in ["whim", "ehim"]:
        data = read_rows(args.data_root, "train", design["sources"][region], cap=200)
        fit(data, names, args.out / f"source_{region}.json", params, rounds, args.threads)
        del data
        gc.collect()
    sources = sorted(set(design["sources"]["whim"]) | set(design["sources"]["ehim"]))
    data = read_rows(args.data_root, "train", sources, cap=200)
    fit(data, names, args.out / "coordinates_off.json", params, rounds, args.threads)
    fit(data, names + ["easting_m", "northing_m"], args.out / "coordinates_on.json",
        params, rounds, args.threads, coordinates=True)


if __name__ == "__main__":
    main()

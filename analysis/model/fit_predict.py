#!/usr/bin/env python3
"""Fit the manuscript hybrid with fixed settings, or predict the default evaluated targets.

The stored manuscript booster is used for prediction unless --model is given.
Fitting writes a new model below --out and never replaces the stored manuscript model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from model_data import REPO, context, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "predict"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group", choices=["val", "test"], default="test")
    parser.add_argument("--model", type=Path, default=REPO / "models/hybrid/model.json")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((REPO / "models/hybrid/training_config.json").read_text())
    data = read_rows(args.data_root, "train" if args.command == "fit" else args.group)
    features = context.HYBRID_FEATURES
    if args.command == "fit":
        if (args.out / "model.json").exists():
            parser.error("Output model already exists; choose a fresh --out directory")
        matrix = xgb.QuantileDMatrix(data["X"], label=data["y"] - data["h"], weight=data["weight"],
                                    feature_names=features, max_bin=256, nthread=args.threads)
        model = xgb.train({**config["training"]["params"], "nthread": args.threads}, matrix,
                          num_boost_round=config["training"]["rounds"])
        model.save_model(args.out / "model.json")
        print(f"Fitted {model.num_boosted_rounds()} trees on {len(data['y']):,} targets")
    else:
        model = xgb.Booster(model_file=args.model)
        matrix = xgb.DMatrix(data["X"], feature_names=features, nthread=args.threads)
        prediction = data["h"] + model.predict(matrix)
        keys = ["rgiid", "realization", "pixel_flat_index", "y", "h", "denominator"]
        np.savez_compressed(args.out / f"{args.group}_predictions.npz",
                            **{key: data[key] for key in keys}, prediction=prediction)
        print(f"Predicted {len(prediction):,} targets on {len(np.unique(data['rgiid'])):,} glaciers")


if __name__ == "__main__":
    main()

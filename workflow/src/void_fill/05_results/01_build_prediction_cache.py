#!/usr/bin/env python3
'Build a duplicate-free, exactly paired point-prediction cache.'

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RGI_PATTERN = re.compile(r"(RGI\d{2}-\d{2}\.\d+)")
REALIZATION_PATTERN = re.compile(r"test_predictions_r(\d+)$")


def decode_scalar_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_scalar_string(value.item())
        if value.size == 1:
            return decode_scalar_string(value.ravel()[0])
    return str(value)


def index_npz_by_glacier(directory: Path) -> dict[str, Path]:
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {directory}")
    result: dict[str, Path] = {}
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            if "RGIId" in data.files:
                gid = decode_scalar_string(data["RGIId"])
            else:
                match = RGI_PATTERN.search(path.name)
                gid = match.group(1) if match else ""
        if not gid:
            raise ValueError(f"Could not identify glacier from {path}")
        if gid in result:
            raise ValueError(f"Duplicate baseline files for {gid}: {result[gid]} and {path}")
        result[gid] = path
    return result


def fill_layer(data: object, realization_index: int, shape: tuple[int, int]) -> np.ndarray:
    fill_key = next(
        (
            key
            for key in (
                "hypsometric_prediction",
                "dhdt_fill_value",
                "constant_prediction",
                "constant_surface",
                "dhdt_filled_constant",
                "dhdt_filled",
                "constant_value",
            )
            if key in data  # type: ignore[operator]
        ),
        None,
    )
    if fill_key is None:
        available = getattr(data, "files", [])
        raise KeyError(
            "No recognized fill array found; expected one of "
            "hypsometric_prediction, dhdt_fill_value, constant_prediction, "
            "constant_surface, dhdt_filled_constant, dhdt_filled, or "
            f"constant_value. Available arrays: {available}"
        )

    fill = np.asarray(data[fill_key], dtype=float)  # type: ignore[index]
    if fill.ndim == 0:
        return np.full(shape, float(fill), dtype=float)
    if fill.ndim == 1:
        if realization_index >= fill.shape[0]:
            raise IndexError(
                f"realization {realization_index} outside {fill_key} array {fill.shape}"
            )
        return np.full(shape, float(fill[realization_index]), dtype=float)
    if fill.ndim == 2 and fill.shape == shape:
        return fill
    if fill.ndim == 3 and fill.shape[1:] == shape:
        if realization_index >= fill.shape[0]:
            raise IndexError(
                f"realization {realization_index} outside {fill_key} array {fill.shape}"
            )
        return fill[realization_index]
    raise ValueError(f"Unsupported {fill_key} shape {fill.shape}; expected {shape}")


def chip_xy(data: object, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data["x"], dtype=float)  # type: ignore[index]
    y = np.asarray(data["y"], dtype=float)  # type: ignore[index]
    if x.ndim == 1 and y.ndim == 1:
        if len(x) != shape[1] or len(y) != shape[0]:
            raise ValueError(f"Coordinate lengths do not match raster shape {shape}")
        return np.meshgrid(x, y)
    if x.shape == shape and y.shape == shape:
        return x, y
    raise ValueError(f"Unsupported coordinate shapes x={x.shape}, y={y.shape}")


def parse_realization_index(path: Path) -> int:
    match = REALIZATION_PATTERN.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Cannot parse zero-based realization from {path.name}")
    return int(match.group(1))


def string_array(values: object) -> np.ndarray:
    """Decode an array of byte or string glacier identifiers."""
    array = np.asarray(values).reshape(-1)
    return np.asarray([decode_scalar_string(value) for value in array], dtype=object)


def native_flat_index(xgb: object, xgb_path: Path) -> tuple[np.ndarray, str]:
    'Read integer target-pixel indices from a saved prediction file.'
    files = set(getattr(xgb, "files", []))
    if "pixel_flat_index" in files:
        return np.asarray(xgb["pixel_flat_index"], dtype=np.int64), "pixel_flat_index"  # type: ignore[index]
    if "match_index" in files:
        print(
            f"WARNING: {xgb_path.name} uses the match_index field; "
            "new XGBoost outputs should contain pixel_flat_index"
        )
        return np.asarray(xgb["match_index"], dtype=np.int64), "match_index"  # type: ignore[index]
    raise KeyError(
        f"{xgb_path.name} is missing pixel_flat_index. This file was not "
        "produced by the new XGBoost workflow and has not been patched."
    )


def build_common_xy_dataframe(
    xgb_dir: Path,
    constant_dir: Path,
    hypsometric_dir: Path,
    region_map: dict[str, str],
    tolerance: float = 1e-6,
    chip_dir: Path | None = None,
) -> pd.DataFrame:
    constant_index = index_npz_by_glacier(constant_dir)
    hypso_index = index_npz_by_glacier(hypsometric_dir)
    xgb_files = sorted(xgb_dir.glob("test_predictions_r*.npz"))
    if not xgb_files:
        raise FileNotFoundError(f"No XGBoost predictions found in {xgb_dir}")

    frames: list[pd.DataFrame] = []
    seen_keys: set[tuple[str, int, int]] = set()

    for xgb_path in xgb_files:
        realization_index = parse_realization_index(xgb_path)
        with np.load(xgb_path, allow_pickle=True) as xgb:
            required = {"y_true", "y_pred", "rgiid", "x", "y"}
            missing = required.difference(xgb.files)
            if missing:
                raise KeyError(f"{xgb_path.name} is missing arrays: {sorted(missing)}")

            y_true_all = np.asarray(xgb["y_true"], dtype=float)
            y_pred_all = np.asarray(xgb["y_pred"], dtype=float)
            gids_all = string_array(xgb["rgiid"])
            x_all = np.asarray(xgb["x"], dtype=float)
            y_all = np.asarray(xgb["y"], dtype=float)
            flat_all, index_source = native_flat_index(xgb, xgb_path)
            xgb_hypso_all = (
                np.asarray(xgb["hypso_pred"], dtype=float)
                if "hypso_pred" in xgb.files
                else None
            )
            if "realization" in xgb.files:
                realization_all = np.asarray(xgb["realization"], dtype=int)
                if not np.all(realization_all == realization_index):
                    raise ValueError(
                        f"{xgb_path.name}: realization array disagrees with filename"
                    )

        arrays_to_check = [y_true_all, y_pred_all, gids_all, x_all, y_all, flat_all]
        if xgb_hypso_all is not None:
            arrays_to_check.append(xgb_hypso_all)
        lengths = {len(a) for a in arrays_to_check}
        if len(lengths) != 1:
            raise ValueError(f"{xgb_path.name}: prediction-array lengths differ")

        for gid in pd.unique(gids_all):
            select = gids_all == gid
            flat = flat_all[select]
            if len(np.unique(flat)) != len(flat):
                raise ValueError(
                    f"{xgb_path.name}/{gid}: duplicate {index_source} values"
                )
            if gid not in constant_index or gid not in hypso_index:
                raise FileNotFoundError(f"Missing baseline output for {gid}")

            with np.load(constant_index[gid], allow_pickle=True) as const, np.load(
                hypso_index[gid], allow_pickle=True
            ) as hypso:
                if chip_dir is not None:
                    original_path = chip_dir / f"{gid}.npz"
                    with np.load(original_path, allow_pickle=True) as original:
                        for name, baseline in (("constant", const), ("hypsometric", hypso)):
                            for key in ("dhdt", "z", "glacier_mask", "void_mask", "x", "y"):
                                if key not in baseline or not np.array_equal(baseline[key], original[key], equal_nan=True):
                                    raise ValueError(f"{gid}: {name} {key} differs from original chip; check run directories")
                dhdt = np.asarray(const["dhdt"], dtype=float)
                glacier = np.asarray(const["glacier_mask"], dtype=bool)
                void = np.asarray(const["void_mask"], dtype=bool)
                if realization_index >= void.shape[0]:
                    raise IndexError(
                        f"{gid}: r{realization_index} outside void stack {void.shape}"
                    )
                if dhdt.shape != glacier.shape or void.shape[1:] != dhdt.shape:
                    raise ValueError(f"{gid}: inconsistent chip shapes")
                if np.any(flat < 0) or np.any(flat >= dhdt.size):
                    raise IndexError(f"{gid}: pixel flat index outside chip raster")

                const_pred = fill_layer(const, realization_index, dhdt.shape).ravel()[flat]
                hypso_pred = fill_layer(hypso, realization_index, dhdt.shape).ravel()[flat]
                obs_chip = dhdt.ravel()[flat]
                intended = (
                    glacier & void[realization_index] & np.isfinite(dhdt)
                ).ravel()[flat]
                x_grid, y_grid = chip_xy(const, dhdt.shape)
                x_chip = x_grid.ravel()[flat]
                y_chip = y_grid.ravel()[flat]

            if not np.all(intended):
                raise ValueError(
                    f"{gid}/r{realization_index}: XGBoost rows include pixels "
                    "outside the intended valid void"
                )
            if not np.allclose(obs_chip, y_true_all[select], rtol=0, atol=tolerance):
                raise ValueError(f"{gid}/r{realization_index}: y_true does not match chip dhdt")
            if not np.allclose(x_chip, x_all[select], rtol=0, atol=tolerance) or not np.allclose(
                y_chip, y_all[select], rtol=0, atol=tolerance
            ):
                raise ValueError(
                    f"{gid}/r{realization_index}: native x/y do not match "
                    f"{index_source}"
                )

            # The row-preparation and stand-alone hypsometric scripts use the
            # same implementation.  Confirm that the baseline carried into
            # XGBoost agrees with the independently saved hypsometric raster.
            if xgb_hypso_all is not None and not np.allclose(
                hypso_pred,
                xgb_hypso_all[select],
                rtol=0,
                atol=tolerance,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{gid}/r{realization_index}: XGBoost hypso_pred does not "
                    "match the hypsometric raster output"
                )

            xgb_pred = y_pred_all[select]
            common_finite = (
                np.isfinite(obs_chip)
                & np.isfinite(const_pred)
                & np.isfinite(hypso_pred)
                & np.isfinite(xgb_pred)
                & np.isfinite(x_chip)
                & np.isfinite(y_chip)
            )
            if not np.all(common_finite):
                removed = int(np.sum(~common_finite))
                print(
                    f"WARNING: {gid}/r{realization_index}: removing {removed:,} "
                    "pixels without finite predictions from all methods"
                )
                flat = flat[common_finite]
                obs_chip = obs_chip[common_finite]
                const_pred = const_pred[common_finite]
                hypso_pred = hypso_pred[common_finite]
                xgb_pred = xgb_pred[common_finite]
                x_chip = x_chip[common_finite]
                y_chip = y_chip[common_finite]
            if flat.size == 0:
                print(
                    f"WARNING: {gid}/r{realization_index}: no finite common-support pixels"
                )
                continue

            keys = {(gid, realization_index, int(value)) for value in flat}
            overlap = keys.intersection(seen_keys)
            if overlap:
                raise ValueError(
                    f"Duplicate glacier-realization-pixel keys while reading {xgb_path.name}: "
                    f"{list(overlap)[:3]}"
                )
            seen_keys.update(keys)

            base = {
                "rgiid": gid,
                "region": str(region_map.get(gid, "unknown")).lower(),
                "realization": f"r{realization_index + 1}",
                "realization_index": realization_index,
                "flat_index": flat,
                "x": x_chip,
                "y": y_chip,
                "obs": obs_chip,
            }
            for method, pred in (
                ("constant", const_pred),
                ("hypsometric", hypso_pred),
                ("xgb", xgb_pred),
            ):
                frame = pd.DataFrame(base)
                frame.insert(0, "method", method)
                frame["pred"] = np.asarray(pred, dtype=float)
                frames.append(frame)

    if not frames:
        raise ValueError("No finite common-support prediction rows were produced")
    result = pd.concat(frames, ignore_index=True)
    keys = ["method", "rgiid", "realization", "flat_index"]
    if result.duplicated(keys).any():
        raise AssertionError("Common-XY builder produced duplicate physical-pixel keys")

    counts = result.groupby(["rgiid", "realization", "method"]).size().unstack("method")
    if counts.isna().any().any() or (counts.nunique(axis=1) != 1).any():
        raise AssertionError("Common-XY builder produced unpaired method counts")
    return result.sort_values(
        ["rgiid", "realization_index", "flat_index", "method"]
    ).reset_index(drop=True)


def read_region_map(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".csv":
        meta = pd.read_csv(path)
    else:
        meta = pd.read_parquet(path)
    id_column = "RGIId" if "RGIId" in meta.columns else "rgiid"
    required = {id_column, "region"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    return (
        meta[[id_column, "region"]]
        .drop_duplicates(id_column)
        .assign(**{id_column: lambda frame: frame[id_column].astype(str)})
        .set_index(id_column)["region"]
        .astype(str)
        .to_dict()
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def restrict_to_split(frame: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    split = json.loads(Path(split_path).read_text())
    predicted = set(frame["rgiid"])
    wrong = sorted(g for g in predicted if g in split and str(split[g]).lower() != "test")
    if wrong:
        raise ValueError(f"Predicted glaciers assigned to a non-test group in {split_path}: {wrong[:5]}")
    dropped = sorted(predicted - set(split))
    if dropped:
        print(f"Excluded {len(dropped)} test glaciers absent from {split_path.name}: {', '.join(dropped)}")
    return frame[frame["rgiid"].isin(split)].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xgb-dir",
        type=Path,
        default=workflow_path('xgb'),
    )
    parser.add_argument(
        "--constant-dir", type=Path,
        default=workflow_path('constant'),
    )
    parser.add_argument(
        "--hypsometric-dir", type=Path,
        default=workflow_path('hypsometric'),
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=workflow_path('metadata'),
    )
    parser.add_argument(
        "--output", type=Path,
        default=workflow_path('cache'),
    )
    parser.add_argument("--chip-dir", type=Path, default=workflow_path("chips"))
    parser.add_argument("--split", type=Path, default=workflow_path("split"))
    return parse_stage_args(parser)


def main() -> None:
    args = parse_args()
    result = build_common_xy_dataframe(
        xgb_dir=args.xgb_dir,
        constant_dir=args.constant_dir,
        hypsometric_dir=args.hypsometric_dir,
        region_map=read_region_map(args.metadata),
        chip_dir=args.chip_dir,
    )
    result = restrict_to_split(result, args.split)
    write_table(result, args.output)
    n_physical_pixels = result[
        ["rgiid", "realization_index", "flat_index"]
    ].drop_duplicates().shape[0]
    print(
        f"Saved {len(result):,} paired method-pixel rows "
        f"({n_physical_pixels:,} unique evaluated pixels) to {args.output}"
    )


if __name__ == "__main__":
    main()

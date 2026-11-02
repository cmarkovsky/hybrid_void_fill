#!/usr/bin/env python3
"""Validate and summarize paired pixel-scale reconstruction errors.

This script is designed for the realization-specific workflow in which
constant and hypsometric baselines are refitted for every artificial-void
realization and XGBoost is trained on pooled artificial-void targets. The
input must be the common-support point cache: every physical pixel must have
finite predictions from constant, hypsometric, and XGBoost.

Two kinds of summaries are written:

* pooled pixel summaries, retained as descriptive diagnostics; and
* primary balanced summaries in which every glacier has equal total weight,
  every retained realization within a glacier has equal weight, and pixels
  divide their realization's weight equally.

The signed-error convention is prediction minus observation.
"""

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = workflow_path('cache')
DEFAULT_OUTPUT_DIR = workflow_path('pixel_summaries')
METHODS = ("constant", "hypsometric", "xgb")
REGION_ALIASES = {
    "western himalaya": "whim",
    "western_himalaya": "whim",
    "w. him": "whim",
    "eastern himalaya": "ehim",
    "eastern_himalaya": "ehim",
    "e. him": "ehim",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--residual-tolerance",
        type=float,
        default=1e-7,
        help="Tolerance when checking a supplied residual against pred - obs.",
    )
    return parse_stage_args(parser)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Common-pixel input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("--input must be a CSV or parquet file")


def parse_realization_number(value: object) -> int:
    text = str(value).strip().lower()
    if text.startswith("r"):
        text = text[1:]
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse realization label {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"Realization label is not an integer: {value!r}")
    return int(number)


def canonicalize_realizations(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the cache's zero-based realization_index when it is available."""
    out = frame.copy()
    parsed = out["realization"].map(parse_realization_number).astype(int)

    if "realization_index" in out.columns:
        index = pd.to_numeric(out["realization_index"], errors="coerce")
        if index.isna().any() or not np.allclose(index, np.round(index)):
            raise ValueError("realization_index must contain finite integers")
        index = index.astype(int)
        if (index < 0).any():
            raise ValueError("realization_index must be zero-based and non-negative")
        agrees = (parsed == index + 1) | (parsed == index)
        if not agrees.all():
            bad = out.loc[~agrees, ["realization", "realization_index"]].head(12)
            raise ValueError(
                "realization labels disagree with realization_index:\n"
                + bad.to_string(index=False)
            )
        out["realization_index"] = index
    else:
        unique = set(parsed.unique())
        index = parsed if 0 in unique else parsed - 1
        if (index < 0).any():
            raise ValueError("Realization labels must be zero-based or one-based integers")
        out["realization_index"] = index.astype(int)
        warnings.warn(
            "Input lacks realization_index; inferred the index convention from labels. "
            "The new cache builder writes realization_index explicitly.",
            stacklevel=2,
        )

    out["realization"] = out["realization_index"].map(lambda value: f"r{value + 1}")
    return out


def canonicalize(
    frame: pd.DataFrame,
    residual_tolerance: float,
) -> tuple[pd.DataFrame, list[str]]:
    required = {"method", "rgiid", "region", "realization"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    out = canonicalize_realizations(frame)
    out["method"] = out["method"].astype(str).str.lower().str.strip()
    out["rgiid"] = out["rgiid"].astype(str)
    out["region"] = (
        out["region"].astype(str).str.lower().str.strip().replace(REGION_ALIASES)
    )
    if out["region"].isin(("", "nan", "none", "unknown")).any():
        raise ValueError("Input contains missing or unknown region labels")

    if "pixel_flat_index" in out.columns and "flat_index" not in out.columns:
        out = out.rename(columns={"pixel_flat_index": "flat_index"})
    if "flat_index" in out.columns:
        out["flat_index"] = pd.to_numeric(out["flat_index"], errors="coerce")
        if out["flat_index"].isna().any():
            raise ValueError("flat_index contains non-numeric or missing values")
        out["flat_index"] = out["flat_index"].astype(np.int64)
        pixel_cols = ["flat_index"]
    elif {"x", "y"}.issubset(out.columns):
        pixel_cols = ["x", "y"]
        for column in pixel_cols:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            if not np.isfinite(out[column]).all():
                raise ValueError(f"{column} contains non-finite coordinates")
    else:
        raise ValueError("Exact pairing requires flat_index or both x and y")

    has_obs_pred = {"obs", "pred"}.issubset(out.columns)
    if has_obs_pred:
        out["obs"] = pd.to_numeric(out["obs"], errors="coerce")
        out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
        calculated = out["pred"] - out["obs"]
        if "residual" in out.columns:
            supplied = pd.to_numeric(out["residual"], errors="coerce")
            finite_both = np.isfinite(supplied) & np.isfinite(calculated)
            if finite_both.any() and not np.allclose(
                supplied[finite_both], calculated[finite_both],
                rtol=0.0, atol=residual_tolerance,
            ):
                raise ValueError(
                    "Supplied residual does not equal pred - obs. Rebuild the cache "
                    "or verify the residual sign convention."
                )
        out["residual"] = calculated
    elif "residual" in out.columns:
        out["residual"] = pd.to_numeric(out["residual"], errors="coerce")
    else:
        raise ValueError("Input must contain residual or both obs and pred")

    observed_methods = set(out["method"].unique())
    if observed_methods != set(METHODS):
        raise ValueError(
            f"Expected methods {list(METHODS)}, found {sorted(observed_methods)}"
        )
    if not np.isfinite(out["residual"]).all():
        raise ValueError(
            "The common-support cache contains non-finite residuals. Rebuild the "
            "cache so it retains only pixels finite for all three methods."
        )
    if out.empty:
        raise RuntimeError("The common-support cache is empty")
    return out, pixel_cols


def validate_exact_support(frame: pd.DataFrame, pixel_cols: list[str]) -> pd.DataFrame:
    physical_key = ["rgiid", "realization_index", *pixel_cols]
    method_key = ["method", *physical_key]
    duplicates = frame.duplicated(method_key, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, method_key].head(12)
        raise ValueError(
            "Duplicate method/glacier/realization/pixel rows found:\n"
            + example.to_string(index=False)
        )

    region_counts = frame.groupby("rgiid", observed=True)["region"].nunique()
    if (region_counts != 1).any():
        raise ValueError("At least one glacier has conflicting region labels")

    support = (
        frame.groupby(physical_key, observed=True)["method"]
        .agg(n_methods="nunique", methods=lambda x: ",".join(sorted(x)))
        .reset_index()
    )
    bad = support[support["n_methods"] != len(METHODS)]
    if not bad.empty:
        raise ValueError(
            "Methods are not evaluated on identical physical pixels. "
            "First incomplete supports:\n" + bad.head(12).to_string(index=False)
        )

    counts = (
        frame.groupby(
            ["rgiid", "region", "realization", "realization_index", "method"],
            observed=True,
        )
        .size().rename("n_pixels").reset_index()
    )
    wide = counts.pivot(
        index=["rgiid", "region", "realization", "realization_index"],
        columns="method", values="n_pixels",
    )
    if wide.isna().any().any() or (wide.nunique(axis=1) != 1).any():
        raise AssertionError("Pixel counts differ across methods after support validation")
    return counts.sort_values(["region", "rgiid", "realization_index", "method"])


def nmad(values: np.ndarray) -> float:
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def unweighted_metrics(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "mae": float(np.mean(np.abs(array))),
        "bias": float(np.mean(array)),
        "nmad": nmad(array),
        "median": float(np.median(array)),
        "p2.5": float(np.percentile(array, 2.5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p97.5": float(np.percentile(array, 97.5)),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = quantile * ordered_weights.sum()
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def balanced_metrics(values: pd.Series, weights: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    weight = weights.to_numpy(dtype=float)
    if len(array) == 0 or not np.isfinite(array).all() or not np.isfinite(weight).all():
        raise ValueError("Balanced metrics require finite values and weights")
    if (weight <= 0).any():
        raise ValueError("Balanced evaluation weights must be positive")
    median = weighted_quantile(array, weight, 0.5)
    return {
        "balanced_rmse": float(np.sqrt(np.average(np.square(array), weights=weight))),
        "balanced_mae": float(np.average(np.abs(array), weights=weight)),
        "balanced_bias": float(np.average(array, weights=weight)),
        "balanced_nmad": float(
            1.4826 * weighted_quantile(np.abs(array - median), weight, 0.5)
        ),
        "balanced_median": median,
        "balanced_p2.5": weighted_quantile(array, weight, 0.025),
        "balanced_p25": weighted_quantile(array, weight, 0.25),
        "balanced_p75": weighted_quantile(array, weight, 0.75),
        "balanced_p97.5": weighted_quantile(array, weight, 0.975),
    }


def add_balanced_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign 1 / (R_g * n_gr) within each method/glacier/realization."""
    out = frame.copy()
    method_glacier = ["method", "rgiid"]
    out["n_realizations_for_glacier"] = out.groupby(
        method_glacier, observed=True
    )["realization_index"].transform("nunique")
    stratum = ["method", "rgiid", "realization_index"]
    out["n_pixels_in_realization"] = out.groupby(stratum, observed=True)[
        "residual"
    ].transform("size")
    out["evaluation_weight"] = 1.0 / (
        out["n_realizations_for_glacier"] * out["n_pixels_in_realization"]
    )
    totals = out.groupby(method_glacier, observed=True)["evaluation_weight"].sum()
    if not np.allclose(totals.to_numpy(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("Balanced weights do not sum to one per method/glacier")
    return out


def pooled_summary(frame: pd.DataFrame, by_region: bool) -> pd.DataFrame:
    group_cols = ["region", "method"] if by_region else ["method"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, observed=True, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_pixels": int(len(group)),
            "n_glaciers": int(group["rgiid"].nunique()),
            "n_glacier_realizations": int(
                group[["rgiid", "realization_index"]].drop_duplicates().shape[0]
            ),
            **unweighted_metrics(group["residual"]),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def balanced_summary(frame: pd.DataFrame, by_region: bool) -> pd.DataFrame:
    group_cols = ["region", "method"] if by_region else ["method"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, observed=True, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_pixels": int(len(group)),
            "n_glaciers": int(group["rgiid"].nunique()),
            "n_glacier_realizations": int(
                group[["rgiid", "realization_index"]].drop_duplicates().shape[0]
            ),
            "weighting": "equal_glacier_equal_realization_equal_pixel_within_realization",
            **balanced_metrics(group["residual"], group["evaluation_weight"]),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def per_glacier_realization_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["region", "method", "rgiid", "realization", "realization_index"]
    for key, group in frame.groupby(group_cols, observed=True, sort=True):
        region, method, rgiid, realization, realization_index = key
        values = group["residual"].to_numpy(dtype=float)
        rows.append({
            "region": region,
            "method": method,
            "rgiid": rgiid,
            "realization": realization,
            "realization_index": int(realization_index),
            "n_pixels": int(len(group)),
            "mse": float(np.mean(np.square(values))),
            **unweighted_metrics(group["residual"]),
        })
    return pd.DataFrame(rows)


def per_glacier_metrics(realization: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (region, method, rgiid), group in realization.groupby(
        ["region", "method", "rgiid"], observed=True, sort=True
    ):
        rows.append({
            "region": region,
            "method": method,
            "rgiid": rgiid,
            "n_realizations": int(group["realization_index"].nunique()),
            "mean_pixels_per_realization": float(group["n_pixels"].mean()),
            "rmse": float(np.sqrt(group["mse"].mean())),
            "mae": float(group["mae"].mean()),
            "bias": float(group["bias"].mean()),
            "mean_realization_nmad": float(group["nmad"].mean()),
            "mean_realization_median": float(group["median"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["region", "method", "rgiid"])


def write_outputs(frame: pd.DataFrame, pixel_cols: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = validate_exact_support(frame, pixel_cols)
    weighted = add_balanced_weights(frame)
    realization = per_glacier_realization_metrics(weighted)
    glaciers = per_glacier_metrics(realization)

    outputs = {
        "pixel_counts_by_method_glacier_realization_common_xy.csv": counts,
        "pixel_level_summary_test_common_xy.csv": pooled_summary(weighted, False),
        "pixel_level_summary_test_by_region_common_xy.csv": pooled_summary(weighted, True),
        "per_glacier_realization_metrics_test_common_xy.csv": realization,
        "per_glacier_metrics_test_common_xy.csv": glaciers,
        "glacier_balanced_pixel_level_summary_test_common_xy.csv": balanced_summary(weighted, False),
        "glacier_balanced_pixel_level_summary_test_by_region_common_xy.csv": balanced_summary(weighted, True),
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / name, index=False)

    keep = [
        "method", "rgiid", "region", "realization", "realization_index",
        *pixel_cols, "residual", "evaluation_weight",
    ]
    residual_path = output_dir / "test_residuals_long_common_xy.parquet"
    try:
        weighted[keep].to_parquet(residual_path, index=False)
        residual_message = str(residual_path)
    except ImportError:
        residual_path = output_dir / "test_residuals_long_common_xy.csv"
        weighted[keep].to_csv(residual_path, index=False)
        residual_message = (
            f"{residual_path} (CSV fallback because no parquet engine is installed)"
        )

    print(f"Validated {len(weighted):,} method-pixel rows")
    print(
        f"Physical evaluated pixels: {len(weighted) // len(METHODS):,}; "
        f"glaciers: {weighted['rgiid'].nunique():,}"
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Saved weighted plot/bootstrap input to {residual_message}")
    print("\nBalanced pixel summary by region:\n")
    print(
        outputs[
            "glacier_balanced_pixel_level_summary_test_by_region_common_xy.csv"
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    frame, pixel_cols = canonicalize(read_table(args.input), args.residual_tolerance)
    write_outputs(frame, pixel_cols, args.output_dir)


if __name__ == "__main__":
    main()

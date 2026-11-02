#!/usr/bin/env python3
"""Calculate tables for the glacier-wide reconstruction comparison.

The comparison contains the original complete glacier-wide estimate and three
reconstructions: constant, hypsometric, and hybrid XGBoost. Artificial-void
realizations are averaged within each glacier so glaciers, rather than pixels
or realization counts, are the statistical units.

This script performs calculations only. It writes three CSV files consumed by
``05_fig5_glacier_wide_comparison.py``:

* one glacier-level table containing original and reconstructed estimates;
* one paired table comparing hypsometric and XGBoost absolute errors; and
* one overall/regional summary table.

By default, only complete-void glacier-wide results are accepted. Results
explicitly labeled ``common_subset_only`` are rejected unless
``--allow-common-subset`` is supplied for a sensitivity analysis.
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


DEFAULT_COORDS_ON = project_path('outputs/coordinate_ablation/coords_on/per_glacier_realization_area_normalized_metrics.csv')
DEFAULT_COORDS_OFF = workflow_path('glacier_metrics') / 'per_glacier_realization_area_normalized_metrics.csv'
DEFAULT_OUTPUT_DIR = workflow_path('comparison')

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
    parser.add_argument(
        "--coordinate-mode",
        choices=("include", "exclude"),
        default="exclude",
        help="Select the XGBoost coordinate variant (default: exclude).",
    )
    parser.add_argument("--coords-on", type=Path, default=DEFAULT_COORDS_ON)
    parser.add_argument("--coords-off", type=Path, default=DEFAULT_COORDS_OFF)
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Metrics CSV overriding the coordinate-mode default path.",
    )
    parser.add_argument(
        "--quantity",
        choices=("specific", "volume"),
        default="specific",
        help="Glacier-wide specific change (primary) or volume-change rate.",
    )
    parser.add_argument(
        "--region",
        default="all",
        help="All regions or one code such as whim or ehim.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-common-subset",
        action="store_true",
        help=(
            "Allow incomplete common-subset glacier-wide integrals. "
            "Use only for an explicitly labeled sensitivity analysis."
        ),
    )
    return parse_stage_args(parser)


def quantity_columns(quantity: str) -> tuple[str, str, str, float]:
    if quantity == "specific":
        return (
            "truth_glacier_mean_dhdt_m_per_yr",
            "pred_glacier_mean_dhdt_m_per_yr",
            "glacier_wide_specific_error_m_per_yr",
            1.0,
        )
    return (
        "truth_glacier_volume_rate_m3_per_yr",
        "pred_glacier_volume_rate_m3_per_yr",
        "void_integrated_error_m3_per_yr",
        1e6,
    )


def normalize_region(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().str.strip().replace(REGION_ALIASES)


def load_metrics(
    path: Path,
    quantity: str,
    region: str,
    allow_common_subset: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    truth_col, pred_col, error_col, divisor = quantity_columns(quantity)
    frame = pd.read_csv(path)
    required = {
        "rgiid",
        "region",
        "realization",
        "method",
        truth_col,
        pred_col,
        error_col,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    keep = list(required)
    if "realization_index" in frame.columns:
        keep.append("realization_index")
    if "glacier_wide_error_scope" in frame.columns:
        keep.append("glacier_wide_error_scope")
    frame = frame[keep].copy()
    frame["rgiid"] = frame["rgiid"].astype(str)
    frame["region"] = normalize_region(frame["region"])
    frame["method"] = frame["method"].astype(str).str.lower().str.strip()
    frame = frame[frame["method"].isin(METHODS)].copy()

    requested_region = normalize_region(pd.Series([region])).iloc[0]
    if requested_region != "all":
        frame = frame[frame["region"] == requested_region].copy()

    if "glacier_wide_error_scope" in frame.columns:
        scope = frame["glacier_wide_error_scope"].astype(str).str.lower()
        common_subset = scope == "common_subset_only"
        if common_subset.any() and not allow_common_subset:
            n_groups = frame.loc[
                common_subset, ["rgiid", "realization"]
            ].drop_duplicates().shape[0]
            raise ValueError(
                f"{n_groups} glacier-realizations are labeled common_subset_only. "
                "They are not complete glacier-wide reconstructions. Rerun the "
                "upstream analysis with complete XGBoost coverage, or pass "
                "--allow-common-subset only for a sensitivity analysis."
            )
        if not allow_common_subset:
            frame = frame[scope == "complete_void"].copy()
    else:
        warnings.warn(
            "Metrics lack glacier_wide_error_scope, so complete-void provenance "
            "cannot be confirmed. Regenerate them with the updated "
            "03_glacier_wide_metrics.py when possible.",
            stacklevel=2,
        )

    for column in (truth_col, pred_col, error_col):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / divisor
    frame = frame[
        np.isfinite(frame[truth_col])
        & np.isfinite(frame[pred_col])
        & np.isfinite(frame[error_col])
    ].copy()
    if frame.empty:
        raise RuntimeError(
            "No finite complete-void results remain. Inspect the coverage tables "
            "written by 03_glacier_wide_metrics.py."
        )

    group_key = ["rgiid", "region", "realization"]
    method_sets = frame.groupby(group_key, observed=True)["method"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    expected = tuple(sorted(METHODS))
    bad_methods = method_sets[method_sets != expected]
    if not bad_methods.empty:
        raise ValueError(
            "At least one glacier-realization does not contain all three methods. "
            f"First problems:\n{bad_methods.head(12).to_string()}"
        )
    if frame.duplicated([*group_key, "method"]).any():
        raise ValueError("Duplicate method rows exist within glacier-realizations")

    truth_spread = frame.groupby(group_key, observed=True)[truth_col].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    tolerance = max(1e-12, 1e-10 * float(np.nanmax(np.abs(frame[truth_col]))))
    if (truth_spread > tolerance).any():
        bad = truth_spread[truth_spread > tolerance].head(12)
        raise ValueError(
            "Original complete-field values differ among methods. "
            f"First problems:\n{bad.to_string()}"
        )

    frame = frame.rename(
        columns={truth_col: "truth", pred_col: "prediction", error_col: "error"}
    )
    return frame.sort_values(["region", "rgiid", "realization", "method"])


def aggregate_by_glacier(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["absolute_error"] = np.abs(work["error"])
    result = (
        work.groupby(["rgiid", "region", "method"], as_index=False, observed=True)
        .agg(
            original=("truth", "mean"),
            reconstructed=("prediction", "mean"),
            mean_signed_error=("error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            root_mean_squared_error=(
                "error",
                lambda values: float(np.sqrt(np.mean(np.square(values)))),
            ),
            n_realizations=("realization", "nunique"),
        )
    )

    glacier_sets = {
        method: set(result.loc[result["method"] == method, "rgiid"])
        for method in METHODS
    }
    reference = glacier_sets[METHODS[0]]
    if any(glacier_sets[method] != reference for method in METHODS[1:]):
        differences = {
            method: len(reference.symmetric_difference(glacier_sets[method]))
            for method in METHODS[1:]
        }
        raise ValueError(f"Methods do not contain identical glaciers: {differences}")

    realization_counts = result.pivot(
        index=["rgiid", "region"], columns="method", values="n_realizations"
    )
    if realization_counts.isna().any().any() or (
        realization_counts.nunique(axis=1) != 1
    ).any():
        raise ValueError("Methods do not contain identical realization counts")
    return result.sort_values(["region", "rgiid", "method"]).reset_index(drop=True)


def paired_recovery_table(glaciers: pd.DataFrame) -> pd.DataFrame:
    wide = glaciers.pivot(
        index=["rgiid", "region"],
        columns="method",
        values="mean_absolute_error",
    )
    missing = set(METHODS).difference(wide.columns)
    if missing or wide[list(METHODS)].isna().any().any():
        raise ValueError(f"Cannot pair all methods; missing={sorted(missing)}")
    paired = wide.reset_index()
    paired["hypsometric_absolute_error_reduction_by_xgb"] = (
        paired["hypsometric"] - paired["xgb"]
    )
    paired["xgb_improved_over_hypsometric"] = (
        paired["hypsometric_absolute_error_reduction_by_xgb"] > 0
    )
    return paired.sort_values(["region", "rgiid"]).reset_index(drop=True)


def bootstrap_recovery(
    paired: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    hyp = paired["hypsometric"].to_numpy(dtype=float)
    xgb = paired["xgb"].to_numpy(dtype=float)
    if len(hyp) == 0:
        return np.nan, np.nan, np.nan
    hyp_mean = float(np.mean(hyp))
    point = 100.0 * (hyp_mean - np.mean(xgb)) / hyp_mean if hyp_mean > 0 else np.nan
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(hyp), size=len(hyp))
        sampled_hyp = np.mean(hyp[indices])
        sampled_xgb = np.mean(xgb[indices])
        boot[draw] = (
            100.0 * (sampled_hyp - sampled_xgb) / sampled_hyp
            if sampled_hyp > 0
            else np.nan
        )
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return point, np.nan, np.nan
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def build_summary(
    glaciers: pd.DataFrame,
    paired: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = [
        ("all", "all", glaciers, paired)
    ]
    for region, subset in glaciers.groupby("region", sort=True, observed=True):
        scopes.append(
            ("region", str(region), subset, paired[paired["region"] == region])
        )

    for scope, region, glacier_subset, paired_subset in scopes:
        recovery, low, high = bootstrap_recovery(paired_subset, draws, rng)
        for method in METHODS:
            method_subset = glacier_subset[glacier_subset["method"] == method]
            errors = method_subset["mean_signed_error"].to_numpy(dtype=float)
            median_error = float(np.median(errors))
            rows.append(
                {
                    "scope": scope,
                    "region": region,
                    "method": method,
                    "n_glaciers": int(method_subset["rgiid"].nunique()),
                    "mean_realizations_per_glacier": float(
                        method_subset["n_realizations"].mean()
                    ),
                    "bias": float(method_subset["mean_signed_error"].mean()),
                    "mae": float(method_subset["mean_absolute_error"].mean()),
                    "nmad_of_glacier_mean_errors": float(
                        1.4826 * np.median(np.abs(errors - median_error))
                    ),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    method_subset["root_mean_squared_error"]
                                )
                            )
                        )
                    ),
                    "fraction_xgb_improved_over_hypsometric": float(
                        paired_subset["xgb_improved_over_hypsometric"].mean()
                    ),
                    "relative_hypsometric_mae_reduction_percent": recovery,
                    "reduction_ci_low_percent": low,
                    "reduction_ci_high_percent": high,
                }
            )
    return pd.DataFrame(rows)


def output_paths(
    output_dir: Path,
    quantity: str,
    coordinate_mode: str,
    region: str,
) -> dict[str, Path]:
    suffix = f"{quantity}_{coordinate_mode}_{region.lower()}"
    prefix = f"glacier_wide_comparison_{suffix}"
    return {
        "glaciers": output_dir / f"{prefix}_glaciers.csv",
        "paired": output_dir / f"{prefix}_paired.csv",
        "summary": output_dir / f"{prefix}_summary.csv",
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws <= 0:
        raise ValueError("--bootstrap-draws must be positive")
    metrics_path = args.metrics
    if metrics_path is None:
        metrics_path = (
            args.coords_on if args.coordinate_mode == "include" else args.coords_off
        )

    realization = load_metrics(
        metrics_path,
        quantity=args.quantity,
        region=args.region,
        allow_common_subset=args.allow_common_subset,
    )
    glaciers = aggregate_by_glacier(realization)
    paired = paired_recovery_table(glaciers)
    summary = build_summary(
        glaciers,
        paired,
        args.bootstrap_draws,
        np.random.default_rng(args.seed),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(
        args.output_dir, args.quantity, args.coordinate_mode, args.region
    )
    glaciers.to_csv(paths["glaciers"], index=False)
    paired.to_csv(paths["paired"], index=False)
    summary.to_csv(paths["summary"], index=False)
    for label, path in paths.items():
        print(f"Saved {label} table to {path}")


if __name__ == "__main__":
    main()

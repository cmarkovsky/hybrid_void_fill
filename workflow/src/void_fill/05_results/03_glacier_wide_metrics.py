#!/usr/bin/env python3
"""
Compute area-normalized void-fill errors for glacier-wide applications.

This analysis deliberately retains the existing void-only pixel evaluation.
Observed, non-void pixels are identical for every method, so adding their zero
residuals to pixel-level MAE/NMAD would dilute differences among methods.

For the glacier-wide calculation, however, those observed pixels cancel
algebraically. If e_i = prediction_i - truth_i on evaluated void pixels V,
then the error in glacier-wide mean elevation-change rate is

    E_glacier = pixel_area * sum_{i in V}(e_i) / valid_glacier_area
              = sum_{i in V}(e_i) / N_valid_glacier_pixels.

The script therefore never pads the pixel residual distribution with zeros.
It integrates only void-pixel errors and normalizes that integral by either:

  * evaluated void area: interpolation skill within the missing area; or
  * valid glacier area: consequence for a whole-glacier geodetic estimate.

The common-support cache uses XGBoost predictions as its master support. If
XGBoost lacks finite predictors at part of an artificial void, the cache does
not cover the complete intended void. By default, this script still reports
void-specific skill on the evaluated common subset but sets integrated and
glacier-wide quantities to NaN for incomplete glacier-realizations. This
prevents a partial integral from being mislabeled as a complete reconstruction.

Expected inputs
---------------
1. The common-XY point cache used by the existing final figures, containing
   method, rgiid, region, realization, obs, and pred.  All methods must refer
   to the same void pixels within each glacier-realization.
2. One baseline/chip NPZ per glacier, containing glacier_mask, dhdt, and
   void_mask.  These arrays define the complete valid glacier support and the
   intended artificial void support.

Primary outputs
---------------
per_glacier_realization_area_normalized_metrics.csv
    One paired row per method, glacier, and artificial-void realization.
per_glacier_area_normalized_metrics.csv
    Realization-averaged values for every method and glacier.
glacier_balanced_area_normalized_summary.csv
    Overall and regional bias, MAE, RMSE, NMAD, and quantiles.  Glaciers, not
    pixels, are the statistical units.
hypsometric_vs_xgb_paired.csv
    Paired absolute-error reductions; positive values favor XGBoost.
hypsometric_vs_xgb_paired_summary.csv
    Overall and regional paired improvements before confidence intervals.

Sign convention
---------------
All signed errors are prediction minus observation.  Positive values mean
that the reconstructed elevation-change rate is more positive than truth.
"""

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_POINT_CACHE = workflow_path('cache')
DEFAULT_CHIP_DIR = workflow_path('chips')
DEFAULT_SPLIT_PATH = workflow_path('split')
DEFAULT_METADATA_PATH = workflow_path('metadata')
DEFAULT_OUTPUT_DIR = workflow_path('glacier_metrics')

METHOD_ORDER = ("constant", "hypsometric", "xgb")
RGI_PATTERN = re.compile(r"(RGI\d{2}-\d{2}\.\d+)")

VOID_SPECIFIC_ERROR = "void_specific_error_m_per_yr"
GLACIER_SPECIFIC_ERROR = "glacier_wide_specific_error_m_per_yr"
VOLUME_ERROR = "void_integrated_error_m3_per_yr"
ERROR_COLUMNS = (VOID_SPECIFIC_ERROR, GLACIER_SPECIFIC_ERROR, VOLUME_ERROR)

COMMON_VOLUME_ERROR = "common_support_void_integrated_error_m3_per_yr"
COMMON_GLACIER_CONTRIBUTION = (
    "common_support_glacier_specific_error_contribution_m_per_yr"
)


def nmad(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def load_test_glaciers(path: Path) -> set[str]:
    with path.open("r") as f:
        split = json.load(f)
    return {str(gid) for gid, group in split.items() if str(group).lower() == "test"}


def load_region_map(path: Path) -> dict[str, str]:
    meta = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    required = {"RGIId", "region"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    return (
        meta[["RGIId", "region"]]
        .drop_duplicates("RGIId")
        .assign(RGIId=lambda x: x["RGIId"].astype(str))
        .set_index("RGIId")["region"]
        .astype(str)
        .str.lower()
        .to_dict()
    )


def parse_realization_number(value: object) -> int:
    text = str(value).strip().lower()
    if text.startswith("r"):
        text = text[1:]
    try:
        number_float = float(text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse realization label {value!r}") from exc
    if not number_float.is_integer():
        raise ValueError(f"Realization label is not an integer: {value!r}")
    return int(number_float)


def normalize_realizations(
    values: pd.Series,
    index_base: str = "auto",
) -> pd.Series:
    """
    Convert an entire realization column to canonical one-based labels.

    The XGBoost preparation code writes is_void_r0 ... is_void_r4 and therefore
    normally produces test_predictions_r0 ... test_predictions_r4.  Some later
    tables use r1 ... r5.  The base must be detected for the complete column;
    converting individual values would incorrectly collapse r0 and r1 to the
    same label.
    """
    if index_base not in {"auto", "zero", "one"}:
        raise ValueError("index_base must be 'auto', 'zero', or 'one'")

    numbers = values.map(parse_realization_number).astype(int)
    unique = set(numbers.unique())

    if index_base == "auto":
        if 0 in unique and 5 in unique:
            raise ValueError(
                "Realization labels contain both r0 and r5, suggesting mixed "
                "zero-based and one-based conventions. Rebuild the cache or set "
                "--realization-base only after verifying its provenance."
            )
        if 0 in unique:
            index_base = "zero"
        elif 5 in unique:
            index_base = "one"
        else:
            index_base = "one"
            warnings.warn(
                "Realization labels contain neither r0 nor r5, so their index base "
                "is ambiguous. Assuming one-based labels; use "
                "--realization-base zero if these came from is_void_r0...r4.",
                stacklevel=2,
            )

    if index_base == "zero":
        if (numbers < 0).any():
            raise ValueError("Zero-based realization labels must be >= 0")
        canonical = numbers + 1
    else:
        if (numbers < 1).any():
            raise ValueError("One-based realization labels must be >= 1")
        canonical = numbers

    return canonical.map(lambda number: f"r{number}")


def canonicalize_realization_columns(
    frame: pd.DataFrame,
    index_base: str = "auto",
) -> pd.DataFrame:
    """Prefer the new cache's explicit zero-based realization_index."""
    out = frame.copy()
    if "realization_index" not in out.columns:
        out["realization"] = normalize_realizations(
            out["realization"], index_base=index_base
        )
        out["realization_index"] = (
            out["realization"].map(parse_realization_number).astype(int) - 1
        )
        return out

    index = pd.to_numeric(out["realization_index"], errors="coerce")
    if index.isna().any() or not np.allclose(index, np.round(index)):
        raise ValueError("realization_index must contain finite integers")
    index = index.astype(int)
    if (index < 0).any():
        raise ValueError("realization_index must be zero-based and non-negative")

    parsed = out["realization"].map(parse_realization_number).astype(int)
    agrees = (parsed == index) | (parsed == index + 1)
    if not agrees.all():
        bad = out.loc[~agrees, ["realization", "realization_index"]].head(12)
        raise ValueError(
            "realization labels disagree with realization_index:\n"
            + bad.to_string(index=False)
        )
    out["realization_index"] = index
    out["realization"] = index.map(lambda value: f"r{value + 1}")
    return out


def collapse_exact_duplicate_pixels(
    df: pd.DataFrame,
    value_tolerance: float = 1e-7,
) -> pd.DataFrame:
    """
    Collapse repeated records for the same physical method-pixel.

    Duplicate cache rows are harmless only when their observed and predicted
    values agree.  Conflicting duplicates are not averaged because doing so
    could conceal a bad merge or incorrect coordinate recovery.
    """
    if "pixel_flat_index" in df.columns and "flat_index" not in df.columns:
        df = df.rename(columns={"pixel_flat_index": "flat_index"})
    if "flat_index" in df.columns:
        physical_keys = ["flat_index"]
    elif {"x", "y"}.issubset(df.columns):
        physical_keys = ["x", "y"]
    else:
        return df

    out = df.copy()
    for key in physical_keys:
        out[key] = pd.to_numeric(out[key], errors="coerce")
        finite = np.isfinite(out[key])
        if not finite.all():
            raise ValueError(
                f"Point cache contains {int((~finite).sum()):,} rows with "
                f"non-finite {key} values"
            )

    keys = ["method", "rgiid", "realization_index", *physical_keys]
    duplicate_mask = out.duplicated(keys, keep=False)
    if not duplicate_mask.any():
        return out

    duplicate_rows = out.loc[duplicate_mask]
    ranges = duplicate_rows.groupby(keys, sort=False, dropna=False)[["obs", "pred"]].agg(
        ["min", "max", "size"]
    )
    conflicting = (
        (ranges[("obs", "max")] - ranges[("obs", "min")] > value_tolerance)
        | (ranges[("pred", "max")] - ranges[("pred", "min")] > value_tolerance)
    )
    if conflicting.any():
        bad = ranges.loc[conflicting].reset_index().head(12)
        raise ValueError(
            "Duplicate physical-pixel rows contain conflicting observed or predicted values. "
            "Do not average these rows; rebuild or repair the common-XY cache. "
            f"First conflicts:\n{bad.to_string(index=False)}"
        )

    before = len(out)
    out = out.drop_duplicates(keys, keep="first").copy()
    removed = before - len(out)
    affected = duplicate_rows[["rgiid", "realization_index"]].drop_duplicates().shape[0]
    warnings.warn(
        f"Removed {removed:,} exact duplicate cache rows across {affected:,} "
        "glacier-realizations before integration.",
        stacklevel=2,
    )
    return out


def load_point_cache(
    path: Path,
    test_glaciers: set[str],
    region_map: dict[str, str],
    realization_base: str = "auto",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Common-XY point cache not found: {path}")

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    df = df.copy()
    required = {"method", "rgiid", "realization", "obs", "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Point cache is missing columns: {sorted(missing)}")

    df["method"] = df["method"].astype(str).str.lower()
    df["rgiid"] = df["rgiid"].astype(str)
    df = canonicalize_realization_columns(df, index_base=realization_base)
    df["obs"] = pd.to_numeric(df["obs"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")

    df = df[df["method"].isin(METHOD_ORDER)]
    df = df[df["rgiid"].isin(test_glaciers)]
    df = df[np.isfinite(df["obs"]) & np.isfinite(df["pred"])].copy()

    if "region" not in df.columns:
        df["region"] = df["rgiid"].map(region_map)
    else:
        region = df["region"].astype(str).str.lower()
        bad = df["region"].isna() | region.isin(("", "none", "nan"))
        df["region"] = region
        df.loc[bad, "region"] = df.loc[bad, "rgiid"].map(region_map)

    if df.empty:
        raise RuntimeError("No finite test-glacier predictions remain in the point cache")
    if df["region"].isna().any():
        gids = sorted(df.loc[df["region"].isna(), "rgiid"].unique())
        raise ValueError(f"Missing region labels for glaciers: {gids[:10]}")

    df = collapse_exact_duplicate_pixels(df)
    df["error"] = df["pred"] - df["obs"]
    return df.reset_index(drop=True)


def verify_paired_evaluation_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """Verify identical methods and observed void pixels in each paired group."""
    problems: list[str] = []
    rows: list[dict[str, object]] = []
    if "flat_index" in df.columns:
        physical_keys = ["flat_index"]
        check_name = "flat raster index"
    elif {"x", "y"}.issubset(df.columns):
        physical_keys = ["x", "y"]
        check_name = "x/y coordinates"
    else:
        physical_keys = []
        check_name = "sorted observed values"

    for (gid, realization_index), sub in df.groupby(
        ["rgiid", "realization_index"], sort=True
    ):
        realization = f"r{int(realization_index) + 1}"
        methods = tuple(sorted(sub["method"].unique()))
        if methods != tuple(sorted(METHOD_ORDER)):
            problems.append(f"{gid}/{realization}: methods={methods}")
            continue

        reference_obs: np.ndarray | None = None
        reference_keys: np.ndarray | None = None
        reference_method: str | None = None

        for method in METHOD_ORDER:
            part = sub[sub["method"] == method].copy()
            if physical_keys:
                if part.duplicated(physical_keys).any():
                    problems.append(
                        f"{gid}/{realization}/{method}: duplicate physical-pixel rows"
                    )
                    continue
                part = part.sort_values(physical_keys)
                keys = part[physical_keys].to_numpy()
            else:
                # Exact locations cannot be checked without coordinates.  Sorting
                # obs still detects count or truth-support differences relevant
                # to the aggregate calculation.
                part = part.sort_values("obs")
                keys = np.empty((len(part), 0))

            obs = part["obs"].to_numpy(dtype=float)
            if reference_obs is None:
                reference_obs = obs
                reference_keys = keys
                reference_method = method
                continue

            if len(obs) != len(reference_obs):
                problems.append(
                    f"{gid}/{realization}: {method} has {len(obs)} rows; "
                    f"{reference_method} has {len(reference_obs)}"
                )
                continue
            if physical_keys and not np.array_equal(keys, reference_keys):
                problems.append(f"{gid}/{realization}: physical support differs by method")
                continue
            if not np.allclose(obs, reference_obs, rtol=0.0, atol=1e-7, equal_nan=False):
                problems.append(f"{gid}/{realization}: observed values differ by method")

        rows.append(
            {
                "rgiid": gid,
                "realization": realization,
                "realization_index": int(realization_index),
                "n_evaluated_void_pixels": int(
                    len(sub[sub["method"] == METHOD_ORDER[0]])
                ),
                "support_check": check_name,
            }
        )

    if problems:
        preview = "\n".join(problems[:12])
        raise ValueError(
            "Methods are not paired on identical evaluation pixels. "
            f"First problems:\n{preview}"
        )

    if not physical_keys:
        warnings.warn(
            "Point cache has no x/y columns. Pairing was verified using counts "
            "and sorted observed values; retain x/y in future caches for a strict check.",
            stacklevel=2,
        )
    return pd.DataFrame(rows)


def decode_scalar_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_scalar_string(value.item())
        if value.size == 1:
            return decode_scalar_string(value.ravel()[0])
    return str(value)


def glacier_id_from_npz(path: Path, data: object) -> str | None:
    files = set(data.files)  # type: ignore[attr-defined]
    if "RGIId" in files:
        return decode_scalar_string(data["RGIId"])  # type: ignore[index]
    match = RGI_PATTERN.search(path.name)
    return match.group(1) if match else None


def load_glacier_support(
    chip_dir: Path,
    glacier_ids: set[str],
    pixel_area_m2: float,
) -> pd.DataFrame:
    """Load complete valid-glacier and intended-void support from chip NPZs."""
    if pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be positive")
    files = sorted(chip_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in chip directory: {chip_dir}")

    rows: list[dict[str, object]] = []
    found: set[str] = set()

    for path in files:
        with np.load(path, allow_pickle=True) as data:
            gid = glacier_id_from_npz(path, data)
            if gid is None or gid not in glacier_ids:
                continue
            required = {"glacier_mask", "dhdt", "void_mask"}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f"{path.name} is missing arrays: {sorted(missing)}")

            glacier_mask = np.asarray(data["glacier_mask"], dtype=bool)
            dhdt = np.asarray(data["dhdt"], dtype=float)
            void_mask = np.asarray(data["void_mask"], dtype=bool)
            if glacier_mask.shape != dhdt.shape:
                raise ValueError(f"{path.name}: glacier_mask and dhdt shapes differ")
            if void_mask.ndim != 3 or void_mask.shape[1:] != dhdt.shape:
                raise ValueError(f"{path.name}: incompatible void_mask shape {void_mask.shape}")

            valid_glacier = glacier_mask & np.isfinite(dhdt)
            n_glacier = int(valid_glacier.sum())
            if n_glacier == 0:
                raise ValueError(f"{path.name}: no valid glacier pixels")

            truth_sum = float(dhdt[valid_glacier].sum())
            truth_mean = truth_sum / n_glacier
            found.add(gid)

            for index, one_void in enumerate(void_mask):
                expected_void = valid_glacier & one_void
                rows.append(
                    {
                        "rgiid": gid,
                        "realization": f"r{index + 1}",
                        "realization_index": index,
                        "n_valid_glacier_pixels": n_glacier,
                        "valid_glacier_area_m2": n_glacier * pixel_area_m2,
                        "truth_glacier_mean_dhdt_m_per_yr": truth_mean,
                        "truth_glacier_volume_rate_m3_per_yr": truth_sum * pixel_area_m2,
                        "n_expected_void_pixels": int(expected_void.sum()),
                        "expected_void_area_m2": int(expected_void.sum()) * pixel_area_m2,
                    }
                )

    missing_glaciers = sorted(glacier_ids.difference(found))
    if missing_glaciers:
        raise FileNotFoundError(
            f"No chip NPZ found for {len(missing_glaciers)} evaluated glaciers; "
            f"first IDs: {missing_glaciers[:10]}"
        )
    return pd.DataFrame(rows)


def build_realization_metrics(
    points: pd.DataFrame,
    support: pd.DataFrame,
    pixel_area_m2: float,
    incomplete_policy: str = "exclude",
) -> pd.DataFrame:
    if incomplete_policy not in {"exclude", "error", "common-subset"}:
        raise ValueError(
            "incomplete_policy must be exclude, error, or common-subset"
        )

    rows: list[dict[str, object]] = []
    for (method, gid, realization_index), sub in points.groupby(
        ["method", "rgiid", "realization_index"], sort=True
    ):
        obs = sub["obs"].to_numpy(dtype=float)
        pred = sub["pred"].to_numpy(dtype=float)
        error = pred - obs
        realization = f"r{int(realization_index) + 1}"
        common_volume_error = float(np.sum(error) * pixel_area_m2)
        rows.append(
            {
                "method": method,
                "rgiid": gid,
                "region": str(sub["region"].iloc[0]).lower(),
                "realization": realization,
                "realization_index": int(realization_index),
                "n_evaluated_void_pixels": int(len(sub)),
                "evaluated_void_area_m2": len(sub) * pixel_area_m2,
                "obs_void_mean_dhdt_m_per_yr": float(np.mean(obs)),
                "pred_void_mean_dhdt_m_per_yr": float(np.mean(pred)),
                VOID_SPECIFIC_ERROR: float(np.mean(error)),
                "obs_evaluated_void_volume_rate_m3_per_yr": float(
                    np.sum(obs) * pixel_area_m2
                ),
                "pred_evaluated_void_volume_rate_m3_per_yr": float(
                    np.sum(pred) * pixel_area_m2
                ),
                COMMON_VOLUME_ERROR: common_volume_error,
            }
        )

    # Enumerate every retained void in the fixed test inventory, including
    # glacier-realizations with no prediction rows at all.
    support = support.copy()
    if "region" not in support:
        region_lookup = points.groupby("rgiid")["region"].first().to_dict()
        support["region"] = support["rgiid"].map(region_lookup)
    if support["region"].isna().any():
        raise ValueError("Missing region for an inventory glacier; supply metadata for the full split")
    inventory = support.merge(pd.DataFrame({"method": METHOD_ORDER}), how="cross")
    measured = pd.DataFrame(rows)
    if not measured.empty:
        measured = measured.drop(columns="region")
    result = inventory.merge(
        measured,
        on=["method", "rgiid", "realization", "realization_index"],
        how="left", validate="one_to_one",
    )
    result["n_evaluated_void_pixels"] = result["n_evaluated_void_pixels"].fillna(0).astype(int)
    result["evaluated_void_area_m2"] = result["evaluated_void_area_m2"].fillna(0.0)
    if result["n_valid_glacier_pixels"].isna().any():
        bad = result.loc[result["n_valid_glacier_pixels"].isna(), "rgiid"].unique()
        raise ValueError(f"Missing glacier support for: {sorted(bad)[:10]}")

    result["evaluation_coverage_fraction"] = (
        result["n_evaluated_void_pixels"] / result["n_expected_void_pixels"]
    )
    zero_expected = result["n_expected_void_pixels"] == 0
    result.loc[zero_expected, "evaluation_coverage_fraction"] = np.nan
    if (result["evaluation_coverage_fraction"] > 1.0 + 1e-12).any():
        bad = result.loc[
            result["evaluation_coverage_fraction"] > 1.0 + 1e-12,
            ["rgiid", "realization", "method", "evaluation_coverage_fraction"],
        ]
        raise ValueError(
            "Evaluated point count exceeds intended valid void support:\n"
            + bad.head().to_string(index=False)
        )

    result["n_missing_void_pixels"] = (
        result["n_expected_void_pixels"] - result["n_evaluated_void_pixels"]
    ).astype(int)
    result["complete_void_coverage"] = (
        (result["n_missing_void_pixels"] == 0) & ~zero_expected
    )
    incomplete = ~result["complete_void_coverage"]
    if incomplete.any():
        n_groups = result.loc[incomplete, ["rgiid", "realization"]].drop_duplicates().shape[0]
        coverage = result.loc[incomplete, "evaluation_coverage_fraction"]
        message = (
            f"{n_groups} glacier-realizations have incomplete common-support "
            f"coverage (median={coverage.median():.3f}, "
            f"minimum={coverage.min():.3f})."
        )
        if incomplete_policy == "error":
            raise ValueError(
                message
                + " Complete XGBoost prediction coverage is required by "
                "--incomplete-policy error."
            )
        if incomplete_policy == "exclude":
            message += (
                " Integrated and glacier-wide quantities are set to NaN for "
                "those groups; void-specific common-support errors remain valid."
            )
        else:
            message += (
                " --incomplete-policy common-subset retains partial "
                "integrals; they are not complete glacier reconstructions."
            )
        warnings.warn(message, stacklevel=2)

    result["evaluated_void_fraction_of_valid_glacier"] = (
        result["n_evaluated_void_pixels"] / result["n_valid_glacier_pixels"]
    )
    result[COMMON_GLACIER_CONTRIBUTION] = (
        result[COMMON_VOLUME_ERROR] / result["valid_glacier_area_m2"]
    )
    result[VOLUME_ERROR] = result[COMMON_VOLUME_ERROR]
    result[GLACIER_SPECIFIC_ERROR] = result[COMMON_GLACIER_CONTRIBUTION]
    if incomplete_policy == "exclude":
        result.loc[incomplete, [VOLUME_ERROR, GLACIER_SPECIFIC_ERROR]] = np.nan

    result["glacier_wide_error_scope"] = np.where(
        result["complete_void_coverage"],
        "complete_void",
        (
            "excluded_incomplete"
            if incomplete_policy == "exclude"
            else "common_subset_only"
        ),
    )
    result.loc[zero_expected, "glacier_wide_error_scope"] = "no_valid_void"
    result["pred_glacier_mean_dhdt_m_per_yr"] = (
        result["truth_glacier_mean_dhdt_m_per_yr"]
        + result[GLACIER_SPECIFIC_ERROR]
    )
    result["pred_glacier_volume_rate_m3_per_yr"] = (
        result["truth_glacier_volume_rate_m3_per_yr"] + result[VOLUME_ERROR]
    )

    # Explicit identity check: glacier-wide error equals void-mean error times
    # the evaluated void fraction. This guards against accidental zero-padding.
    common_identity = (
        result[VOID_SPECIFIC_ERROR]
        * result["evaluated_void_fraction_of_valid_glacier"]
    )
    if not np.allclose(
        result[COMMON_GLACIER_CONTRIBUTION],
        common_identity,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError("Area-normalization identity failed")

    return result.sort_values(
        ["region", "rgiid", "realization", "method"]
    ).reset_index(drop=True)


def build_per_glacier_metrics(realization: pd.DataFrame) -> pd.DataFrame:
    """Average realization-level quantities so each glacier has one row/method."""
    id_cols = ["method", "rgiid", "region"]
    mean_cols = [
        "n_evaluated_void_pixels",
        "n_expected_void_pixels",
        "n_missing_void_pixels",
        "evaluation_coverage_fraction",
        "evaluated_void_fraction_of_valid_glacier",
        "obs_void_mean_dhdt_m_per_yr",
        "pred_void_mean_dhdt_m_per_yr",
        "obs_evaluated_void_volume_rate_m3_per_yr",
        "pred_evaluated_void_volume_rate_m3_per_yr",
        VOLUME_ERROR,
        COMMON_VOLUME_ERROR,
        VOID_SPECIFIC_ERROR,
        GLACIER_SPECIFIC_ERROR,
        COMMON_GLACIER_CONTRIBUTION,
        "truth_glacier_mean_dhdt_m_per_yr",
        "pred_glacier_mean_dhdt_m_per_yr",
        "truth_glacier_volume_rate_m3_per_yr",
        "pred_glacier_volume_rate_m3_per_yr",
        "valid_glacier_area_m2",
    ]
    out = realization.groupby(id_cols, as_index=False)[mean_cols].mean()
    counts = (
        realization.groupby(id_cols, as_index=False)["realization"]
        .nunique()
        .rename(columns={"realization": "n_realizations"})
    )
    completeness = (
        realization.groupby(id_cols, as_index=False)["complete_void_coverage"]
        .all()
        .rename(columns={"complete_void_coverage": "all_realizations_complete"})
    )
    usable_counts = realization.groupby(id_cols, as_index=False)["complete_void_coverage"].sum().rename(columns={"complete_void_coverage": "n_complete_realizations"})
    counts = counts.merge(usable_counts, on=id_cols, validate="one_to_one")
    return (
        counts.merge(out, on=id_cols, validate="one_to_one")
        .merge(completeness, on=id_cols, validate="one_to_one")
    )


def summarize_glacier_balanced(realization: pd.DataFrame) -> pd.DataFrame:
    """Summarize errors with glaciers as equal-weight statistical units."""
    rows: list[dict[str, object]] = []
    scopes = [("all", "all", realization)]
    scopes.extend(("region", str(region), sub) for region, sub in realization.groupby("region"))

    for scope, region, scoped in scopes:
        for method, method_df in scoped.groupby("method"):
            for metric in ERROR_COLUMNS:
                usable = method_df[np.isfinite(method_df[metric])].copy()
                if usable.empty:
                    rows.append(
                        {
                            "scope": scope,
                            "region": region,
                            "method": method,
                            "metric": metric,
                            "n_glaciers": 0,
                            "n_glacier_realizations": 0,
                            "n_excluded_incomplete_realizations": int(len(method_df)),
                            "bias": np.nan,
                            "mae": np.nan,
                            "rmse": np.nan,
                            "nmad_of_glacier_mean_errors": np.nan,
                            "median_glacier_mean_error": np.nan,
                            "p2.5_glacier_mean_error": np.nan,
                            "p97.5_glacier_mean_error": np.nan,
                        }
                    )
                    continue
                # First compute moments across realizations for each glacier.
                per_glacier = usable.groupby("rgiid")[metric].agg(
                    mean_error="mean",
                    mean_abs_error=lambda x: float(np.mean(np.abs(x))),
                    mean_squared_error=lambda x: float(np.mean(np.square(x))),
                )
                glacier_mean_error = per_glacier["mean_error"].to_numpy(dtype=float)
                rows.append(
                    {
                        "scope": scope,
                        "region": region,
                        "method": method,
                        "metric": metric,
                        "n_glaciers": int(len(per_glacier)),
                        "n_glacier_realizations": int(len(usable)),
                        "n_excluded_incomplete_realizations": int(
                            len(method_df) - len(usable)
                        ),
                        "bias": float(per_glacier["mean_error"].mean()),
                        "mae": float(per_glacier["mean_abs_error"].mean()),
                        "rmse": float(np.sqrt(per_glacier["mean_squared_error"].mean())),
                        "nmad_of_glacier_mean_errors": nmad(glacier_mean_error),
                        "median_glacier_mean_error": float(np.median(glacier_mean_error)),
                        "p2.5_glacier_mean_error": float(
                            np.percentile(glacier_mean_error, 2.5)
                        ),
                        "p97.5_glacier_mean_error": float(
                            np.percentile(glacier_mean_error, 97.5)
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["scope", "region", "metric", "method"]
    ).reset_index(drop=True)


def build_paired_comparison(
    realization: pd.DataFrame,
    reference: str = "hypsometric",
    candidate: str = "xgb",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["rgiid", "region", "realization"]
    metrics = list(ERROR_COLUMNS)
    wide = realization.pivot(index=index, columns="method", values=metrics)
    # Flatten the pivot's two-level columns before reset_index/CSV export so
    # downstream readers get one header row and no synthetic all-NaN record.
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]

    missing = [
        f"{metric}_{method}"
        for metric in metrics
        for method in (reference, candidate)
        if f"{metric}_{method}" not in wide.columns
    ]
    if missing:
        raise ValueError(f"Cannot create paired comparison; missing columns: {missing}")

    paired = wide.reset_index()[index].copy()
    for metric in metrics:
        ref_error = wide[f"{metric}_{reference}"].to_numpy(dtype=float)
        cand_error = wide[f"{metric}_{candidate}"].to_numpy(dtype=float)
        paired[f"{reference}_{metric}"] = ref_error
        paired[f"{candidate}_{metric}"] = cand_error
        paired[f"absolute_error_reduction_{metric}"] = (
            np.abs(ref_error) - np.abs(cand_error)
        )
        paired[f"{candidate}_improved_{metric}"] = np.abs(cand_error) < np.abs(ref_error)

    summary_rows: list[dict[str, object]] = []
    scopes = [("all", "all", paired)]
    scopes.extend(("region", str(region), sub) for region, sub in paired.groupby("region"))
    for scope, region, scoped in scopes:
        for metric in metrics:
            valid_pair = np.isfinite(scoped[f"{reference}_{metric}"]) & np.isfinite(
                scoped[f"{candidate}_{metric}"]
            )
            usable = scoped.loc[valid_pair].copy()
            if usable.empty:
                summary_rows.append(
                    {
                        "scope": scope,
                        "region": region,
                        "metric": metric,
                        "reference_method": reference,
                        "candidate_method": candidate,
                        "n_glaciers": 0,
                        "n_glacier_realizations": 0,
                        "n_excluded_unpaired_or_incomplete_realizations": int(
                            len(scoped)
                        ),
                        f"{reference}_mae": np.nan,
                        f"{candidate}_mae": np.nan,
                        "absolute_mae_reduction": np.nan,
                        "relative_mae_reduction_percent": np.nan,
                        "fraction_of_glaciers_improved": np.nan,
                        "fraction_of_glacier_realizations_improved": np.nan,
                    }
                )
                continue
            ref = usable[f"{reference}_{metric}"].to_numpy(dtype=float)
            cand = usable[f"{candidate}_{metric}"].to_numpy(dtype=float)
            # Average within glacier first so glaciers with missing realizations
            # cannot receive extra weight.
            tmp = usable[["rgiid"]].copy()
            tmp["abs_ref"] = np.abs(ref)
            tmp["abs_cand"] = np.abs(cand)
            tmp["improved"] = np.abs(cand) < np.abs(ref)
            by_glacier = tmp.groupby("rgiid").mean(numeric_only=True)
            ref_mae = float(by_glacier["abs_ref"].mean())
            cand_mae = float(by_glacier["abs_cand"].mean())
            summary_rows.append(
                {
                    "scope": scope,
                    "region": region,
                    "metric": metric,
                    "reference_method": reference,
                    "candidate_method": candidate,
                    "n_glaciers": int(len(by_glacier)),
                    "n_glacier_realizations": int(len(usable)),
                    "n_excluded_unpaired_or_incomplete_realizations": int(
                        len(scoped) - len(usable)
                    ),
                    f"{reference}_mae": ref_mae,
                    f"{candidate}_mae": cand_mae,
                    "absolute_mae_reduction": ref_mae - cand_mae,
                    "relative_mae_reduction_percent": (
                        100.0 * (ref_mae - cand_mae) / ref_mae
                        if ref_mae > 0
                        else np.nan
                    ),
                    "fraction_of_glaciers_improved": float(
                        (by_glacier["abs_cand"] < by_glacier["abs_ref"]).mean()
                    ),
                    "fraction_of_glacier_realizations_improved": float(
                        tmp["improved"].mean()
                    ),
                }
            )

    return (
        paired.sort_values(["region", "rgiid", "realization"]).reset_index(drop=True),
        pd.DataFrame(summary_rows).sort_values(["scope", "region", "metric"]),
    )


def build_coverage_diagnostics(
    realization: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one coverage row per glacier-realization plus regional summaries."""
    fields = [
        "n_valid_glacier_pixels",
        "n_expected_void_pixels",
        "n_evaluated_void_pixels",
        "n_missing_void_pixels",
        "evaluation_coverage_fraction",
        "complete_void_coverage",
    ]
    grouped = realization.groupby(
        ["region", "rgiid", "realization", "realization_index"],
        observed=True,
        sort=True,
    )
    disagreement = grouped[fields].nunique(dropna=False)
    if (disagreement > 1).any().any():
        raise AssertionError("Coverage diagnostics differ across methods")
    detailed = grouped[fields].first().reset_index()

    rows: list[dict[str, object]] = []
    scopes = [("all", "all", detailed)]
    scopes.extend(
        ("region", str(region), sub)
        for region, sub in detailed.groupby("region", observed=True)
    )
    for scope, region, scoped in scopes:
        coverage = scoped["evaluation_coverage_fraction"].to_numpy(dtype=float)
        rows.append(
            {
                "scope": scope,
                "region": region,
                "n_glaciers": int(scoped["rgiid"].nunique()),
                "n_glacier_realizations": int(len(scoped)),
                "n_complete_glacier_realizations": int(
                    scoped["complete_void_coverage"].sum()
                ),
                "fraction_complete_glacier_realizations": float(
                    scoped["complete_void_coverage"].mean()
                ),
                "mean_coverage_fraction": float(np.mean(coverage)),
                "median_coverage_fraction": float(np.median(coverage)),
                "p5_coverage_fraction": float(np.percentile(coverage, 5)),
                "minimum_coverage_fraction": float(np.min(coverage)),
                "total_expected_void_pixels": int(
                    scoped["n_expected_void_pixels"].sum()
                ),
                "total_evaluated_void_pixels": int(
                    scoped["n_evaluated_void_pixels"].sum()
                ),
            }
        )
    return detailed, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cache", type=Path, default=DEFAULT_POINT_CACHE)
    parser.add_argument("--chip-dir", type=Path, default=DEFAULT_CHIP_DIR)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--realization-base",
        choices=("auto", "zero", "one"),
        default="auto",
        help=(
            "Input cache realization convention. 'auto' maps r0-r4 to r1-r5 "
            "when r0 is present and otherwise preserves one-based labels."
        ),
    )
    parser.add_argument(
        "--pixel-area-m2",
        type=float,
        default=900.0,
        help="Projected area of one analysis pixel (default: 900 for 30 m pixels)",
    )
    parser.add_argument(
        "--incomplete-policy",
        choices=("exclude", "error", "common-subset"),
        default="exclude",
        help=(
            "How integrated/glacier-wide metrics handle incomplete common support. "
            "Default 'exclude' keeps void-specific metrics but sets incomplete "
            "integrated quantities to NaN. 'error' stops. 'common-subset' retains "
            "partial integrals and should be used only as a sensitivity test."
        ),
    )
    return parse_stage_args(parser)


def validate_original_point_support(points: pd.DataFrame, chip_dir: Path) -> None:
    """Reject stale/wrong cache indices even when group counts happen to match."""
    if "flat_index" not in points:
        raise ValueError("Native flat_index is required; rebuild the point cache")
    reference = points[points["method"] == METHOD_ORDER[0]]
    for gid, glacier in reference.groupby("rgiid", sort=False):
        with np.load(chip_dir / f"{gid}.npz", allow_pickle=True) as data:
            truth = np.asarray(data["dhdt"], dtype=float)
            valid = np.asarray(data["glacier_mask"], dtype=bool) & np.isfinite(truth)
            masks = np.asarray(data["void_mask"], dtype=bool)
            x, y = np.asarray(data["x"]), np.asarray(data["y"])
            xx, yy = np.meshgrid(x, y) if x.ndim == 1 and y.ndim == 1 else (x, y)
            for r, part in glacier.groupby("realization_index"):
                raw = part["flat_index"].to_numpy()
                indices = raw.astype(np.int64)
                if not np.array_equal(raw, indices) or np.any(indices < 0) or np.any(indices >= truth.size):
                    raise ValueError(f"{gid}/r{r}: invalid native pixel indices")
                if r >= len(masks) or not np.all((valid & masks[r]).ravel()[indices]):
                    raise ValueError(f"{gid}/r{r}: point cache differs from original intended void")
                if not np.allclose(part["obs"], truth.ravel()[indices], rtol=0, atol=1e-6):
                    raise ValueError(f"{gid}/r{r}: cached truth differs from original chip")
                if {"x", "y"}.issubset(part):
                    if not np.allclose(part["x"], xx.ravel()[indices], rtol=0, atol=1e-6) or not np.allclose(part["y"], yy.ravel()[indices], rtol=0, atol=1e-6):
                        raise ValueError(f"{gid}/r{r}: cached coordinates differ from original chip")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_glaciers = load_test_glaciers(args.split)
    region_map = load_region_map(args.metadata)
    points = load_point_cache(
        args.point_cache,
        test_glaciers,
        region_map,
        realization_base=args.realization_base,
    )
    pairing = verify_paired_evaluation_pixels(points)
    validate_original_point_support(points, args.chip_dir)

    # Read all split members, not only glaciers surviving the prediction cache.
    evaluated_glaciers = test_glaciers
    support = load_glacier_support(
        chip_dir=args.chip_dir,
        glacier_ids=evaluated_glaciers,
        pixel_area_m2=args.pixel_area_m2,
    )
    support["region"] = support["rgiid"].map(region_map)
    realization = build_realization_metrics(
        points,
        support,
        args.pixel_area_m2,
        incomplete_policy=args.incomplete_policy,
    )
    # Separate actual generated counts from usable complete counts.
    per_glacier = build_per_glacier_metrics(realization)
    summary = summarize_glacier_balanced(realization)
    paired, paired_summary = build_paired_comparison(realization)
    coverage, coverage_summary = build_coverage_diagnostics(realization)

    outputs = {
        "paired_pixel_support.csv": pairing,
        "per_glacier_realization_area_normalized_metrics.csv": realization,
        "per_glacier_area_normalized_metrics.csv": per_glacier,
        "glacier_balanced_area_normalized_summary.csv": summary,
        "hypsometric_vs_xgb_paired.csv": paired,
        "hypsometric_vs_xgb_paired_summary.csv": paired_summary,
        "evaluation_coverage_by_glacier_realization.csv": coverage,
        "evaluation_coverage_summary.csv": coverage_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)

    print("\n=== GLACIER-BALANCED AREA-NORMALIZED SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== HYPSOMETRIC VS XGB PAIRED SUMMARY ===")
    print(paired_summary.to_string(index=False))
    print(f"\nSaved {len(outputs)} tables to {args.output_dir}")


if __name__ == "__main__":
    main()

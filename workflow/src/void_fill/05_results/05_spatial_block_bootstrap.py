#!/usr/bin/env python3
"""Paired spatial-block bootstrap for the final void-filling workflow.

This script replaces the former observational-uncertainty/Q-Q analysis with
confidence intervals on the actual reconstruction-performance statistics.

Statistical design
------------------
* Pixel metrics use the exact common-support residual table from step 02.
* Residual is prediction minus reference: e = pred - obs.
* Each glacier has unit pixel-analysis weight, divided equally among its valid
  realizations and then among evaluated pixels within each realization.
* Integrated-volume and glacier-specific metrics come from the coverage-audited
  step-03 table and use only rows explicitly labelled ``complete_void``.
* Glacier-wide metrics average realizations within glaciers before regional
  summaries, so glaciers remain the statistical units.
* Whole spatial blocks are sampled with replacement within each region.
* Every glacier and all its realizations remain together during resampling.
* The same sampled blocks are used for every method (paired bootstrap).
* Pixels are never treated as independent bootstrap observations.
* The fitted models are not retrained within bootstrap draws.

Performance implementation
--------------------------
Pixel MAE and bias are pre-aggregated to one row per glacier; this is exactly
equivalent to applying the workflow's pixel weights in every draw. The pixel
NMAD point estimate is also exact. For bootstrap NMAD intervals, residuals are
compressed once into weighted spatial-block histograms and those histograms
are resampled. This avoids sorting millions of residuals thousands of times.
``pixel_nmad_histogram_diagnostics.csv`` records the exact-versus-histogram
point-estimate difference for every region, method, and block size.

Interpretation
--------------
The resulting intervals quantify uncertainty in regional performance metrics
from the finite, spatially structured test sample. They are not predictive
intervals for individual reconstructions, and they do not isolate reference-
product measurement error. Any ``sigma`` column in the input cache is ignored.

Expected inputs
---------------
``test_residuals_long_common_xy.parquet`` from step 02 and
``per_glacier_realization_area_normalized_metrics.csv`` from step 03.
The latter must include the complete-coverage provenance fields written by the
updated workflow. Partial common-support integrals are never promoted to
complete glacier reconstructions.

Required glacier chips
----------------------
One NPZ per glacier containing:
  glacier_mask, x, y

The original chips provide projected glacier centroids for block assignment.
"""

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


EXPECTED_METHODS = ("Constant", "Hypsometric", "XGBoost")
MAX_REALIZATIONS = 5
VOLUME_ERROR = "void_integrated_error_m3_per_yr"
SPECIFIC_ERROR = "glacier_wide_specific_error_m_per_yr"
COMPARISONS = (
    ("Constant", "XGBoost"),
    ("Hypsometric", "XGBoost"),
    ("Constant", "Hypsometric"),
)

REPORT_METRICS = (
    "pixel_mae_m_per_yr",
    "pixel_nmad_m_per_yr",
    "pixel_bias_m_per_yr",
    "void_volume_mae_m3_per_yr",
    "void_volume_nmad_m3_per_yr",
    "void_volume_bias_m3_per_yr",
    "glacier_specific_mae_m_per_yr",
    "glacier_specific_nmad_m_per_yr",
    "glacier_specific_bias_m_per_yr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired spatial-block bootstrap of void-filling metrics."
    )
    parser.add_argument(
        "--pixel-residuals",
        type=Path,
        default=workflow_path('pixel_summaries') / 'test_residuals_long_common_xy.parquet',
        help="Exact-common-support residual table written by step 02.",
    )
    parser.add_argument(
        "--glacier-wide-metrics",
        type=Path,
        default=workflow_path('glacier_metrics') / 'per_glacier_realization_area_normalized_metrics.csv',
        help="Coverage-audited glacier-realization table written by step 03.",
    )
    parser.add_argument(
        "--chip-dir",
        type=Path,
        default=workflow_path('chips'),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workflow_path('bootstrap'),
    )
    parser.add_argument(
        "--block-sizes-km",
        nargs="+",
        type=int,
        default=[50, 100, 200],
    )
    parser.add_argument("--primary-block-size-km", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--coordinate-rounding-m",
        type=float,
        default=0.001,
        help="Tolerance used to create pixel keys from x/y; default is 1 mm.",
    )
    parser.add_argument(
        "--pixel-nmad-bins",
        type=int,
        default=4096,
        help=(
            "Number of weighted residual bins used for bootstrap pixel NMAD. "
            "The reported point estimate remains exact (default: 4096)."
        ),
    )
    parser.add_argument(
        "--pixel-nmad-clip-quantile",
        type=float,
        default=0.001,
        help=(
            "Fraction clipped into each outer histogram bin to prevent rare "
            "outliers from degrading NMAD resolution (default: 0.001)."
        ),
    )
    parser.add_argument(
        "--pixel-nmad-warning-tolerance",
        type=float,
        default=0.002,
        help=(
            "Warn when histogram and exact full-sample pixel NMAD differ by "
            "more than this many m/yr (default: 0.002)."
        ),
    )
    return parse_stage_args(parser)


def rename_first(
    frame: pd.DataFrame,
    canonical: str,
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> pd.DataFrame:
    """Rename the first available alias to a canonical column name."""
    if canonical in frame.columns:
        return frame
    for name in candidates:
        if name in frame.columns:
            return frame.rename(columns={name: canonical})
    if required:
        raise KeyError(
            f"Could not find column for {canonical!r}. Tried {candidates}. "
            f"Available columns: {list(frame.columns)}"
        )
    return frame


def normalize_method(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if "constant" in text:
        return "Constant"
    if "hypso" in text:
        return "Hypsometric"
    if "xgb" in text or "boost" in text or "hybrid" in text:
        return "XGBoost"
    return str(value).strip()


def normalize_region(value: object) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "whim": "whim",
        "western": "whim",
        "western himalaya": "whim",
        "w. him": "whim",
        "ehim": "ehim",
        "eastern": "ehim",
        "eastern himalaya": "ehim",
        "e. him": "ehim",
    }
    return aliases.get(text, text)


def parse_realization_number(value: object) -> int:
    text = str(value).strip().lower()
    if text.startswith("r"):
        text = text[1:]
    number = float(text)
    if not number.is_integer():
        raise ValueError(f"Realization label is not an integer: {value!r}")
    return int(number)


def canonicalize_realizations(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer the workflow's explicit zero-based realization_index."""
    out = frame.copy()
    if "realization_index" in out.columns:
        index = pd.to_numeric(out["realization_index"], errors="coerce")
        if index.isna().any() or not np.allclose(index, np.round(index)):
            raise ValueError("realization_index must contain finite integers")
        index = index.astype(int)
        if (index < 0).any():
            raise ValueError("realization_index must be zero-based and non-negative")
        if "realization" in out.columns:
            parsed = out["realization"].map(parse_realization_number).astype(int)
            agrees = (parsed == index) | (parsed == index + 1)
            if not agrees.all():
                raise ValueError("realization labels disagree with realization_index")
        out["realization_index"] = index
    else:
        if "realization" not in out.columns:
            raise KeyError("Input needs realization or realization_index")
        numbers = out["realization"].map(parse_realization_number).astype(int)
        unique = set(numbers.unique())
        if 0 in unique and 5 in unique:
            raise ValueError("Mixed zero- and one-based realization labels")
        if 0 in unique:
            out["realization_index"] = numbers
        else:
            if 5 not in unique:
                warnings.warn(
                    "Realization labels contain neither r0 nor r5; assuming "
                    "one-based labels. Retain realization_index upstream to "
                    "remove this ambiguity.",
                    stacklevel=2,
                )
            out["realization_index"] = numbers - 1
    if (out["realization_index"] >= MAX_REALIZATIONS).any():
        raise ValueError(
            f"Expected at most {MAX_REALIZATIONS} realizations with indices 0-4"
        )
    out["realization"] = out["realization_index"].map(
        lambda value: f"r{value + 1}"
    )
    return out


def nmad(values: pd.Series | np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan
    median = np.median(data)
    return float(1.4826 * np.median(np.abs(data - median)))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Step weighted quantile, matching step 02's balanced summary."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return np.nan

    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    target = quantile * weights.sum()
    index = int(np.searchsorted(np.cumsum(weights), target, side="left"))
    return float(values[min(index, len(values) - 1)])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return weighted_quantile(values, weights, 0.5)


def weighted_nmad(values: np.ndarray, weights: np.ndarray) -> float:
    median = weighted_median(values, weights)
    if not np.isfinite(median):
        return np.nan
    absolute_deviation = np.abs(np.asarray(values, dtype=float) - median)
    return float(1.4826 * weighted_median(absolute_deviation, weights))


def load_pixel_residuals(
    path: Path,
    *,
    coordinate_rounding_m: float,
) -> pd.DataFrame:
    if not path.exists():
        csv_fallback = path.with_suffix(".csv")
        if path.suffix.lower() in {".parquet", ".pq"} and csv_fallback.exists():
            warnings.warn(
                f"{path} is absent; using step-02 CSV fallback {csv_fallback}.",
                stacklevel=2,
            )
            path = csv_fallback
        else:
            raise FileNotFoundError(f"Pixel-residual table not found: {path}")
    if coordinate_rounding_m <= 0:
        raise ValueError("--coordinate-rounding-m must be positive.")

    frame = (
        pd.read_csv(path) if path.suffix.lower() == ".csv"
        else pd.read_parquet(path)
    ).copy()
    frame = rename_first(frame, "rgiid", ("RGIId", "rgi_id", "glacier_id"))
    frame = rename_first(frame, "method", ("model", "fill_method"))
    frame = rename_first(frame, "region", ("subregion", "study_region"))
    frame = rename_first(
        frame,
        "residual",
        ("error", "prediction_error", "dhdt_residual"),
    )
    frame = canonicalize_realizations(frame)

    frame["rgiid"] = frame["rgiid"].astype(str)
    frame["method"] = frame["method"].map(normalize_method)
    frame["region"] = frame["region"].map(normalize_region)
    frame["residual"] = pd.to_numeric(frame["residual"], errors="coerce")

    frame = frame[frame["method"].isin(EXPECTED_METHODS)].copy()
    found_methods = set(frame["method"].unique())
    if found_methods != set(EXPECTED_METHODS):
        raise ValueError(
            f"Expected methods {EXPECTED_METHODS}; found {sorted(found_methods)}."
        )

    finite = np.isfinite(frame["residual"].to_numpy(dtype=float))
    if not finite.all():
        examples = frame.loc[~finite, ["rgiid", "realization", "method"]].head(20)
        raise ValueError(
            f"Point cache contains {(~finite).sum():,} non-finite evaluation rows. "
            f"Examples:\n{examples}"
        )

    if "pixel_flat_index" in frame.columns:
        frame["pixel_key"] = pd.to_numeric(
            frame["pixel_flat_index"], errors="coerce"
        )
    elif "flat_index" in frame.columns:
        frame["pixel_key"] = pd.to_numeric(frame["flat_index"], errors="coerce")
    elif {"x", "y"}.issubset(frame.columns):
        x = pd.to_numeric(frame["x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Pixel residual table contains non-finite x/y")
        scale = 1.0 / coordinate_rounding_m
        coordinate_keys = pd.DataFrame(
            {
                "x": np.rint(x * scale).astype(np.int64),
                "y": np.rint(y * scale).astype(np.int64),
            }
        )
        frame["pixel_key"] = pd.util.hash_pandas_object(
            coordinate_keys, index=False
        ).to_numpy(dtype=np.uint64)
        warnings.warn(
            "No native flat pixel index found; exact support uses rounded x/y.",
            stacklevel=2,
        )
    else:
        raise KeyError(
            "Exact support cannot be verified: retain pixel_flat_index or x/y "
            "in the step-02 residual table."
        )
    if frame["pixel_key"].isna().any():
        raise ValueError("Pixel key contains missing or non-numeric values")
    pixel_key = frame["pixel_key"].to_numpy()
    if not np.isfinite(pixel_key).all() or not np.allclose(
        pixel_key, np.round(pixel_key), rtol=0.0, atol=0.0
    ):
        raise ValueError("Flat pixel indices must contain finite integers")

    duplicate_key = ["rgiid", "realization_index", "pixel_key", "method"]
    duplicated = frame.duplicated(duplicate_key, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, duplicate_key].head(20)
        raise ValueError(f"Duplicate method/pixel rows found:\n{examples}")

    support_key = ["rgiid", "realization_index", "pixel_key"]
    method_count = frame.groupby(
        support_key, observed=True, sort=False
    )["method"].nunique()
    incomplete = method_count[method_count != len(EXPECTED_METHODS)]
    if not incomplete.empty:
        raise ValueError(
            f"{len(incomplete):,} pixel locations do not contain all three "
            "methods. Rebuild the common-support cache before bootstrapping."
        )

    region_count = frame.groupby("rgiid", observed=True)["region"].nunique()
    if (region_count != 1).any():
        raise ValueError("At least one glacier has multiple region labels.")

    realization_counts = frame.groupby("rgiid", observed=True)[
        "realization_index"
    ].nunique()
    if ((realization_counts < 1) | (realization_counts > MAX_REALIZATIONS)).any():
        raise ValueError(f"Glaciers must have 1-{MAX_REALIZATIONS} realizations")

    n_realizations = frame.groupby("rgiid", observed=True)[
        "realization_index"
    ].transform("nunique")
    stratum = ["method", "rgiid", "realization_index"]
    n_pixels = frame.groupby(stratum, observed=True)["residual"].transform("size")
    expected_weight = 1.0 / (n_realizations * n_pixels)
    if "evaluation_weight" in frame.columns:
        supplied = pd.to_numeric(frame["evaluation_weight"], errors="coerce")
        if not np.isfinite(supplied).all() or not np.allclose(
            supplied, expected_weight, rtol=1e-10, atol=1e-14
        ):
            raise ValueError(
                "evaluation_weight does not equal 1/(R_g * n_gr); regenerate "
                "the step-02 residual table."
            )
    frame["evaluation_weight"] = expected_weight
    totals = frame.groupby(["method", "rgiid"], observed=True)[
        "evaluation_weight"
    ].sum()
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("Pixel weights do not sum to one per method/glacier")

    columns = [
        "rgiid",
        "region",
        "realization",
        "realization_index",
        "method",
        "residual",
        "evaluation_weight",
    ]
    return frame[columns]


def decode_scalar_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return decode_scalar_string(value.item())
        if value.size == 1:
            return decode_scalar_string(value.ravel()[0])
    return str(value)


def coordinates_as_2d(
    x: np.ndarray,
    y: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1 and y.ndim == 1:
        x2d, y2d = np.meshgrid(x, y)
    elif x.ndim == 2 and y.ndim == 2:
        x2d, y2d = x, y
    else:
        raise ValueError(f"Unsupported coordinate dimensions: x={x.ndim}, y={y.ndim}")
    if x2d.shape != target_shape or y2d.shape != target_shape:
        raise ValueError(
            f"Coordinate shapes {x2d.shape}/{y2d.shape} do not match "
            f"glacier mask {target_shape}."
        )
    return x2d, y2d


def index_chip_paths(
    chip_dir: Path,
    glacier_ids: set[str],
) -> dict[str, Path]:
    if not chip_dir.exists():
        raise FileNotFoundError(f"Chip directory not found: {chip_dir}")
    paths = sorted(chip_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ chip files found in {chip_dir}")

    index: dict[str, Path] = {}
    for path in paths:
        stem_id = path.stem.split("__")[0]
        if stem_id in glacier_ids:
            glacier_id = stem_id
        elif stem_id.startswith("RGI"):
            # A valid-looking filename that is not part of the test set does
            # not need to be opened merely to inspect its embedded RGIId.
            continue
        else:
            with np.load(path, allow_pickle=True) as data:
                glacier_id = (
                    decode_scalar_string(data["RGIId"])
                    if "RGIId" in data.files
                    else stem_id
                )
            if glacier_id not in glacier_ids:
                continue
        if glacier_id in index:
            raise ValueError(
                f"Multiple chip files found for glacier {glacier_id}: "
                f"{index[glacier_id]} and {path}"
            )
        index[glacier_id] = path
    return index


def load_glacier_geometry(
    chip_dir: Path,
    glacier_ids: set[str],
) -> pd.DataFrame:
    chip_index = index_chip_paths(chip_dir, glacier_ids)
    missing = sorted(glacier_ids - set(chip_index))
    if missing:
        raise FileNotFoundError(
            f"Missing chip files for {len(missing):,} test glaciers. "
            f"First examples: {missing[:20]}"
        )

    rows: list[dict[str, object]] = []
    for glacier_id in sorted(glacier_ids):
        path = chip_index[glacier_id]
        with np.load(path, allow_pickle=True) as data:
            required = ("glacier_mask", "x", "y")
            absent = [name for name in required if name not in data.files]
            if absent:
                raise KeyError(f"{path} is missing chip arrays {absent}")

            mask = np.asarray(data["glacier_mask"], dtype=bool)
            if mask.ndim != 2 or not np.any(mask):
                raise ValueError(f"Invalid or empty glacier_mask in {path}")
            x2d, y2d = coordinates_as_2d(data["x"], data["y"], mask.shape)
            if not np.isfinite(x2d[mask]).all() or not np.isfinite(y2d[mask]).all():
                raise ValueError(f"Non-finite glacier coordinates in {path}")

            rows.append(
                {
                    "rgiid": glacier_id,
                    "centroid_x": float(np.mean(x2d[mask])),
                    "centroid_y": float(np.mean(y2d[mask])),
                    "glacier_pixel_count": int(np.count_nonzero(mask)),
                    "chip_path": str(path),
                }
            )

    geometry = pd.DataFrame(rows)
    if (
        geometry["centroid_x"].abs().le(360).all()
        and geometry["centroid_y"].abs().le(90).all()
    ):
        raise ValueError(
            "Chip coordinates appear geographic; spatial blocks require "
            "projected metre coordinates."
        )
    return geometry


def parse_bool(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True,
         "false": False, "0": False, "no": False}
    )
    if parsed.isna().any():
        raise ValueError(f"Cannot parse {name} as boolean")
    return parsed.astype(bool)


def load_complete_glacier_metrics(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only explicitly complete glacier-realization integrations."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path).copy()
    frame = rename_first(frame, "rgiid", ("RGIId", "rgi_id", "glacier_id"))
    frame = rename_first(frame, "method", ("model", "fill_method"))
    frame = rename_first(frame, "region", ("subregion", "study_region"))
    required = {
        VOLUME_ERROR, SPECIFIC_ERROR, "complete_void_coverage",
        "glacier_wide_error_scope",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} lacks coverage-audited columns: {sorted(missing)}"
        )
    frame = canonicalize_realizations(frame)
    frame["rgiid"] = frame["rgiid"].astype(str)
    frame["method"] = frame["method"].map(normalize_method)
    frame["region"] = frame["region"].map(normalize_region)
    frame = frame[frame["method"].isin(EXPECTED_METHODS)].copy()
    if set(frame["method"].unique()) != set(EXPECTED_METHODS):
        raise ValueError("Glacier-wide table does not contain all three methods")
    frame["complete_void_coverage"] = parse_bool(
        frame["complete_void_coverage"], "complete_void_coverage"
    )
    frame["glacier_wide_error_scope"] = (
        frame["glacier_wide_error_scope"].astype(str).str.strip().str.lower()
    )
    is_complete_scope = frame["glacier_wide_error_scope"] == "complete_void"
    if (is_complete_scope != frame["complete_void_coverage"]).any():
        raise ValueError(
            "complete_void_coverage and glacier_wide_error_scope disagree"
        )
    keys = ["rgiid", "realization_index", "method"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate glacier-realization-method rows")
    group_keys = ["rgiid", "realization_index"]
    method_counts = frame.groupby(group_keys, observed=True)["method"].nunique()
    if (method_counts != len(EXPECTED_METHODS)).any():
        raise ValueError("Glacier-wide rows are not paired across methods")
    coverage_counts = frame.groupby(group_keys, observed=True)[
        "complete_void_coverage"
    ].nunique()
    if (coverage_counts != 1).any():
        raise ValueError("Coverage eligibility differs among methods")
    for column in (VOLUME_ERROR, SPECIFIC_ERROR):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    inventory = frame[
        ["region", "rgiid", "realization_index", "complete_void_coverage"]
    ].drop_duplicates()
    diagnostics = (
        inventory.groupby("region", observed=True)
        .agg(
            n_glaciers=("rgiid", "nunique"),
            n_glacier_realizations=("realization_index", "size"),
            n_complete_glacier_realizations=("complete_void_coverage", "sum"),
        )
        .reset_index()
    )
    diagnostics["n_excluded_incomplete_glacier_realizations"] = (
        diagnostics["n_glacier_realizations"]
        - diagnostics["n_complete_glacier_realizations"]
    )
    diagnostics["complete_fraction"] = (
        diagnostics["n_complete_glacier_realizations"]
        / diagnostics["n_glacier_realizations"]
    )

    complete = frame[frame["complete_void_coverage"] & is_complete_scope].copy()
    finite = np.isfinite(complete[VOLUME_ERROR]) & np.isfinite(
        complete[SPECIFIC_ERROR]
    )
    if not finite.all():
        raise ValueError("Complete-void rows contain non-finite integrated metrics")
    if complete.empty:
        raise RuntimeError(
            "No complete-void results remain; partial common support cannot be "
            "used for glacier-wide inference."
        )
    return complete, diagnostics


def summarize_complete_glaciers(frame: pd.DataFrame) -> pd.DataFrame:
    """Average valid realizations first, yielding one row/glacier/method."""
    rows: list[dict[str, object]] = []
    for (region, method, rgiid), group in frame.groupby(
        ["region", "method", "rgiid"], observed=True, sort=True
    ):
        volume = group[VOLUME_ERROR].to_numpy(dtype=float)
        specific = group[SPECIFIC_ERROR].to_numpy(dtype=float)
        rows.append(
            {
                "region": region,
                "method": method,
                "rgiid": rgiid,
                "n_realizations": int(group["realization_index"].nunique()),
                "volume_mean_error": float(np.mean(volume)),
                "volume_mean_absolute_error": float(np.mean(np.abs(volume))),
                "specific_mean_error": float(np.mean(specific)),
                "specific_mean_absolute_error": float(np.mean(np.abs(specific))),
            }
        )
    result = pd.DataFrame(rows)
    method_counts = result.groupby(["region", "rgiid"], observed=True)[
        "method"
    ].nunique()
    if (method_counts != len(EXPECTED_METHODS)).any():
        raise ValueError("Complete-glacier method pairing is incomplete")
    realization_spread = result.groupby(["region", "rgiid"], observed=True)[
        "n_realizations"
    ].nunique()
    if (realization_spread != 1).any():
        raise ValueError("Complete realization counts differ among methods")
    return result


def make_block_assignments(
    geometry: pd.DataFrame,
    region_lookup: pd.DataFrame,
    block_sizes_km: list[int],
) -> pd.DataFrame:
    centroids = geometry.merge(
        region_lookup,
        on="rgiid",
        how="left",
        validate="one_to_one",
    )
    if centroids["region"].isna().any():
        raise ValueError("At least one glacier centroid lacks a region label.")

    outputs = []
    for block_size_km in block_sizes_km:
        if block_size_km <= 0:
            raise ValueError("All block sizes must be positive.")
        block_size_m = block_size_km * 1000.0
        current = centroids.copy()
        current["block_x"] = np.floor(
            current["centroid_x"] / block_size_m
        ).astype(int)
        current["block_y"] = np.floor(
            current["centroid_y"] / block_size_m
        ).astype(int)
        current["block_id"] = [
            f"{region.lower().replace(' ', '_')}__{block_size_km}km__x{bx}__y{by}"
            for region, bx, by in zip(
                current["region"], current["block_x"], current["block_y"]
            )
        ]
        current["block_size_km"] = block_size_km
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True)


def calculate_pixel_metrics(
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, float]:
    weights = frame["evaluation_weight"].to_numpy(dtype=float) * np.asarray(
        weights, dtype=float
    )
    residual = frame["residual"].to_numpy(dtype=float)
    return {
        "pixel_mae_m_per_yr": weighted_mean(np.abs(residual), weights),
        "pixel_nmad_m_per_yr": weighted_nmad(residual, weights),
        "pixel_bias_m_per_yr": weighted_mean(residual, weights),
    }


def repeated_nmad(values: np.ndarray, multiplicity: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    multiplicity = np.asarray(multiplicity, dtype=int)
    valid = np.isfinite(values) & (multiplicity > 0)
    if not valid.any():
        return np.nan
    return nmad(np.repeat(values[valid], multiplicity[valid]))


def calculate_glacier_metrics(
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, float]:
    weights = np.asarray(weights, dtype=int)
    volume_error = frame["volume_mean_error"].to_numpy(dtype=float)
    specific_error = frame["specific_mean_error"].to_numpy(dtype=float)
    return {
        "void_volume_mae_m3_per_yr": weighted_mean(
            frame["volume_mean_absolute_error"].to_numpy(dtype=float), weights
        ),
        "void_volume_nmad_m3_per_yr": repeated_nmad(volume_error, weights),
        "void_volume_bias_m3_per_yr": weighted_mean(volume_error, weights),
        "glacier_specific_mae_m_per_yr": weighted_mean(
            frame["specific_mean_absolute_error"].to_numpy(dtype=float), weights
        ),
        "glacier_specific_nmad_m_per_yr": repeated_nmad(specific_error, weights),
        "glacier_specific_bias_m_per_yr": weighted_mean(
            specific_error, weights
        ),
    }


def original_estimates(
    pixel: pd.DataFrame,
    glaciers: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (region, method), group in pixel.groupby(
        ["region", "method"], observed=True, sort=True
    ):
        values = calculate_pixel_metrics(group, np.ones(len(group), dtype=int))
        counts = group.groupby("rgiid", observed=True)[
            "realization_index"
        ].nunique()
        for metric, value in values.items():
            rows.append(
                {
                    "region": region,
                    "method": method,
                    "metric": metric,
                    "analysis_scope": "exact_common_pixel_support",
                    "estimate": value,
                    "n_glaciers": group["rgiid"].nunique(),
                    "n_glacier_realizations": group[
                        ["rgiid", "realization_index"]
                    ].drop_duplicates().shape[0],
                    "min_realizations_per_glacier": int(counts.min()),
                    "max_realizations_per_glacier": int(counts.max()),
                }
            )
    for (region, method), group in glaciers.groupby(
        ["region", "method"], observed=True, sort=True
    ):
        values = calculate_glacier_metrics(group, np.ones(len(group), dtype=int))
        for metric, value in values.items():
            rows.append(
                {
                    "region": region,
                    "method": method,
                    "metric": metric,
                    "analysis_scope": "complete_void_only",
                    "estimate": value,
                    "n_glaciers": group["rgiid"].nunique(),
                    "n_glacier_realizations": int(group["n_realizations"].sum()),
                    "min_realizations_per_glacier": int(
                        group["n_realizations"].min()
                    ),
                    "max_realizations_per_glacier": int(
                        group["n_realizations"].max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_pixel_glaciers(pixel: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregate the linear pixel metrics exactly to glacier rows."""
    rows: list[dict[str, object]] = []
    for (region, method, rgiid), group in pixel.groupby(
        ["region", "method", "rgiid"], observed=True, sort=True
    ):
        residual = group["residual"].to_numpy(dtype=float)
        weight = group["evaluation_weight"].to_numpy(dtype=float)
        if not np.isclose(weight.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise AssertionError(
                f"Pixel weights do not sum to one for {method}/{rgiid}"
            )
        rows.append(
            {
                "region": region,
                "method": method,
                "rgiid": rgiid,
                "pixel_mae_m_per_yr": float(np.sum(weight * np.abs(residual))),
                "pixel_bias_m_per_yr": float(np.sum(weight * residual)),
            }
        )
    return pd.DataFrame(rows)


def generate_block_multiplicities(
    n_blocks: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate all block draws once so every method uses identical draws."""
    sampled = rng.integers(0, n_blocks, size=(n_bootstrap, n_blocks))
    multiplicity = np.zeros((n_bootstrap, n_blocks), dtype=np.int16)
    for draw in range(n_bootstrap):
        multiplicity[draw] = np.bincount(
            sampled[draw], minlength=n_blocks
        ).astype(np.int16)
    return multiplicity


def build_block_residual_histogram(
    frame: pd.DataFrame,
    blocks: np.ndarray,
    *,
    n_bins: int,
    clip_quantile: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Compress a weighted residual distribution to one histogram per block."""
    residual = frame["residual"].to_numpy(dtype=float)
    weight = frame["evaluation_weight"].to_numpy(dtype=float)
    lower = weighted_quantile(residual, weight, clip_quantile)
    upper = weighted_quantile(residual, weight, 1.0 - clip_quantile)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Cannot define finite pixel-NMAD histogram bounds")
    if upper <= lower:
        lower = float(np.min(residual))
        upper = float(np.max(residual))
    if upper <= lower:
        # A constant distribution has exactly zero NMAD. A small artificial
        # range keeps the generic histogram code well-defined.
        lower -= 0.5e-12
        upper += 0.5e-12

    edges = np.linspace(lower, upper, n_bins + 1, dtype=float)
    width = float(edges[1] - edges[0])
    bin_index = np.searchsorted(edges, residual, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    block_lookup = {block: index for index, block in enumerate(blocks)}
    block_index = frame["block_id"].map(block_lookup).to_numpy(dtype=int)
    flat_index = block_index * n_bins + bin_index
    histogram = np.bincount(
        flat_index,
        weights=weight,
        minlength=len(blocks) * n_bins,
    ).reshape(len(blocks), n_bins)
    clipped_weight = float(
        weight[(residual < lower) | (residual > upper)].sum() / weight.sum()
    )
    diagnostics = {
        "histogram_lower_m_per_yr": float(lower),
        "histogram_upper_m_per_yr": float(upper),
        "histogram_bin_width_m_per_yr": width,
        "clipped_weight_fraction": clipped_weight,
    }
    return histogram, width, diagnostics


def histogram_nmad_draws(
    block_histogram: np.ndarray,
    multiplicity: np.ndarray,
    bin_width: float,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    """Calculate binned weighted NMAD without sorting pixel rows per draw."""
    n_draws = multiplicity.shape[0]
    n_bins = block_histogram.shape[1]
    output = np.empty(n_draws, dtype=float)

    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        draw_histogram = (
            multiplicity[start:stop].astype(float) @ block_histogram
        )
        cumulative = np.cumsum(draw_histogram, axis=1)
        target = 0.5 * cumulative[:, -1]
        median_index = np.sum(cumulative < target[:, None], axis=1)

        for local_draw, center in enumerate(median_index):
            prefix = cumulative[local_draw]
            threshold = target[local_draw]
            low_radius = 0
            high_radius = max(int(center), n_bins - 1 - int(center))
            while low_radius < high_radius:
                radius = (low_radius + high_radius) // 2
                left = max(0, int(center) - radius)
                right = min(n_bins - 1, int(center) + radius)
                enclosed = prefix[right] - (prefix[left - 1] if left else 0.0)
                if enclosed >= threshold:
                    high_radius = radius
                else:
                    low_radius = radius + 1
            output[start + local_draw] = 1.4826 * low_radius * bin_width
    return output


def bootstrap_pixel_scope(
    pixel: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    block_size_km: int,
    n_bootstrap: int,
    seed: int,
    nmad_bins: int,
    nmad_clip_quantile: float,
    nmad_warning_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fast pixel bootstrap: exact linear metrics and binned weighted NMAD."""
    assignment = assignments.loc[
        assignments["block_size_km"] == block_size_km,
        ["rgiid", "block_id"],
    ]
    raw = pixel.merge(
        assignment, on="rgiid", how="left", validate="many_to_one"
    )
    glaciers = summarize_pixel_glaciers(pixel).merge(
        assignment, on="rgiid", how="left", validate="many_to_one"
    )
    if raw["block_id"].isna().any() or glaciers["block_id"].isna().any():
        raise ValueError(
            "exact_common_pixel_support: at least one row lacks a block assignment"
        )

    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for region_index, (region, region_raw) in enumerate(
        raw.groupby("region", observed=True, sort=True)
    ):
        blocks = np.sort(region_raw["block_id"].unique())
        if len(blocks) < 2:
            raise ValueError(f"Only {len(blocks)} pixel blocks exist in {region}")
        if len(blocks) < 10:
            warnings.warn(
                f"exact_common_pixel_support: only {len(blocks)} occupied "
                f"{block_size_km}-km blocks in {region}; intervals may be unstable.",
                stacklevel=2,
            )
        block_lookup = {block: index for index, block in enumerate(blocks)}
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, block_size_km, 1, region_index])
        )
        multiplicity = generate_block_multiplicities(
            len(blocks), n_bootstrap, rng
        )
        selected_unique = np.count_nonzero(multiplicity, axis=1)
        region_glaciers = glaciers[glaciers["region"] == region]

        for method in EXPECTED_METHODS:
            method_raw = region_raw[region_raw["method"] == method].copy()
            method_glaciers = region_glaciers[
                region_glaciers["method"] == method
            ].copy()
            glacier_block = method_glaciers["block_id"].map(block_lookup).to_numpy(
                dtype=int
            )
            block_count = np.bincount(
                glacier_block, minlength=len(blocks)
            ).astype(float)
            block_mae = np.bincount(
                glacier_block,
                weights=method_glaciers["pixel_mae_m_per_yr"].to_numpy(float),
                minlength=len(blocks),
            )
            block_bias = np.bincount(
                glacier_block,
                weights=method_glaciers["pixel_bias_m_per_yr"].to_numpy(float),
                minlength=len(blocks),
            )
            denominator = multiplicity @ block_count
            mae_draws = (multiplicity @ block_mae) / denominator
            bias_draws = (multiplicity @ block_bias) / denominator

            block_histogram, bin_width, histogram_info = (
                build_block_residual_histogram(
                    method_raw,
                    blocks,
                    n_bins=nmad_bins,
                    clip_quantile=nmad_clip_quantile,
                )
            )
            nmad_draws = histogram_nmad_draws(
                block_histogram, multiplicity, bin_width
            )
            exact_nmad = weighted_nmad(
                method_raw["residual"].to_numpy(dtype=float),
                method_raw["evaluation_weight"].to_numpy(dtype=float),
            )
            approximate_nmad = histogram_nmad_draws(
                block_histogram,
                np.ones((1, len(blocks)), dtype=np.int16),
                bin_width,
            )[0]
            approximation_error = abs(approximate_nmad - exact_nmad)
            if approximation_error > nmad_warning_tolerance:
                warnings.warn(
                    f"{region}/{method}: histogram NMAD differs from the exact "
                    f"point estimate by {approximation_error:.4f} m/yr. Increase "
                    "--pixel-nmad-bins if this is too large.",
                    stacklevel=2,
                )
            diagnostic_rows.append(
                {
                    "block_size_km": block_size_km,
                    "region": region,
                    "method": method,
                    "n_bins": nmad_bins,
                    "clip_quantile_each_tail": nmad_clip_quantile,
                    **histogram_info,
                    "exact_full_sample_nmad_m_per_yr": exact_nmad,
                    "histogram_full_sample_nmad_m_per_yr": approximate_nmad,
                    "absolute_approximation_error_m_per_yr": approximation_error,
                }
            )

            metric_draws = {
                "pixel_mae_m_per_yr": mae_draws,
                "pixel_nmad_m_per_yr": nmad_draws,
                "pixel_bias_m_per_yr": bias_draws,
            }
            for draw in range(n_bootstrap):
                for metric, values in metric_draws.items():
                    rows.append(
                        {
                            "block_size_km": block_size_km,
                            "region": region,
                            "draw": draw,
                            "method": method,
                            "metric": metric,
                            "analysis_scope": "exact_common_pixel_support",
                            "value": float(values[draw]),
                            "n_occupied_blocks": len(blocks),
                            "n_selected_unique_blocks": int(selected_unique[draw]),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(diagnostic_rows)


def bootstrap_scope(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    block_size_km: int,
    n_bootstrap: int,
    seed: int,
    scope: str,
) -> pd.DataFrame:
    assignment = assignments.loc[
        assignments["block_size_km"] == block_size_km,
        ["rgiid", "block_id"],
    ]
    data = data.merge(
        assignment, on="rgiid", how="left", validate="many_to_one"
    )
    if data["block_id"].isna().any():
        raise ValueError(f"{scope}: at least one row lacks a block assignment")

    rows: list[dict[str, object]] = []

    for region_index, (region, region_data) in enumerate(
        data.groupby("region", observed=True, sort=True)
    ):
        blocks = np.sort(region_data["block_id"].unique())
        if len(blocks) < 2:
            raise ValueError(
                f"{scope}: only {len(blocks)} occupied block(s) in {region}"
            )
        if len(blocks) < 10:
            warnings.warn(
                f"{scope}: only {len(blocks)} occupied {block_size_km}-km "
                f"blocks in {region}; intervals may be unstable.",
                stacklevel=2,
            )
        family_code = 1 if scope == "exact_common_pixel_support" else 2
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, block_size_km, family_code, region_index]
            )
        )

        for draw in range(n_bootstrap):
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            multiplicity = Counter(sampled_blocks.tolist())
            draw_data = region_data.copy()
            draw_data["block_weight"] = (
                draw_data["block_id"].map(multiplicity).fillna(0).astype(int)
            )
            draw_data = draw_data.loc[draw_data["block_weight"] > 0].copy()

            for method, method_data in draw_data.groupby(
                "method", observed=True, sort=False
            ):
                block_weight = method_data["block_weight"].to_numpy(dtype=int)
                if scope == "exact_common_pixel_support":
                    values = calculate_pixel_metrics(method_data, block_weight)
                else:
                    values = calculate_glacier_metrics(method_data, block_weight)
                for metric, value in values.items():
                    rows.append(
                        {
                            "block_size_km": block_size_km,
                            "region": region,
                            "draw": draw,
                            "method": method,
                            "metric": metric,
                            "analysis_scope": scope,
                            "value": value,
                            "n_occupied_blocks": len(blocks),
                            "n_selected_unique_blocks": len(multiplicity),
                        }
                    )
    return pd.DataFrame(rows)


def run_bootstrap(
    pixel: pd.DataFrame,
    glaciers: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    block_size_km: int,
    n_bootstrap: int,
    seed: int,
    nmad_bins: int,
    nmad_clip_quantile: float,
    nmad_warning_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pixel_draws, diagnostics = bootstrap_pixel_scope(
        pixel,
        assignments,
        block_size_km=block_size_km,
        n_bootstrap=n_bootstrap,
        seed=seed,
        nmad_bins=nmad_bins,
        nmad_clip_quantile=nmad_clip_quantile,
        nmad_warning_tolerance=nmad_warning_tolerance,
    )
    glacier_draws = bootstrap_scope(
        glaciers,
        assignments,
        block_size_km=block_size_km,
        n_bootstrap=n_bootstrap,
        seed=seed,
        scope="complete_void_only",
    )
    return pd.concat([pixel_draws, glacier_draws], ignore_index=True), diagnostics


def summarize_marginal_intervals(
    point_estimates: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    intervals = (
        draws.groupby(
            ["block_size_km", "region", "method", "metric", "analysis_scope"],
            observed=True,
        )
        .agg(
            ci_low=("value", lambda values: values.quantile(0.025)),
            ci_high=("value", lambda values: values.quantile(0.975)),
            bootstrap_se=("value", "std"),
            n_blocks=("n_occupied_blocks", "first"),
        )
        .reset_index()
    )
    output = intervals.merge(
        point_estimates,
        on=["region", "method", "metric", "analysis_scope"],
        how="left",
        validate="many_to_one",
    )
    output["n_bootstrap"] = n_bootstrap
    output["seed"] = seed
    columns = [
        "block_size_km",
        "region",
        "method",
        "metric",
        "analysis_scope",
        "estimate",
        "ci_low",
        "ci_high",
        "bootstrap_se",
        "n_glaciers",
        "n_glacier_realizations",
        "min_realizations_per_glacier",
        "max_realizations_per_glacier",
        "n_blocks",
        "n_bootstrap",
        "seed",
    ]
    return output[columns].sort_values(
        ["block_size_km", "region", "metric", "method"]
    )


def calculate_paired_differences(
    point_estimates: pd.DataFrame,
    draws: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_wide = point_estimates.pivot(
        index=["region", "metric", "analysis_scope"],
        columns="method",
        values="estimate",
    )
    draw_wide = draws.pivot(
        index=["block_size_km", "region", "draw", "metric", "analysis_scope"],
        columns="method",
        values="value",
    )

    point_rows = []
    draw_rows = []
    for reference_method, comparison_method in COMPARISONS:
        for (region, metric, scope), values in point_wide.iterrows():
            reference = float(values[reference_method])
            comparison = float(values[comparison_method])
            is_bias = "_bias_" in metric
            difference = (
                abs(comparison) - abs(reference)
                if is_bias
                else comparison - reference
            )
            relative = np.nan
            if not is_bias and reference != 0:
                relative = 100.0 * (reference - comparison) / reference
            point_rows.append(
                {
                    "region": region,
                    "metric": metric,
                    "analysis_scope": scope,
                    "reference_method": reference_method,
                    "comparison_method": comparison_method,
                    "difference": difference,
                    "relative_improvement_pct": relative,
                }
            )

        for index, values in draw_wide.iterrows():
            block_size_km, region, draw, metric, scope = index
            reference = float(values[reference_method])
            comparison = float(values[comparison_method])
            is_bias = "_bias_" in metric
            difference = (
                abs(comparison) - abs(reference)
                if is_bias
                else comparison - reference
            )
            relative = np.nan
            if not is_bias and reference != 0:
                relative = 100.0 * (reference - comparison) / reference
            draw_rows.append(
                {
                    "block_size_km": block_size_km,
                    "region": region,
                    "draw": draw,
                    "metric": metric,
                    "analysis_scope": scope,
                    "reference_method": reference_method,
                    "comparison_method": comparison_method,
                    "difference": difference,
                    "relative_improvement_pct": relative,
                }
            )
    return pd.DataFrame(point_rows), pd.DataFrame(draw_rows)


def summarize_paired_intervals(
    point_differences: pd.DataFrame,
    draw_differences: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    keys = [
        "block_size_km",
        "region",
        "metric",
        "analysis_scope",
        "reference_method",
        "comparison_method",
    ]
    intervals = (
        draw_differences.groupby(keys, observed=True)
        .agg(
            ci_low=("difference", lambda values: values.quantile(0.025)),
            ci_high=("difference", lambda values: values.quantile(0.975)),
            relative_ci_low=(
                "relative_improvement_pct", lambda values: values.quantile(0.025)
            ),
            relative_ci_high=(
                "relative_improvement_pct", lambda values: values.quantile(0.975)
            ),
        )
        .reset_index()
    )
    merge_keys = [
        "region", "metric", "analysis_scope",
        "reference_method", "comparison_method"
    ]
    output = intervals.merge(
        point_differences,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    output["n_bootstrap"] = n_bootstrap
    output["seed"] = seed
    return output.sort_values(keys)


def formatted_value(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    scale: float = 1.0,
    decimals: int = 3,
) -> str:
    return (
        f"{estimate * scale:.{decimals}f} "
        f"[{ci_low * scale:.{decimals}f}, {ci_high * scale:.{decimals}f}]"
    )


def build_formatted_tables(
    estimates: pd.DataFrame,
    *,
    primary_block_size_km: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = estimates.loc[
        estimates["block_size_km"] == primary_block_size_km
    ].copy()

    pixel_names = {
        "pixel_mae_m_per_yr": "MAE (m yr-1)",
        "pixel_nmad_m_per_yr": "NMAD (m yr-1)",
        "pixel_bias_m_per_yr": "Bias (m yr-1)",
    }
    volume_names = {
        "void_volume_mae_m3_per_yr": "MAE (10^6 m3 yr-1)",
        "void_volume_nmad_m3_per_yr": "NMAD (10^6 m3 yr-1)",
        "void_volume_bias_m3_per_yr": "Bias (10^6 m3 yr-1)",
    }
    specific_names = {
        "glacier_specific_mae_m_per_yr": "MAE (m yr-1)",
        "glacier_specific_nmad_m_per_yr": "NMAD (m yr-1)",
        "glacier_specific_bias_m_per_yr": "Bias (m yr-1)",
    }

    def create_table(names: dict[str, str], scale: float) -> pd.DataFrame:
        subset = primary.loc[primary["metric"].isin(names)].copy()
        subset["display_metric"] = subset["metric"].map(names)
        subset["estimate_95ci"] = [
            formatted_value(est, low, high, scale=scale)
            for est, low, high in zip(
                subset["estimate"], subset["ci_low"], subset["ci_high"]
            )
        ]
        table = subset.pivot(
            index=["region", "method"],
            columns="display_metric",
            values="estimate_95ci",
        ).reset_index()
        ordered_columns = ["region", "method", *names.values()]
        return table.reindex(columns=ordered_columns)

    return (
        create_table(pixel_names, 1.0),
        create_table(volume_names, 1e-6),
        create_table(specific_names, 1.0),
    )


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2.")
    if args.pixel_nmad_bins < 256:
        raise ValueError("--pixel-nmad-bins must be at least 256.")
    if not 0.0 <= args.pixel_nmad_clip_quantile < 0.1:
        raise ValueError(
            "--pixel-nmad-clip-quantile must be between 0 and 0.1."
        )
    if args.pixel_nmad_warning_tolerance < 0:
        raise ValueError("--pixel-nmad-warning-tolerance cannot be negative.")
    block_sizes = list(dict.fromkeys(args.block_sizes_km))
    if any(size <= 0 for size in block_sizes):
        raise ValueError("All block sizes must be positive.")
    if args.primary_block_size_km not in block_sizes:
        raise ValueError(
            "--primary-block-size-km must also appear in --block-sizes-km."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading exact-common-support pixel residuals...")
    points = load_pixel_residuals(
        args.pixel_residuals,
        coordinate_rounding_m=args.coordinate_rounding_m,
    )
    print(
        f"  {len(points):,} method-pixel rows; "
        f"{points['rgiid'].nunique():,} test glaciers"
    )

    print("Loading coverage-audited glacier-wide metrics...")
    complete, coverage_diagnostics = load_complete_glacier_metrics(
        args.glacier_wide_metrics
    )
    point_regions = points[["rgiid", "region"]].drop_duplicates()
    complete_regions = complete[["rgiid", "region"]].drop_duplicates()
    extra = set(complete_regions["rgiid"]) - set(point_regions["rgiid"])
    if extra:
        raise ValueError(
            "Glacier-wide table contains glaciers absent from the pixel test set: "
            f"{sorted(extra)[:20]}"
        )
    region_check = complete_regions.merge(
        point_regions, on="rgiid", suffixes=("_glacier", "_pixel")
    )
    if (region_check["region_glacier"] != region_check["region_pixel"]).any():
        raise ValueError("Region labels disagree between workflow inputs")
    glacier_metrics = summarize_complete_glaciers(complete)
    n_excluded = int(
        coverage_diagnostics[
            "n_excluded_incomplete_glacier_realizations"
        ].sum()
    )
    print(
        f"  {len(complete) // len(EXPECTED_METHODS):,} complete "
        f"glacier-realizations across {complete['rgiid'].nunique():,} glaciers"
    )
    print(
        f"  Excluded {n_excluded:,} incomplete glacier-realizations from "
        "integrated and glacier-wide inference"
    )

    print("Loading projected glacier centroids and assigning blocks...")
    geometry = load_glacier_geometry(
        args.chip_dir,
        set(points["rgiid"].unique()),
    )
    geometry.to_csv(args.output_dir / "glacier_geometry.csv", index=False)
    glacier_metrics.to_csv(
        args.output_dir / "per_glacier_complete_void_metrics.csv", index=False
    )
    coverage_diagnostics.to_csv(
        args.output_dir / "complete_void_coverage_summary.csv", index=False
    )

    region_lookup = points[["rgiid", "region"]].drop_duplicates()
    assignments = make_block_assignments(
        geometry,
        region_lookup,
        block_sizes,
    )
    assignments.to_csv(args.output_dir / "block_assignments.csv", index=False)
    pixel_block_summary = (
        assignments.groupby(["block_size_km", "region"], observed=True)
        .agg(n_glaciers=("rgiid", "nunique"), n_blocks=("block_id", "nunique"))
        .reset_index()
    )
    pixel_block_summary["analysis_scope"] = "exact_common_pixel_support"
    complete_assignments = assignments[
        assignments["rgiid"].isin(glacier_metrics["rgiid"].unique())
    ]
    complete_block_summary = (
        complete_assignments.groupby(
            ["block_size_km", "region"], observed=True
        )
        .agg(n_glaciers=("rgiid", "nunique"), n_blocks=("block_id", "nunique"))
        .reset_index()
    )
    complete_block_summary["analysis_scope"] = "complete_void_only"
    block_summary = pd.concat(
        [pixel_block_summary, complete_block_summary], ignore_index=True
    )[["analysis_scope", "block_size_km", "region", "n_glaciers", "n_blocks"]]
    block_summary.to_csv(args.output_dir / "block_assignment_summary.csv", index=False)
    print("\nBlock summary:")
    print(block_summary.to_string(index=False))

    if set(glacier_metrics["region"].unique()) != set(points["region"].unique()):
        raise RuntimeError(
            "At least one region has no complete-void glacier-wide results; "
            "the glacier-wide bootstrap cannot be reported for every region."
        )
    point_estimates = original_estimates(points, glacier_metrics)
    point_estimates.to_csv(args.output_dir / "point_estimates.csv", index=False)

    draw_tables = []
    nmad_diagnostic_tables = []
    for block_size_km in block_sizes:
        print(
            f"Running {args.n_bootstrap:,} paired draws with "
            f"{block_size_km}-km blocks..."
        )
        block_draws, block_diagnostics = run_bootstrap(
            points,
            glacier_metrics,
            assignments,
            block_size_km=block_size_km,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            nmad_bins=args.pixel_nmad_bins,
            nmad_clip_quantile=args.pixel_nmad_clip_quantile,
            nmad_warning_tolerance=args.pixel_nmad_warning_tolerance,
        )
        draw_tables.append(block_draws)
        nmad_diagnostic_tables.append(block_diagnostics)
    draws = pd.concat(draw_tables, ignore_index=True)
    nmad_diagnostics = pd.concat(nmad_diagnostic_tables, ignore_index=True)
    if draws["value"].isna().any():
        raise RuntimeError(
            "A bootstrap draw contained no eligible observations; use smaller "
            "blocks or inspect complete-coverage distribution."
        )
    draw_path = args.output_dir / "performance_bootstrap_draws.parquet"
    try:
        draws.to_parquet(draw_path, index=False)
    except ImportError:
        draw_path = args.output_dir / "performance_bootstrap_draws.csv"
        draws.to_csv(draw_path, index=False)
        warnings.warn(
            "No parquet engine is installed; wrote bootstrap draws as CSV.",
            stacklevel=2,
        )

    estimates = summarize_marginal_intervals(
        point_estimates,
        draws,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    estimates.to_csv(args.output_dir / "performance_estimates_ci.csv", index=False)

    point_differences, draw_differences = calculate_paired_differences(
        point_estimates, draws
    )
    paired = summarize_paired_intervals(
        point_differences,
        draw_differences,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    paired.to_csv(args.output_dir / "paired_method_differences_ci.csv", index=False)
    nmad_diagnostics.to_csv(
        args.output_dir / "pixel_nmad_histogram_diagnostics.csv", index=False
    )

    pixel_table, volume_table, specific_table = build_formatted_tables(
        estimates,
        primary_block_size_km=args.primary_block_size_km,
    )
    pixel_table.to_csv(args.output_dir / "table_pixel_metrics_95ci.csv", index=False)
    volume_table.to_csv(args.output_dir / "table_void_volume_metrics_95ci.csv", index=False)
    specific_table.to_csv(
        args.output_dir / "table_glacier_specific_metrics_95ci.csv", index=False
    )

    print("\nPrimary pixel-scale table:")
    print(pixel_table.to_string(index=False))
    print("\nPrimary complete-void glacier-specific table:")
    print(specific_table.to_string(index=False))

    print("\nSaved outputs in", args.output_dir)
    print(
        "Interpretation: these are spatial block-bootstrap confidence intervals "
        "on aggregate test-set performance. Integrated metrics use complete "
        "void coverage only; these are not predictive uncertainty intervals."
    )


if __name__ == "__main__":
    main()

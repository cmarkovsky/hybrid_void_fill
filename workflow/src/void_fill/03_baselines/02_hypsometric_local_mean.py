#!/usr/bin/env python3
"""Realization-specific local-mean hypsometric void filling.

For every glacier and every retained synthetic-void realization, fit the
hypsometric baseline using only that realization's observed (nonvoid) pixels.
Each realization has its own observed source mask.

The elevation-band workflow follows McNabb et al. (2019):

* glacier relief >= 500 m: use 50 m elevation bands;
* glacier relief < 500 m: use ten equal-width elevation bands;
* fill completely unobserved bands with an unweighted cubic fit to occupied
  band means when observations span at least two-thirds of glacier relief;
* otherwise use linear interpolation with flat end extrapolation.  This last
  rule is an explicit fallback because the paper does not prescribe behavior
  when the cubic-fit coverage criterion is not met.

Example
-------
python 02_hypsometric_local_mean.py \
    --chip-dir data_prepared/voids/chips \
    --output-dir outputs/hypsometric_local_mean_realization_specific/run_0
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional progress bar
    def tqdm(iterable: Iterable, *args, **kwargs):
        return iterable


@dataclass(frozen=True)
class HypsometricConfig:
    bin_size_m: float = 50.0
    relief_threshold_m: float = 500.0
    n_bins_low_relief: int = 10
    polynomial_degree: int = 3
    minimum_bins_for_polynomial: int = 4
    polynomial_coverage_fraction: float = 2.0 / 3.0
    fallback: str = "linear"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        return {name: source[name] for name in source.files}


def parse_rgiid(path: Path, chip: dict[str, np.ndarray]) -> str:
    if "RGIId" in chip:
        value = np.asarray(chip["RGIId"])
        if value.size:
            item = value.reshape(-1)[0]
            return item.decode() if isinstance(item, bytes) else str(item)
    match = re.search(r"(RGI\d{2}-\d{2}\.\d+)", path.name)
    if not match:
        raise ValueError(f"Could not determine RGIId from {path}")
    return match.group(1)


def make_elevation_bins(
    z: np.ndarray,
    glacier_mask: np.ndarray,
    config: HypsometricConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    valid = glacier_mask & np.isfinite(z)
    if not np.any(valid):
        raise ValueError("No finite on-glacier elevations")

    zmin = float(np.min(z[valid]))
    zmax = float(np.max(z[valid]))
    relief = zmax - zmin
    if relief <= 0:
        raise ValueError("Glacier elevation range must be greater than zero")

    if relief >= config.relief_threshold_m:
        start = np.floor(zmin / config.bin_size_m) * config.bin_size_m
        stop = np.ceil(zmax / config.bin_size_m) * config.bin_size_m
        if stop <= start:
            stop = start + config.bin_size_m
        edges = np.arange(
            start,
            stop + 0.5 * config.bin_size_m,
            config.bin_size_m,
            dtype=np.float64,
        )
    else:
        edges = np.linspace(
            zmin,
            zmax,
            config.n_bins_low_relief + 1,
            dtype=np.float64,
        )

    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, zmin, zmax


def digitize(z: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign elevations to bins, including values on the final upper edge."""
    values = np.asarray(z, dtype=np.float64)
    n_bins = len(edges) - 1
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[np.isclose(values, edges[-1], rtol=0.0, atol=1e-8)] = n_bins - 1
    valid = np.isfinite(values) & (indices >= 0) & (indices < n_bins)
    return indices, valid


def elevation_band_means(
    z: np.ndarray,
    dhdt: np.ndarray,
    fit_mask: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = len(edges) - 1
    indices, in_range = digitize(z, edges)
    use = fit_mask & in_range & np.isfinite(dhdt)

    means = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    if not np.any(use):
        return means, counts

    counts = np.bincount(indices[use], minlength=n_bins).astype(np.int64)
    sums = np.bincount(
        indices[use],
        weights=np.asarray(dhdt[use], dtype=np.float64),
        minlength=n_bins,
    )
    occupied = counts > 0
    means[occupied] = sums[occupied] / counts[occupied]
    return means, counts


def fill_unobserved_bands(
    centers: np.ndarray,
    means: np.ndarray,
    observed_coverage: float,
    config: HypsometricConfig,
) -> tuple[np.ndarray, str]:
    occupied = np.isfinite(means)
    if not np.any(occupied):
        raise ValueError("No occupied elevation bands")
    if np.all(occupied):
        return means.astype(np.float64, copy=True), "none"

    filled = means.astype(np.float64, copy=True)
    if (
        occupied.sum() >= config.minimum_bins_for_polynomial
        and observed_coverage >= config.polynomial_coverage_fraction
    ):
        x = centers[occupied]
        values = means[occupied]
        x_center = float(np.mean(x))
        x_scale = float(np.ptp(x))
        if np.isfinite(x_scale) and x_scale > 0:
            coefficients = np.polyfit(
                (x - x_center) / x_scale,
                values,
                deg=config.polynomial_degree,
            )
            predicted = np.polyval(
                coefficients,
                (centers - x_center) / x_scale,
            )
            filled[~occupied] = predicted[~occupied]
            return filled, "polynomial_3"

    if config.fallback != "linear":
        raise ValueError(f"Unsupported fallback: {config.fallback}")
    if occupied.sum() == 1:
        filled[~occupied] = means[occupied][0]
    else:
        filled = np.interp(
            centers,
            centers[occupied],
            means[occupied],
            left=means[occupied][0],
            right=means[occupied][-1],
        )
    return filled, "linear"


def hypsometric_local_mean_fill_chip(
    chip: dict[str, np.ndarray],
    rgiid: str,
    config: HypsometricConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return a copy of one chip with realization-specific fill products."""
    config = config or HypsometricConfig()
    required = {"dhdt", "void_mask", "glacier_mask", "z"}
    missing = sorted(required - set(chip))
    if missing:
        raise KeyError(f"{rgiid}: missing required arrays: {missing}")

    dhdt = np.asarray(chip["dhdt"], dtype=np.float64)
    z = np.asarray(chip["z"], dtype=np.float64)
    glacier_mask = np.asarray(chip["glacier_mask"], dtype=bool)
    void_masks = np.asarray(chip["void_mask"], dtype=bool)

    if dhdt.ndim != 2:
        raise ValueError(f"{rgiid}: dhdt must be 2-D; got {dhdt.shape}")
    if z.shape != dhdt.shape or glacier_mask.shape != dhdt.shape:
        raise ValueError(f"{rgiid}: z/glacier_mask shape does not match dhdt")
    if void_masks.ndim != 3 or tuple(void_masks.shape[1:]) != dhdt.shape:
        raise ValueError(
            f"{rgiid}: void_mask must have shape (R, {dhdt.shape[0]}, "
            f"{dhdt.shape[1]}); got {void_masks.shape}"
        )

    edges, centers, zmin, zmax = make_elevation_bins(z, glacier_mask, config)
    bin_indices, in_range = digitize(z, edges)
    n_realizations = void_masks.shape[0]
    n_bins = len(centers)

    prediction = np.full((n_realizations,) + dhdt.shape, np.nan, dtype=np.float32)
    fill_values = np.full_like(prediction, np.nan)
    filled_surfaces = np.broadcast_to(dhdt, prediction.shape).astype(np.float32).copy()
    band_means = np.full((n_realizations, n_bins), np.nan, dtype=np.float64)
    band_filled = np.full_like(band_means, np.nan)
    band_counts = np.zeros((n_realizations, n_bins), dtype=np.int64)
    coverage = np.full(n_realizations, np.nan, dtype=np.float64)
    fit_counts = np.zeros(n_realizations, dtype=np.int64)
    fill_modes = np.empty(n_realizations, dtype="U16")

    finite_glacier_z = glacier_mask & np.isfinite(z) & in_range
    glacier_span = zmax - zmin

    for realization in range(n_realizations):
        fit_mask = (
            finite_glacier_z
            & (~void_masks[realization])
            & np.isfinite(dhdt)
        )
        fit_counts[realization] = int(np.sum(fit_mask))
        if fit_counts[realization] == 0:
            raise ValueError(
                f"{rgiid}: realization {realization} has no finite nonvoid "
                "on-glacier pixels"
            )

        observed_span = float(np.ptp(z[fit_mask])) if fit_counts[realization] > 1 else 0.0
        coverage[realization] = observed_span / glacier_span
        means, counts = elevation_band_means(z, dhdt, fit_mask, edges)
        completed, fill_mode = fill_unobserved_bands(
            centers,
            means,
            coverage[realization],
            config,
        )

        predicted_mask = finite_glacier_z
        prediction[realization, predicted_mask] = completed[
            bin_indices[predicted_mask]
        ].astype(np.float32)

        fill_mask = finite_glacier_z & void_masks[realization]
        fill_values[realization, fill_mask] = prediction[realization, fill_mask]
        filled_surfaces[realization, fill_mask] = prediction[realization, fill_mask]

        band_means[realization] = means
        band_filled[realization] = completed
        band_counts[realization] = counts
        fill_modes[realization] = fill_mode

    output = dict(chip)
    output.update(
        {
            "RGIId": np.array(rgiid),
            "dhdt_filled": filled_surfaces,
            "dhdt_fill_value": fill_values,
            "hypsometric_prediction": prediction,
            "hyp_bin_edges": edges,
            "hyp_bin_centers": centers,
            "hyp_bin_mean": band_means,
            "hyp_bin_filled": band_filled,
            "hyp_bin_counts": band_counts,
            "hyp_observed_coverage_fraction": coverage,
            "hyp_fit_pixel_count": fit_counts,
            "hyp_missing_bin_fill_mode": fill_modes,
            "fit_source_mask": np.array("realization_specific_nonvoid"),
            "void_fill_method": np.array(
                "hypsometric_local_mean_realization_specific"
            ),
        }
    )
    return output


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip-dir", type=Path, default=workflow_path('chips'))
    parser.add_argument("--glob", default="*.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workflow_path('hypsometric'),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bin-size-m", type=float, default=50.0)
    parser.add_argument("--relief-threshold-m", type=float, default=500.0)
    parser.add_argument("--low-relief-bins", type=int, default=10)
    return parser


def main() -> None:
    args = parse_stage_args(build_parser())
    config = HypsometricConfig(
        bin_size_m=args.bin_size_m,
        relief_threshold_m=args.relief_threshold_m,
        n_bins_low_relief=args.low_relief_bins,
    )
    chip_files = sorted(args.chip_dir.glob(args.glob))
    if not chip_files:
        raise FileNotFoundError(f"No chip files matched {args.chip_dir / args.glob}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "method": "hypsometric_local_mean_realization_specific",
        "fit_source_mask": "realization_specific_nonvoid",
        "realization_specific": True,
        "input_directory": str(args.chip_dir),
        "number_of_chips": len(chip_files),
        "configuration": asdict(config),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    progress = tqdm(chip_files, desc="Hypsometric fill", unit="glacier")
    failures: list[dict[str, str]] = []
    for source_path in progress:
        output_path = args.output_dir / source_path.name
        if output_path.exists() and not args.overwrite:
            continue
        try:
            chip = load_npz(source_path)
            rgiid = parse_rgiid(source_path, chip)
            if hasattr(progress, "set_postfix_str"):
                progress.set_postfix_str(rgiid)
            output = hypsometric_local_mean_fill_chip(chip, rgiid, config)
            save_npz_atomic(output_path, output)
        except Exception as exc:
            failures.append({"file": str(source_path), "error": repr(exc)})
            print(f"ERROR {source_path.name}: {exc}")

    if failures:
        failure_path = args.output_dir / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Hypsometric filling failed for {len(failures)} chip(s); "
            f"see {failure_path}"
        )


if __name__ == "__main__":
    main()

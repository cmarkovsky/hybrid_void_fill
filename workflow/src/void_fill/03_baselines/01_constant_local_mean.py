#!/usr/bin/env python3
"""Generate realization-specific constant local-mean reconstructions.

For every glacier and retained void realization, the constant value is the
mean dh/dt of the finite glacier pixels that are nonvoid in that realization.
The value is then assigned to the finite pixels inside that realization's
artificial void.

The input is the original per-glacier chip directory. Each chip must contain:
    dhdt          (ny, nx)
    void_mask     (n_realizations, ny, nx), or (ny, nx) for one realization
    glacier_mask  (ny, nx)

All original arrays are preserved in each output NPZ. The following arrays are
added:
    constant_value       (n_realizations,)
        Nonvoid mean used for each realization.
    constant_prediction  (n_realizations, ny, nx)
        Prediction at finite artificial-void pixels; NaN elsewhere.
    constant_surface     (n_realizations, ny, nx)
        Constant value at all finite glacier pixels; NaN elsewhere.
    dhdt_filled_constant (n_realizations, ny, nx)
        Observed dh/dt outside the void and constant predictions inside it.

Example:
    python constant_local_mean_nonvoid.py \
        --chip-dir data_prepared/voids/chips \
        --output-dir outputs/constant_local_mean_nonvoid/run_0 \
        --overwrite
"""

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
import csv
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("dhdt", "void_mask", "glacier_mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chip-dir",
        type=Path,
        default=workflow_path('chips'),
        help="Directory containing the original per-glacier NPZ chips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workflow_path('constant'),
        help="Directory for reconstructed per-glacier NPZ files.",
    )
    parser.add_argument(
        "--glob",
        default="*.npz",
        help="Input filename pattern (default: *.npz).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output NPZ files.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Floating-point type for newly generated arrays.",
    )
    return parse_stage_args(parser)


def normalize_void_stack(void_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean stack with shape (n_realizations, ny, nx)."""
    void_stack = np.asarray(void_mask, dtype=bool)
    if void_stack.ndim == 2:
        void_stack = void_stack[np.newaxis, ...]
    if void_stack.ndim != 3 or tuple(void_stack.shape[1:]) != shape:
        raise ValueError(
            "void_mask must have shape (ny, nx) or "
            f"(n_realizations, ny, nx); received {void_stack.shape}, "
            f"expected spatial shape {shape}"
        )
    return void_stack


def reconstruct_chip(
    chip: dict[str, np.ndarray],
    output_dtype: np.dtype,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    missing = [key for key in REQUIRED_KEYS if key not in chip]
    if missing:
        raise KeyError(f"Missing required arrays: {missing}")

    dhdt = np.asarray(chip["dhdt"], dtype=np.float64)
    glacier_mask = np.asarray(chip["glacier_mask"], dtype=bool)
    if dhdt.ndim != 2:
        raise ValueError(f"dhdt must be two-dimensional; received {dhdt.shape}")
    if glacier_mask.shape != dhdt.shape:
        raise ValueError(
            f"glacier_mask shape {glacier_mask.shape} does not match dhdt {dhdt.shape}"
        )

    void_stack = normalize_void_stack(chip["void_mask"], dhdt.shape)
    valid_glacier = glacier_mask & np.isfinite(dhdt)
    n_realizations = void_stack.shape[0]

    constant_values = np.full(n_realizations, np.nan, dtype=np.float64)
    predictions = np.full(void_stack.shape, np.nan, dtype=np.float64)
    constant_surfaces = np.full(void_stack.shape, np.nan, dtype=np.float64)
    filled = np.repeat(dhdt[np.newaxis, ...], n_realizations, axis=0)
    diagnostics: list[dict[str, object]] = []

    for realization_index, void_mask in enumerate(void_stack):
        observed_mask = valid_glacier & ~void_mask
        target_mask = valid_glacier & void_mask

        n_observed = int(observed_mask.sum())
        n_void = int(target_mask.sum())
        if n_observed == 0:
            raise ValueError(
                f"Realization {realization_index} has no finite nonvoid glacier pixels"
            )

        constant_value = float(np.mean(dhdt[observed_mask], dtype=np.float64))
        constant_values[realization_index] = constant_value
        predictions[realization_index, target_mask] = constant_value
        constant_surfaces[realization_index, valid_glacier] = constant_value
        filled[realization_index, target_mask] = constant_value

        # Internal checks: the observed values must remain unchanged and all
        # finite target pixels must receive exactly the realization mean.
        if not np.array_equal(
            filled[realization_index, observed_mask],
            dhdt[observed_mask],
            equal_nan=True,
        ):
            raise AssertionError("Observed nonvoid values changed during filling")
        if n_void and not np.allclose(
            predictions[realization_index, target_mask],
            constant_value,
            rtol=0.0,
            atol=0.0,
        ):
            raise AssertionError("Not all void pixels received the constant value")

        diagnostics.append(
            {
                "realization_index": realization_index,
                "realization": f"r{realization_index + 1}",
                "n_nonvoid_pixels": n_observed,
                "n_void_pixels": n_void,
                "constant_value": constant_value,
            }
        )

    output = dict(chip)
    output["constant_value"] = constant_values.astype(output_dtype)
    output["constant_prediction"] = predictions.astype(output_dtype)
    output["constant_surface"] = constant_surfaces.astype(output_dtype)
    output["dhdt_filled_constant"] = filled.astype(output_dtype)
    return output, diagnostics


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as src:
        return {key: src[key] for key in src.files}


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "file",
        "rgiid",
        "realization_index",
        "realization",
        "n_nonvoid_pixels",
        "n_void_pixels",
        "constant_value",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scalar_string(value: np.ndarray | object, fallback: str) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return fallback
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode()
    return str(item)


def main() -> None:
    args = parse_args()
    input_files = sorted(args.chip_dir.glob(args.glob))
    if not input_files:
        raise FileNotFoundError(
            f"No files matching {args.glob!r} were found in {args.chip_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_dtype = np.dtype(args.dtype)
    manifest_rows: list[dict[str, object]] = []
    written = 0
    skipped = 0

    for index, input_path in enumerate(input_files, start=1):
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        chip = load_npz(input_path)
        output, diagnostics = reconstruct_chip(chip, output_dtype)
        np.savez_compressed(output_path, **output)

        glacier_id = scalar_string(
            chip.get("RGIId", chip.get("rgiid", input_path.stem)),
            input_path.stem,
        )
        for row in diagnostics:
            manifest_rows.append(
                {"file": output_path.name, "rgiid": glacier_id, **row}
            )

        written += 1
        if index == 1 or index % 250 == 0 or index == len(input_files):
            print(f"[{index}/{len(input_files)}] wrote {output_path.name}")

    manifest_path = args.output_dir.parent / "constant_nonvoid_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(
        f"Finished: wrote {written} glaciers, skipped {skipped}; "
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()

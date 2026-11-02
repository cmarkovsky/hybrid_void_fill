#!/usr/bin/env python3
'Prepare one all-pixel row shard per glacier for residual XGBoost.'

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


GLACIER_ATTRIBUTE_COLUMNS = ("Area", "Zmin", "Zmed", "Zmax", "HI")


def load_hypsometric_module(path: Path, module_name: str = "hypsometric_realization_specific") -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {path}")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load module specification from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def load_npz(path: Path, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    'Read numeric arrays from a compressed glacier shard.'
    with np.load(path, allow_pickle=allow_pickle) as source:
        return {name: source[name] for name in source.files}


def load_split(path: Path) -> dict[str, str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, str] = {}
    for glacier_id, value in mapping.items():
        group = str(value).strip().lower()
        if group not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {value!r} for {glacier_id}")
        output[str(glacier_id)] = group
    return output


def load_glacier_attributes(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(path)
    if "RGIId" in frame:
        id_column = "RGIId"
    elif "rgiid" in frame:
        id_column = "rgiid"
    else:
        raise ValueError("Attribute CSV must contain RGIId or rgiid")

    required = {"Area", "Zmin", "Zmed", "Zmax"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Attribute CSV is missing columns: {missing}")

    if "HI" not in frame:
        denominator = frame["Zmed"] - frame["Zmin"]
        with np.errstate(divide="ignore", invalid="ignore"):
            hi = (frame["Zmax"] - frame["Zmed"]) / denominator
            hi = np.where(hi < 1.0, -1.0 / hi, hi)
        frame["HI"] = hi

    if frame[id_column].duplicated().any():
        raise ValueError("Duplicate glacier ids in the attribute CSV")
    values = frame[list(GLACIER_ATTRIBUTE_COLUMNS)].astype(float)
    bad = ~np.isfinite(values.to_numpy()).all(axis=1)
    if bad.any():
        first = frame.loc[bad, id_column].iloc[0]
        raise ValueError(f"Nonfinite glacier attributes for {int(bad.sum())} glaciers, e.g. {first}")
    return {
        str(glacier_id): dict(zip(GLACIER_ATTRIBUTE_COLUMNS, row))
        for glacier_id, row in zip(frame[id_column], values.itertuples(index=False, name=None))
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


CUBE_MANIFEST_SUFFIX = ".content_manifest.json"


def write_cube_content_manifest(cube: Path) -> Path:
    """SHA256, size and mtime of every file of the cube, written once next to it (never inside it)."""
    cube = Path(cube)
    files = {str(f.relative_to(cube)): [f.stat().st_size, f.stat().st_mtime_ns, file_sha256(f)]
             for f in sorted(cube.rglob("*")) if f.is_file()}
    out = Path(f"{cube}{CUBE_MANIFEST_SUFFIX}")
    if out.exists():
        raise FileExistsError(f"{out} exists; verify it with --verify-cube-content instead of rewriting it")
    out.write_text(json.dumps({"cube": str(cube.resolve()), "files": files}, indent=0))
    return out


def cube_fingerprint(cube: Path, verify_content: bool = False, require_manifest: bool = False) -> str:
    """Identity of the cube's content.

    With a content manifest (``<cube>.content_manifest.json``), every file's size and mtime is checked
    (cheap, every run); ``verify_content`` rehashes every file, which also accepts a copied cube whose
    mtimes changed but whose bytes did not. Without a manifest the provenance record is used and the
    fingerprint is marked unverified (rejected when ``require_manifest``).
    """
    cube = Path(cube)
    manifest = Path(f"{cube}{CUBE_MANIFEST_SUFFIX}")
    if manifest.exists():
        recorded = json.loads(manifest.read_text())["files"]
        present = {str(f.relative_to(cube)): f for f in sorted(cube.rglob("*")) if f.is_file()}
        if set(present) != set(recorded):
            raise ValueError(f"{cube}: files added or removed since {manifest.name} "
                             f"({sorted(set(present) ^ set(recorded))[:3]}); the cube changed")
        for name, path in present.items():
            size, mtime, digest = recorded[name]
            st = path.stat()
            if (st.st_size, st.st_mtime_ns) != (size, mtime) or verify_content:
                if not verify_content:
                    raise ValueError(f"{cube}/{name} changed size or mtime since {manifest.name}; "
                                     "rerun with --verify-cube-content to check its bytes")
                if file_sha256(path) != digest:
                    raise ValueError(f"{cube}/{name}: content differs from {manifest.name}; the cube changed")
        return "manifest:" + file_sha256(manifest)
    if require_manifest:
        raise FileNotFoundError(f"{manifest} is required by the profile; create it with --write-cube-manifest")
    for candidate in (Path(f"{cube}.provenance.json"), cube / "zarr.json"):
        if candidate.exists():
            return "unverified-provenance:" + file_sha256(candidate)
    raise FileNotFoundError(f"No content manifest, provenance or zarr.json for cube {cube}")


def environment_identity(neighbor_rule: str) -> str:
    """The deterministic local-source rule is independent of runtime traversal order."""
    return ""


def source_fingerprint(chip_path: Path, cube_sha: str, code_sha: str, policy: str, context: bool, allow_debris: bool,
                       neighbor_rule: str = "v3", attributes: dict | None = None, environment: str = "") -> str:
    """Identity of everything a shard is built from: chip bytes, cube content, code, settings,
    the glacier's attribute values and (where it matters) the runtime platform."""
    parts = [file_sha256(chip_path), cube_sha, code_sha, policy, f"context={context}", f"debris={allow_debris}",
             f"neighbors={neighbor_rule}", "attributes=" + json.dumps(attributes, sort_keys=True), f"environment={environment}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def code_fingerprint(hypsometric_script: Path) -> str:
    files = [Path(__file__), Path(hypsometric_script), Path(__file__).resolve().parent / "context_features.py"]
    return hashlib.sha256("|".join(file_sha256(f) for f in files).encode()).hexdigest()


def attach_context(table, chip, glacier_id, glacier_attributes, hypsometric_module, context_module, neighbor_rule="v3") -> None:
    """Add ``context_rN`` arrays for each realization's voided rows (in row order)."""
    glacier = np.asarray(chip["glacier_mask"], dtype=bool)
    flat = table["pixel_flat_index"]
    # Inventory elevations exactly as stored in the float32 feature matrix (pilot convention).
    zmin = float(np.float32(glacier_attributes["Zmin"]))
    zmax = float(np.float32(glacier_attributes["Zmax"]))
    n_context = len(context_module.CONTEXT_FEATURES)
    for realization in range(int(table["n_realizations"])):
        rows = np.flatnonzero(table[f"is_void_r{realization}"])
        if not len(rows):
            table[f"context_r{realization}"] = np.empty((0, n_context), np.float32)
            continue
        void = np.asarray(chip["void_mask"][realization], dtype=bool)
        observed = context_module.mask_observations(chip["dhdt"], glacier, void)
        context, h = context_module.context_features(chip["z"], glacier, void, observed, chip["x"], chip["y"],
                                                     flat[rows], zmin, zmax, hypsometric_module, neighbor_rule=neighbor_rule)
        stored = table[f"hypso_fill_r{realization}"][rows]
        if not np.allclose(h, stored, rtol=0, atol=1e-5):
            raise ValueError(f"{glacier_id}/r{realization}: context hypsometry differs from the baseline "
                             f"by {float(np.max(np.abs(h - stored))):.3g} m/year")
        table[f"context_r{realization}"] = context
    table["context_feature_names"] = np.asarray(context_module.CONTEXT_FEATURES, dtype=str)
    table["context_neighbor_rule"] = np.array(neighbor_rule)


def coordinate_grids(chip: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    dhdt = np.asarray(chip["dhdt"])
    x = np.asarray(chip["x"])
    y = np.asarray(chip["y"])
    if x.ndim == 1 and y.ndim == 1:
        if x.size != dhdt.shape[1] or y.size != dhdt.shape[0]:
            raise ValueError("One-dimensional x/y coordinates do not match dhdt")
        return np.meshgrid(x, y)
    if x.shape == dhdt.shape and y.shape == dhdt.shape:
        return x, y
    raise ValueError(f"Unsupported x/y shapes: x={x.shape}, y={y.shape}")


CUBE_VARIABLES = ("slope", "aspect", "debris_thickness")


def _decode_fill(value, dtype) -> float | None:
    """xarray stores _FillValue as base64 bytes in Zarr v3 attributes."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = base64.b64decode(value)
        return float(np.frombuffer(raw, dtype="<f8" if len(raw) == 8 else "<f4")[0])
    return float(value)


class CubeReader:
    """Cube windows read directly from Zarr with an LRU cache of decoded chunks.

    Values are CF-decoded as xarray would (_FillValue -> NaN, scale/offset), so the
    result equals ``xr.open_zarr(cube)[var].isel(...)`` for these variables.
    """

    def __init__(self, path: Path, variables=CUBE_VARIABLES, cache_chunks: int = 24):
        import zarr

        group = zarr.open_group(str(path), mode="r")
        self.x = np.asarray(group["x"][:], dtype=np.float64)
        self.y = np.asarray(group["y"][:], dtype=np.float64)
        self.arrays = {}
        for name in variables:
            try:
                self.arrays[name] = group[name]
            except KeyError:
                continue
        missing = {"slope", "aspect"} - set(self.arrays)
        if missing:
            raise KeyError(f"Datacube is missing required variables {sorted(missing)}")
        self.decode = {}
        for name, array in self.arrays.items():
            attrs = dict(array.attrs)
            self.decode[name] = (
                _decode_fill(attrs.get("_FillValue"), array.dtype),
                float(attrs.get("scale_factor", 1.0)),
                float(attrs.get("add_offset", 0.0)),
            )
        self.cache: OrderedDict = OrderedDict()
        self.cache_chunks = cache_chunks

    @property
    def chunk_shape(self) -> tuple[int, int]:
        return tuple(self.arrays["slope"].chunks)

    def locate(self, x: np.ndarray, y: np.ndarray) -> tuple[slice, slice]:
        """Index window of a chip; chip x/y must be exactly this cube's pixel centres."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        c0 = int(np.searchsorted(self.x, x[0]))
        r0 = int(np.searchsorted(-self.y, -y[0]))
        cols, rows = slice(c0, c0 + len(x)), slice(r0, r0 + len(y))
        if not (np.array_equal(self.x[cols], x) and np.array_equal(self.y[rows], y)):
            raise ValueError(
                "Chip x/y are not pixel centres of this cube; carry the chips onto the "
                "cube first (workflow.py remap-chips)"
            )
        return rows, cols

    def _chunk(self, name: str, cy: int, cx: int) -> np.ndarray:
        key = (name, cy, cx)
        block = self.cache.get(key)
        if block is not None:
            self.cache.move_to_end(key)
            return block
        h, w = self.arrays[name].chunks
        block = np.asarray(self.arrays[name][cy * h:(cy + 1) * h, cx * w:(cx + 1) * w])
        fill, scale, offset = self.decode[name]
        block = block.astype(np.float64 if scale != 1.0 or offset != 0.0 else block.dtype, copy=True)
        if fill is not None and np.isfinite(fill):
            block[block == fill] = np.nan
        if scale != 1.0 or offset != 0.0:
            block = block * scale + offset
        self.cache[key] = block
        if len(self.cache) > self.cache_chunks:
            self.cache.popitem(last=False)
        return block

    def window(self, name: str, rows: slice, cols: slice) -> np.ndarray:
        h, w = self.arrays[name].chunks
        out = np.empty((rows.stop - rows.start, cols.stop - cols.start), dtype=np.float64)
        for cy in range(rows.start // h, (rows.stop - 1) // h + 1):
            for cx in range(cols.start // w, (cols.stop - 1) // w + 1):
                block = self._chunk(name, cy, cx)
                r0, r1 = max(rows.start, cy * h), min(rows.stop, (cy + 1) * h)
                c0, c1 = max(cols.start, cx * w), min(cols.stop, (cx + 1) * w)
                out[r0 - rows.start:r1 - rows.start, c0 - cols.start:c1 - cols.start] = block[
                    r0 - cy * h:r1 - cy * h, c0 - cx * w:c1 - cx * w
                ]
        return out


def extract_datacube_window(dataset, chip: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Slope/aspect (and debris if present) at the chip pixels.

    ``dataset`` may be a CubeReader (fast path) or an xarray Dataset.
    """
    x = np.asarray(chip["x"])
    y = np.asarray(chip["y"])
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Datacube extraction requires one-dimensional x/y coordinates")
    if isinstance(dataset, CubeReader):
        rows, cols = dataset.locate(x, y)
        return {name: dataset.window(name, rows, cols) for name in dataset.arrays}
    names = [name for name in CUBE_VARIABLES if name in dataset]
    missing = {"slope", "aspect"} - set(names)
    if missing:
        raise KeyError(f"Datacube is missing required variables {sorted(missing)}")
    subset = dataset[names].sel(x=x, y=y).load()   # exact labels; KeyError if a pixel is absent
    expected_shape = (len(y), len(x))
    output = {name: np.asarray(subset[name].values) for name in names}
    for name, values in output.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}")
    return output


def build_feature_matrix(
    chip: dict[str, np.ndarray],
    datacube_arrays: dict[str, np.ndarray],
    valid: np.ndarray,
    glacier_attributes: dict[str, float],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    dhdt = np.asarray(chip["dhdt"])
    z = np.asarray(chip["z"], dtype=np.float64)
    if z.shape != dhdt.shape or valid.shape != dhdt.shape:
        raise ValueError("z/valid shape does not match dhdt")

    x_grid, y_grid = coordinate_grids(chip)
    x_scale = float(np.std(x_grid))
    y_scale = float(np.std(y_grid))
    x_relative = (x_grid - np.mean(x_grid)) / (x_scale if x_scale > 0 else 1.0)
    y_relative = (y_grid - np.mean(y_grid)) / (y_scale if y_scale > 0 else 1.0)

    slope = np.asarray(datacube_arrays["slope"], dtype=np.float64)
    aspect = np.asarray(datacube_arrays["aspect"], dtype=np.float64)
    aspect_radians = np.deg2rad(aspect)
    sin_aspect = np.sin(aspect_radians)
    cos_aspect = np.cos(aspect_radians)
    # A perfectly flat cell (slope exactly 0) has no aspect: encode it as the zero
    # direction vector rather than as missing. Any other missing aspect stays NaN.
    flat = (slope == 0) & ~np.isfinite(aspect)
    sin_aspect[flat] = 0.0
    cos_aspect[flat] = 0.0

    columns = [
        x_grid[valid],
        y_grid[valid],
        x_relative[valid],
        y_relative[valid],
        z[valid],
        slope[valid],
        sin_aspect[valid],
        cos_aspect[valid],
    ]
    names = ["x", "y", "x_rel", "y_rel", "z", "slope", "sinaspect", "cosaspect"]

    if "debris_thickness" in datacube_arrays:
        debris = np.asarray(datacube_arrays["debris_thickness"], dtype=np.float64)
        has_debris = np.isfinite(debris) & (debris > 0)
        columns.extend(
            [
                np.where(np.isfinite(debris), debris, 0.0)[valid],
                has_debris.astype(np.float64)[valid],
            ]
        )
        names.extend(["debris_thickness", "has_debris"])

    row_count = int(np.sum(valid))
    for attribute in GLACIER_ATTRIBUTE_COLUMNS:
        columns.append(np.full(row_count, glacier_attributes[attribute]))
        names.append(f"g_{attribute}")

    matrix = np.column_stack(columns).astype(np.float32)
    flat_indices = np.flatnonzero(valid.ravel()).astype(np.int64)
    return (
        matrix,
        names,
        np.asarray(x_grid[valid], dtype=np.float64),
        np.asarray(y_grid[valid], dtype=np.float64),
        flat_indices,
    )


def prepare_glacier_rows(
    chip: dict[str, np.ndarray],
    datacube_arrays: dict[str, np.ndarray],
    glacier_id: str,
    group: str,
    glacier_attributes: dict[str, float],
    hypsometric_module: ModuleType,
    missing_predictors: str = "error",
    context_module: ModuleType | None = None,
    allow_debris: bool = True,
    neighbor_rule: str = "v3",
) -> dict[str, np.ndarray] | None:
    required = {"dhdt", "z", "glacier_mask", "void_mask", "x", "y"}
    missing = sorted(required - set(chip))
    if missing:
        raise KeyError(f"{glacier_id}: chip is missing arrays {missing}")

    dhdt = np.asarray(chip["dhdt"], dtype=np.float64)
    z = np.asarray(chip["z"], dtype=np.float64)
    glacier_mask = np.asarray(chip["glacier_mask"], dtype=bool)
    void_masks = np.asarray(chip["void_mask"], dtype=bool)
    if glacier_mask.shape != dhdt.shape or z.shape != dhdt.shape:
        raise ValueError(f"{glacier_id}: glacier_mask/z shape mismatch")
    if void_masks.ndim != 3 or tuple(void_masks.shape[1:]) != dhdt.shape:
        raise ValueError(f"{glacier_id}: void_mask shape mismatch: {void_masks.shape}")

    if not allow_debris and "debris_thickness" in datacube_arrays:
        raise ValueError(f"{glacier_id}: the profile is debris-free (debris_features=false) but the cube provides "
                         "debris_thickness; use a cube without debris predictors")
    valid = glacier_mask & np.isfinite(dhdt) & np.isfinite(z)
    if not np.any(valid):
        return None

    matrix, feature_names, x_values, y_values, flat_indices = build_feature_matrix(
        chip,
        datacube_arrays,
        valid,
        glacier_attributes,
    )
    response = dhdt[valid].astype(np.float32)

    config = hypsometric_module.HypsometricConfig()
    hypso = hypsometric_module.hypsometric_local_mean_fill_chip(
        chip,
        glacier_id,
        config,
    )
    predictions = np.asarray(hypso["hypsometric_prediction"], dtype=np.float32)
    if predictions.shape != void_masks.shape:
        raise ValueError(
            f"{glacier_id}: hypsometric prediction shape {predictions.shape} "
            f"does not match void masks {void_masks.shape}"
        )

    if missing_predictors not in {"error", "drop", "native"}:
        raise ValueError(f"Unknown missing-predictor policy: {missing_predictors}")
    if np.isinf(matrix).any():
        raise ValueError(f"{glacier_id}: infinite predictor values; inspect the source layers")
    finite_features = np.all(np.isfinite(matrix), axis=1)
    target_rows = np.any(void_masks[:, valid], axis=0)
    if missing_predictors == "error" and np.any(target_rows & ~finite_features):
        raise ValueError(
            f"{glacier_id}: {int(np.sum(target_rows & ~finite_features))} intended void pixels "
            "lack predictors. Repair the source rasters or explicitly use the native missing-predictor "
            "profile and refit the model; drop is available only for an explicit reduced-support sensitivity analysis."
        )
    keep = np.isfinite(response) & np.isfinite(x_values) & np.isfinite(y_values)
    if missing_predictors in {"error", "drop"}:
        keep &= finite_features
    # Do not discard a row in every realization because one realization's baseline
    # is unavailable. The trainer validates the baseline separately for each mask.
    if missing_predictors == "drop":
        for realization in range(void_masks.shape[0]):
            keep &= np.isfinite(predictions[realization][valid])
    else:
        for realization in range(void_masks.shape[0]):
            missing_baseline = void_masks[realization][valid] & ~np.isfinite(predictions[realization][valid])
            if np.any(missing_baseline):
                raise ValueError(f"{glacier_id}/r{realization}: intended target lacks a hypsometric baseline")
    if not np.any(keep):
        return None

    table: dict[str, np.ndarray] = {
        "X": matrix[keep],
        "y": response[keep],
        "feature_names": np.asarray(feature_names, dtype=str),
        "rgiid": np.full(int(np.sum(keep)), glacier_id, dtype=f"<U{len(glacier_id)}"),
        "group": np.full(int(np.sum(keep)), group, dtype="<U5"),
        "x_coord": x_values[keep],
        "y_coord": y_values[keep],
        "pixel_flat_index": flat_indices[keep],
        "n_realizations": np.array(void_masks.shape[0], dtype=np.int16),
        "fit_source_mask": np.array("realization_specific_nonvoid"),
        "missing_predictor_policy": np.array(missing_predictors),
    }
    for realization in range(void_masks.shape[0]):
        table[f"is_void_r{realization}"] = void_masks[realization][valid][keep].astype(
            np.uint8
        )
        table[f"hypso_fill_r{realization}"] = predictions[realization][valid][keep]
    if context_module is not None:
        attach_context(table, chip, glacier_id, glacier_attributes, hypsometric_module, context_module, neighbor_rule)
    return table


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=workflow_path('cube'))
    parser.add_argument("--chip-dir", type=Path, default=workflow_path('chips'))
    parser.add_argument("--split-json", type=Path, default=workflow_path('split'))
    parser.add_argument("--glacier-attributes", type=Path, default=workflow_path('attributes'))
    parser.add_argument("--output-dir", type=Path, default=workflow_path('rows'))
    parser.add_argument(
        "--hypsometric-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "03_baselines" / "02_hypsometric_local_mean.py",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-cube-content", action="store_true", help="Rehash every cube file against its content manifest")
    parser.add_argument("--write-cube-manifest", action="store_true", help="Write <cube>.content_manifest.json and exit")
    parser.add_argument("--context-features", action=argparse.BooleanOptionalAction,
                        default=bool(settings().get("context_features", False)),
                        help="Also store the 16 realization-specific context predictors per realization")
    parser.add_argument("--allow-debris", action=argparse.BooleanOptionalAction,
                        default=bool(settings().get("debris_features", True)),
                        help="Allow debris predictors if the cube has them (optional comparisons)")
    parser.add_argument("--neighbor-rule", choices=("v3",), default=settings().get("context_neighbor_rule", "v3"),
                        help="Deterministic source ordering within the strict 900 m radius")
    parser.add_argument("--missing-predictors", choices=("error", "drop", "native"), default=settings()["missing_predictors"])
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--cache-chunks", type=int, default=24, help="Decoded cube chunks cached per worker (~16 MB each)")
    return parser


_WORKER: dict = {}


CONTEXT_SCRIPT = Path(__file__).resolve().parent / "context_features.py"


def _init_worker(cube: str, hypsometric_script: str, attributes: str, cache_chunks: int, context: bool, allow_debris: bool,
                 neighbor_rule: str = "v3") -> None:
    _WORKER.update(
        reader=CubeReader(Path(cube), cache_chunks=cache_chunks),
        hyp=load_hypsometric_module(Path(hypsometric_script)),
        attrs=load_glacier_attributes(Path(attributes)),
        context=load_hypsometric_module(CONTEXT_SCRIPT, "rc_context_features") if context else None,
        allow_debris=allow_debris,
        neighbor_rule=neighbor_rule,
    )


def _prepare_batch(batch: list[tuple[str, str, str, str, str]]) -> list[dict[str, object]]:
    """Prepare and save one spatially contiguous batch; returns manifest records."""
    records = []
    for glacier_id, group, chip_path, output_path, policy, fingerprint in batch:
        try:
            chip = load_npz(Path(chip_path))
            table = prepare_glacier_rows(
                chip,
                extract_datacube_window(_WORKER["reader"], chip),
                glacier_id,
                group,
                _WORKER["attrs"][glacier_id],
                _WORKER["hyp"],
                missing_predictors=policy,
                context_module=_WORKER["context"],
                allow_debris=_WORKER["allow_debris"],
                neighbor_rule=_WORKER["neighbor_rule"],
            )
            if table is None:
                raise ValueError("No complete finite rows after filtering")
            table["source_fingerprint"] = np.array(fingerprint)
            save_npz_atomic(Path(output_path), table)
            n_realizations = int(np.asarray(table["n_realizations"]).item())
            records.append({
                "rgiid": glacier_id, "group": group, "status": "ok", "n_rows": len(table["y"]),
                "n_realizations": n_realizations,
                "n_expanded_void_rows": int(sum(np.sum(table[f"is_void_r{r}"]) for r in range(n_realizations))),
                "context_features": "context_r0" in table,
                "feature_names": "|".join(str(v) for v in table["feature_names"]), "file": output_path,
            })
        except Exception as exc:
            records.append({"rgiid": glacier_id, "group": group, "status": "error", "error": repr(exc), "file": ""})
    return records


def plan_tasks(args, split, attributes, reader: CubeReader):
    """Manifest records for skipped glaciers and spatially sorted tasks for the rest."""
    records, tasks = [], []
    cube_sha = cube_fingerprint(args.cube, args.verify_cube_content, bool(settings().get("require_cube_content_manifest", False)))
    code_sha, env_id = code_fingerprint(args.hypsometric_script), environment_identity(args.neighbor_rule)
    for glacier_id in sorted(split):
        group = split[glacier_id]
        chip_path = args.chip_dir / f"{glacier_id}.npz"
        output_path = args.output_dir / group / f"{glacier_id}.npz"
        expected = source_fingerprint(chip_path, cube_sha, code_sha, args.missing_predictors, args.context_features, args.allow_debris,
                                      args.neighbor_rule, attributes.get(glacier_id), env_id) \
            if chip_path.exists() else None
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=True) as existing:
                stored = str(existing["source_fingerprint"].item()) if "source_fingerprint" in existing.files else None
                old_policy = str(existing["missing_predictor_policy"].item()) if "missing_predictor_policy" in existing.files else "drop"
                has_context = "context_r0" in existing.files
            if old_policy != args.missing_predictors:
                raise ValueError(f"{output_path}: policy {old_policy} differs from {args.missing_predictors}; choose a fresh rows directory")
            if has_context != args.context_features:
                raise ValueError(f"{output_path}: context features {'present' if has_context else 'absent'} but "
                                 f"--{'' if args.context_features else 'no-'}context-features requested; choose a fresh rows directory")
            if stored != expected:
                # Chip, cube, code or settings changed (or the shard predates fingerprints): never reuse silently.
                raise ValueError(f"{output_path}: stale or unfingerprinted shard (stored {stored}, expected {expected}); "
                                 "choose a fresh rows directory or pass --overwrite")
            records.append({"rgiid": glacier_id, "group": group, "status": "exists", "file": str(output_path)})
            continue
        if not chip_path.exists():
            records.append({"rgiid": glacier_id, "group": group, "status": "missing_chip", "file": ""})
            continue
        if glacier_id not in attributes:
            records.append({"rgiid": glacier_id, "group": group, "status": "missing_attributes", "file": ""})
            continue
        with np.load(chip_path) as chip:
            try:
                rows, cols = reader.locate(chip["x"], chip["y"])
            except ValueError as exc:
                records.append({"rgiid": glacier_id, "group": group, "status": "error", "error": repr(exc), "file": ""})
                continue
        h, w = reader.chunk_shape
        tasks.append(((rows.start // h, cols.start // w, glacier_id),
                      (glacier_id, group, str(chip_path), str(output_path), args.missing_predictors, expected)))
    tasks.sort(key=lambda item: item[0])
    return records, [task for _, task in tasks]


def main() -> None:
    args = parse_stage_args(build_parser())
    if args.write_cube_manifest:
        print(f"Wrote {write_cube_content_manifest(args.cube)}")
        return
    split = load_split(args.split_json)
    attributes = load_glacier_attributes(args.glacier_attributes)
    load_hypsometric_module(args.hypsometric_script)   # fail early if missing
    for group in {"train", "val", "test"}:
        (args.output_dir / group).mkdir(parents=True, exist_ok=True)

    reader = CubeReader(args.cube, cache_chunks=args.cache_chunks)
    manifest, tasks = plan_tasks(args, split, attributes, reader)
    del reader
    # Contiguous spatial batches keep each worker's chunk cache warm.
    n_batches = max(1, min(len(tasks), args.workers * 8))
    bounds = np.linspace(0, len(tasks), n_batches + 1).astype(int)
    batches = [tasks[bounds[i]:bounds[i + 1]] for i in range(n_batches) if bounds[i + 1] > bounds[i]]
    print(f"Preparing {len(tasks):,} glaciers in {len(batches)} spatial batches with {args.workers} workers "
          f"({len(manifest)} already present or skipped)", flush=True)
    done = 0
    with ProcessPoolExecutor(
        args.workers,
        initializer=_init_worker,
        initargs=(str(args.cube), str(args.hypsometric_script), str(args.glacier_attributes), args.cache_chunks,
                  args.context_features, args.allow_debris, args.neighbor_rule),
    ) as pool:
        for records in pool.map(_prepare_batch, batches):
            manifest.extend(records)
            done += len(records)
            errors = [r for r in records if r["status"] == "error"]
            for r in errors[:3]:
                print(f"  {r['rgiid']}: ERROR {r['error']}", flush=True)
            print(f"  {done:,}/{len(tasks):,} glaciers prepared", flush=True)

    frame = pd.DataFrame(manifest).sort_values("rgiid")
    feature_sets = set(frame.get("feature_names", pd.Series(dtype=str)).dropna())
    if len(feature_sets) > 1:
        raise ValueError(f"Feature mismatch between glaciers: {sorted(feature_sets)}")
    reference_features = feature_sets.pop().split("|") if feature_sets else None
    frame.drop(columns=["feature_names"], errors="ignore").to_csv(args.output_dir / "manifest.csv", index=False)
    failures = int(frame.status.isin(["error", "missing_chip", "missing_attributes"]).sum())
    if reference_features is not None:
        (args.output_dir / "feature_names.json").write_text(json.dumps(reference_features, indent=2) + "\n", encoding="utf-8")
    run_metadata = {
        "row_design": "one row per finite on-glacier pixel",
        "training_selection": "artificially voided pixels per realization",
        "fit_source_mask": "realization_specific_nonvoid",
        "missing_predictor_policy": args.missing_predictors,
        "coordinate_features_stored": True,
        "coordinate_features_primary_model": False,
        "cube": str(args.cube),
        "chip_dir": str(args.chip_dir),
        "feature_names": reference_features,
        "context_features": args.context_features,
        "debris_features_allowed": args.allow_debris,
        "context_neighbor_rule": args.neighbor_rule if args.context_features else None,
        "cube_fingerprint": cube_fingerprint(args.cube, require_manifest=bool(settings().get("require_cube_content_manifest", False))),
        "environment_identity": environment_identity(args.neighbor_rule),
        "code_fingerprint": code_fingerprint(args.hypsometric_script),
        "context_feature_names": load_hypsometric_module(CONTEXT_SCRIPT, "rc_context_features").CONTEXT_FEATURES if args.context_features else None,
        "context_source_mask": "same glacier and realization: glacier & finite z & finite masked dh/dt & not void" if args.context_features else None,
        "number_of_glaciers": len(split),
        "number_of_failures": failures,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    write_provenance(args.output_dir, args, __file__)
    print(frame.groupby(["group", "status"]).size().to_string())
    if failures:
        raise RuntimeError(f"Preparation completed with {failures} failed glacier(s); inspect {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()

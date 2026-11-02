"""Read hybrid-model rows with equal glacier and equal realization fitting weights."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "hybrid_context", REPO / "workflow/src/void_fill/04_model/context_features.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)
sys.path.insert(0, str(REPO / "analysis/manuscript"))
from common import Paths, Selection


def glacier_weights(gidx, realization):
    """Return 1/(R_g * N_gr), normalized to mean one."""
    pair = np.asarray(gidx, np.int64) * 64 + realization
    _, first, inv, counts = np.unique(pair, return_index=True, return_inverse=True, return_counts=True)
    realizations_per_glacier = np.bincount(gidx[first], minlength=int(gidx.max()) + 1)
    weights = 1.0 / (realizations_per_glacier[gidx] * counts[inv])
    return weights / weights.mean()


def read_rows(data_root, group, glaciers=None, cap=0):
    """Expand frozen rows, using the manuscript evaluation by default for test predictions.

    Capped samples use a stable per-glacier, per-realization random seed. The full
    training population and row order are preserved for the primary hybrid fit.
    """
    paths, selection = Paths(data_root), Selection()
    ids = {g for g, assigned in selection.split.items() if assigned == group}
    if group == "test":
        ids &= set(selection.evaluated)
    if glaciers is not None:
        requested = set(glaciers)
        if not requested <= ids:
            raise ValueError(f"Requested glaciers outside {group}: {sorted(requested - ids)[:5]}")
        ids = requested
    ids = sorted(ids)
    manifest = pd.read_csv(paths["rows"] / "manifest.csv")
    count = int(manifest.loc[manifest.rgiid.isin(ids) & manifest.group.eq(group), "n_expanded_void_rows"].sum())
    if not count or not ids:
        raise ValueError(f"No {group} rows selected")
    # The full fit uses one preallocated feature matrix to limit peak memory.
    parts = []
    if not cap:
        data = {"X": np.empty((count, len(context.HYBRID_FEATURES)), np.float32),
                "y": np.empty(count), "h": np.empty(count),
                "realization": np.empty(count, np.int16), "pixel_flat_index": np.empty(count, np.int64),
                "gidx": np.empty(count, np.int32), "denominator": np.empty(count, np.int64),
                "x_coord": np.empty(count), "y_coord": np.empty(count)}
    offset = 0
    for i, gid in enumerate(ids):
        with np.load(paths["rows"] / group / f"{gid}.npz", allow_pickle=False) as loaded:
            shard = {key: loaded[key] for key in loaded.files}
        if str(shard["context_neighbor_rule"]) != "v3":
            raise ValueError(f"{gid}: incompatible local-source selection rule")
        expanded = context.expand_void_rows(shard)
        if cap:
            take = []
            for realization in np.unique(expanded["realization"]):
                available = np.flatnonzero(expanded["realization"] == realization)
                seed = 42 + int(hashlib.sha256(f"{gid}|{realization}".encode()).hexdigest()[:12], 16)
                take.extend(np.sort(np.random.default_rng(seed).choice(available, min(len(available), cap), replace=False)))
            expanded = {key: value[np.asarray(take)] for key, value in expanded.items()}
        length = len(expanded["y"])
        expanded["gidx"] = np.full(length, i, np.int32)
        expanded["denominator"] = np.full(length, len(shard["y"]), np.int64)
        if cap:
            parts.append(expanded)
        else:
            for key in data:
                data[key][offset:offset + length] = expanded[key]
        offset += length
    if cap:
        data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    elif offset != count:
        raise ValueError(f"Expanded {offset} targets; manifest specifies {count}")
    data["weight"] = glacier_weights(data["gidx"], data["realization"])
    data["rgiid"] = np.asarray(ids)[data["gidx"]]
    return data

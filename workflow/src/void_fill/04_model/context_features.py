'Construct the 25 hybrid predictors. All 16 context fields use only same-realization nonvoid observations. Local sources are strictly within 900 m, with deterministic distance and raster-index ordering.'
from __future__ import annotations

from types import ModuleType

import numpy as np
from scipy.spatial import cKDTree

STATIC_FEATURES = ["z", "slope", "sinaspect", "cosaspect", "g_Area", "g_Zmin", "g_Zmed", "g_Zmax", "g_HI"]
CONTEXT_FEATURES = [
    "hyp",                     # hypsometric prediction at the target (m/year)
    "normalized_z",            # (z - inventory Zmin) / (Zmax - Zmin); NaN for zero relief
    "band_count",              # observed source pixels in the target's elevation band
    "band_fraction",           # band_count / finite-z glacier pixels in that band
    "band_std",                # source-only dh/dt std in the band, ddof=1; NaN if <2 sources
    "band_empty",              # target band has no source pixels
    "outside_observed_z",      # target z outside [min, max] of source z
    "nearest_band_dz",         # |z - centre| to the nearest occupied band centre (m)
    "observed_span_fraction",  # source elevation span / glacier band range
    "void_fraction",           # voided share of finite-z glacier pixels
    "mode_none",               # missing-band fill mode of this realization's baseline
    "mode_cubic",
    "mode_linear",
    "nearest_source_m",        # distance to the nearest source pixel (m)
    "local_count",             # source pixels within 900 m (at most 32)
    "local_residual",          # exp(-d/300 m)-weighted mean source residual; NaN if <3 sources
]
HYBRID_FEATURES = STATIC_FEATURES + CONTEXT_FEATURES

# How the up-to-32 local sources are chosen (affects local_residual and local_count only).
NEIGHBOR_RULES = {"v3": 'Sources strictly within 900 m; up to 32 neighbours, ordered by distance with a 1e-6 m tie tolerance, then raster flat index. Residual weights exp(-d/300 m); fewer than three sources gives NaN.'}
# Tie tolerance: on the 30 m lattice two distinct source distances within 900 m differ by at least
# 30*(sqrt(n+1) - sqrt(n)) >= 0.5 m (n <= 900 squared-pixel offsets), while floating-point error is ~1e-10 m,
# so any tolerance between ~1e-9 m and 0.4 m gives the same tie groups. 1e-6 m is used.
TIE_TOLERANCE_M = 1e-6


def mask_observations(y, glacier, void):
    """dh/dt with every voided or off-glacier value hidden."""
    return np.where(glacier & ~void, np.asarray(y, dtype=float), np.nan)


def local_summaries(coords, src_res, src_flat, queries, rule="v3", k=32, radius=900.0, scale=300.0):
    """Nearest-source distance, local source count and exp(-d/300 m)-weighted mean residual.

    ``coords``/``src_res``/``src_flat``: source pixel centres, residuals and raster flat
    indices (row-major order). See ``NEIGHBOR_RULES`` for the selection rules.
    """
    if rule not in NEIGHBOR_RULES:
        raise ValueError(f"Unknown neighbor rule {rule!r}")
    tree = cKDTree(coords)
    near = np.empty(len(queries))
    num = np.empty(len(queries))
    local = np.full(len(queries), np.nan)
    tol = TIE_TOLERANCE_M
    for start in range(0, len(queries), 10000):
        stop = min(start + 10000, len(queries))
        q = queries[start:stop]
        near[start:stop] = tree.query(q, k=1, workers=1)[0]
        dist, j = tree.query(q, k=k + 1, distance_upper_bound=radius, workers=1)
        for i in range(len(q)):
            d, idx = dist[i], j[i]
            keep = np.isfinite(d)
            d, idx = d[keep], idx[keep]
            if len(d) == k + 1 and d[k] - d[k - 1] <= tol:
                # Rank-32 tie: retrieve every source up to the tied distance, then order explicitly.
                idx = np.asarray(tree.query_ball_point(q[i], d[k - 1] + tol), dtype=np.int64)
                delta = coords[idx] - q[i]
                dsq = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
                inside = dsq < radius * radius
                idx, d = idx[inside], np.sqrt(dsq[inside])
            by_distance = np.argsort(d, kind="stable")
            group = np.empty(len(d), np.int64)
            group[by_distance] = np.concatenate([[0], np.cumsum(np.diff(d[by_distance]) > tol)])
            order = np.lexsort((src_flat[idx], group))[:k]
            d, idx = d[order], idx[order]
            num[start + i] = len(d)
            if len(d) >= 3:
                w = np.exp(-d / scale)
                local[start + i] = float(np.dot(w, src_res[idx]) / w.sum())
    return near, num, local


def context_features(z, glacier, void, observed, x, y, targets, zmin, zmax, base: ModuleType, neighbor_rule="v3"):
    'Construct the 25 hybrid predictors. All 16 context fields use only same-realization nonvoid observations. Local sources are strictly within 900 m, with deterministic distance and raster-index ordering.'
    if np.any(np.isfinite(observed[void | ~glacier])):
        raise ValueError("Unmasked hidden observations")
    cfg = base.HypsometricConfig()
    edges, centers, low, high = base.make_elevation_bins(z, glacier, cfg)
    bins, inrange = base.digitize(z, edges)
    source = glacier & ~void & np.isfinite(z) & np.isfinite(observed)
    if not source.any():
        raise ValueError("No nonvoid sources")
    means, counts = base.elevation_band_means(z, observed, source, edges)
    span = float(np.ptp(z[source]) / (high - low))
    filled, mode = base.fill_unobserved_bands(centers, means, span, cfg)
    idx = bins.ravel()[targets]
    h = filled[idx].astype(np.float32)
    geom = glacier & np.isfinite(z) & inrange
    total = np.bincount(bins[geom], minlength=len(counts))
    ss = np.bincount(bins[source], weights=observed[source] ** 2, minlength=len(counts))
    std = np.full(len(counts), np.nan)
    ok = counts > 1
    std[ok] = np.sqrt(np.maximum(0, (ss[ok] - counts[ok] * means[ok] ** 2) / (counts[ok] - 1)))
    xx, yy = np.meshgrid(x, y) if np.ndim(x) == 1 else (x, y)
    coords = np.column_stack([xx[source], yy[source]])
    queries = np.column_stack([xx.ravel()[targets], yy.ravel()[targets]])
    src_res = observed[source] - filled[bins[source]]
    near, num, local = local_summaries(coords, src_res, np.flatnonzero(source.ravel()), queries, rule=neighbor_rule)
    zz = z.ravel()[targets]
    relief = zmax - zmin
    norm = (zz - zmin) / relief if relief > 0 else np.full(len(zz), np.nan)
    dz = np.min(np.abs(zz[:, None] - centers[counts > 0]), axis=1)
    empty = counts[idx] == 0
    modevals = [np.full(len(zz), mode == name) for name in ["none", "polynomial_3", "linear"]]
    columns = [h, norm, counts[idx], counts[idx] / np.maximum(total[idx], 1), std[idx], empty,
               (zz < z[source].min()) | (zz > z[source].max()), dz, np.full(len(zz), span),
               np.full(len(zz), (void & geom).sum() / geom.sum()), *modevals, near, num, local]
    return np.column_stack(columns).astype(np.float32), h


def expand_void_rows(shard: dict, features: list[str] = HYBRID_FEATURES) -> dict[str, np.ndarray]:
    """One row per (realization, voided pixel) from a context row shard.

    Returns ``X`` (features in the given order), ``y``, ``h`` (hypsometric baseline),
    ``realization``, ``pixel_flat_index`` and pixel-centre ``x_coord``/``y_coord``.
    The residual target is ``y - h``.
    """
    names = [str(v) for v in shard["feature_names"]]
    context_names = [str(v) for v in shard["context_feature_names"]]
    n_real = int(np.asarray(shard["n_realizations"]).item())
    parts = []
    for r in range(n_real):
        rows = np.flatnonzero(shard[f"is_void_r{r}"])
        cols = {n: shard["X"][rows, names.index(n)] for n in names}
        cols.update(zip(context_names, shard[f"context_r{r}"].T))
        parts.append(dict(X=np.column_stack([cols[f] for f in features]).astype(np.float32),
                          y=shard["y"][rows].astype(np.float64),
                          h=shard[f"hypso_fill_r{r}"][rows].astype(np.float64),
                          realization=np.full(len(rows), r, np.int16),
                          pixel_flat_index=shard["pixel_flat_index"][rows],
                          x_coord=shard["x_coord"][rows], y_coord=shard["y_coord"][rows]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

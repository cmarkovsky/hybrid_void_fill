#!/usr/bin/env python3
'Recalculate manuscript point metrics, paired spatial-block intervals, and sensitivity summaries. The default evaluation excludes RGI60-14.17287; archived full-population values are checked before selection.'
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import finalize_evaluation_outputs, METHODS, Paths, Selection, base_parser, interval, nmad, norm_method, write_json


def weighted_quantile(values, weights, q):
    # Step weighted quantile of 05_spatial_block_bootstrap.py / 03_fig3 (identical definitions).
    order = np.argsort(values, kind="mergesort")
    v, w = values[order], weights[order]
    i = min(int(np.searchsorted(np.cumsum(w), q * w.sum(), side="left")), len(v) - 1)
    return float(v[i])


def pixel_summary(r, w):
    med = weighted_quantile(r, w, 0.5)
    return dict(nmad=1.4826 * weighted_quantile(np.abs(r - med), w, 0.5),
                rmse=float(np.sqrt(np.average(r ** 2, weights=w))), bias=float(np.average(r, weights=w)))


def mults(k, seed, draws=2000):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(draws, k))
    return np.stack([np.bincount(x, minlength=k) for x in pick])


def main() -> None:
    a = base_parser(__doc__).parse_args()
    paths, sel = Paths(a.data_root, a.repo), Selection(a.repo)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    X = sel.excluded
    tree, rev = paths["reference_results"], paths["sensitivity_metrics"]

    con = pd.read_csv(rev / "per_glacier_error_contributions.csv")
    assign = pd.read_csv(tree / "spatial_block_bootstrap/block_assignments.csv")
    pg = pd.read_csv(tree / "pixel_summaries/per_glacier_metrics_test_common_xy.csv")
    pg["method"] = pg.method.map(norm_method)
    pr = pd.read_csv(tree / "pixel_summaries/per_glacier_realization_metrics_test_common_xy.csv")
    pr["method"] = pr.method.map(norm_method)
    pg = pg.merge(pr.groupby(["rgiid", "method"]).mse.mean().rename("mean_mse"), on=["rgiid", "method"], validate="one_to_one")
    per = pd.read_csv(tree / "glacier_wide_metrics/per_glacier_realization_area_normalized_metrics.csv")
    per["method"] = per.method.map(norm_method)
    old = pd.read_csv(tree / "glacier_wide_metrics/glacier_balanced_area_normalized_summary.csv")
    old["method"] = old.method.map(norm_method)
    checks = {}

    print("Calculating glacier and pixel moments", flush=True)
    points = []
    for scope in ["whim", "ehim", "all"]:
        for method in METHODS:
            stem = "hybrid" if method == "xgb" else method
            for excluded in [False, True]:
                s = con.copy() if scope == "all" else con.loc[con.region.eq(scope)].copy()
                t = pg.loc[pg.method.eq(method)].copy()
                if scope != "all":
                    t = t.loc[t.region.eq(scope)]
                if excluded:
                    s, t = s.loc[~s.rgiid.isin(X)], t.loc[~t.rgiid.isin(X)]
                A = s.valid_area_km2.to_numpy() * 1e6
                for kind in ["specific", "volume"]:
                    if kind == "specific":
                        means = s["mean_spec_" + stem]
                        mae, mse = s["mae_spec_" + stem].mean(), np.mean(s["msq_vol_" + stem].to_numpy() / A ** 2)
                    else:
                        means = s["mean_vol_" + stem]
                        mae, mse = np.mean(s["mae_spec_" + stem].to_numpy() * A), s["msq_vol_" + stem].mean()
                    row = dict(scope=scope, method=method, excluded=excluded, kind=kind, n_glaciers=len(s),
                               mae=mae, rmse=np.sqrt(mse), bias=means.mean(), nmad=nmad(means))
                    if not excluded:
                        metric = "glacier_wide_specific_error_m_per_yr" if kind == "specific" else "void_integrated_error_m3_per_yr"
                        e = old.loc[old.region.eq(scope) & old.method.eq(method) & old.metric.eq(metric)].iloc[0]
                        for m, col in [("mae", "mae"), ("rmse", "rmse"), ("bias", "bias"), ("nmad", "nmad_of_glacier_mean_errors")]:
                            assert np.isclose(row[m], e[col], rtol=2e-8, atol=1e-10), (scope, method, kind, m)
                    points.append(row)
                points.append(dict(scope=scope, method=method, excluded=excluded, kind="pixel", n_glaciers=len(t),
                                   mae=t.mae.mean(), rmse=np.sqrt(t.mean_mse.mean()), bias=t.bias.mean()))
    points = pd.DataFrame(points)
    checks["original_glacier_and_volume_points_match_production"] = True

    # Exact weighted pixel NMAD (and RMSE/bias cross-check) from the keyed residual table.
    raw = pd.read_parquet(paths["reference_residuals"], columns=["method", "rgiid", "region", "realization_index", "residual", "evaluation_weight"])
    raw["method"] = raw.method.map(norm_method)
    for excluded in [False, True]:
        r_ = raw.loc[~raw.rgiid.isin(X)] if excluded else raw
        for scope in ["whim", "ehim", "all"]:
            for method in METHODS:
                s = r_.loc[r_.method.eq(method)]
                if scope != "all":
                    s = s.loc[s.region.eq(scope)]
                v = pixel_summary(s.residual.to_numpy(), s.evaluation_weight.to_numpy())
                sel_ = points.scope.eq(scope) & points.method.eq(method) & points.kind.eq("pixel") & points.excluded.eq(excluded)
                points.loc[sel_, "nmad"] = v["nmad"]
                points.loc[sel_, "n_targets"] = len(s)
                assert np.isclose(points.loc[sel_, "rmse"].iloc[0], v["rmse"], rtol=1e-12)
                assert np.isclose(points.loc[sel_, "bias"].iloc[0], v["bias"], rtol=1e-9, atol=1e-15)
    counts = raw.loc[raw.method.eq("constant") & ~raw.rgiid.isin(X)].groupby("region").agg(
        targets=("residual", "size"), glaciers=("rgiid", "nunique"))
    checks["evaluated_targets"] = counts.targets.to_dict()
    checks["evaluated_glaciers"] = counts.glaciers.to_dict()
    exp = sel.expected
    assert checks["evaluated_targets"] == {k: v for k, v in exp["targets_per_method"].items() if k != "all"}
    assert checks["evaluated_glaciers"] == {k: v for k, v in exp["test_glaciers"].items() if k != "all"}
    del raw
    points.to_csv(out / "point_metrics.csv", index=False)

    def main_draws(scope, size, family, excluded):
        s = con.loc[con.region.eq(scope)].copy()
        if excluded:
            s = s.loc[~s.rgiid.isin(X)]
        ids = pd.Index(sorted(s.rgiid))
        s = s.set_index("rgiid").loc[ids]
        b = assign.loc[assign.block_size_km.eq(size)].set_index("rgiid").loc[ids]
        blocks = np.sort(b.block_id.unique())
        codes = pd.Index(blocks).get_indexer(b.block_id)
        mult = mults(len(blocks), np.random.SeedSequence([20260917, size, family, 0 if scope == "ehim" else 1]))
        cnt = np.bincount(codes, minlength=len(blocks))
        den = mult @ cnt
        samples = {}
        for method in METHODS:
            stem = "hybrid" if method == "xgb" else method
            if family == 1:
                t = pg.loc[pg.method.eq(method)].set_index("rgiid").loc[ids]
                maes, means = t.mae.to_numpy(), t.bias.to_numpy()
            else:
                maes, means = s["mae_spec_" + stem].to_numpy(), s["mean_spec_" + stem].to_numpy()
            samples[method] = {"mae": (mult @ np.bincount(codes, weights=maes, minlength=len(blocks))) / den,
                               "abs_bias": abs((mult @ np.bincount(codes, weights=means, minlength=len(blocks))) / den)}
            if family == 2:
                samples[method]["nmad"] = np.array([nmad(np.repeat(means, row[codes])) for row in mult])
        return samples, len(blocks)

    print("Recomputing paired main intervals at 50, 100 and 200 km", flush=True)
    cis = []
    for size in [50, 100, 200]:
        for scope in ["ehim", "whim"]:
            for family, kind in [(1, "pixel"), (2, "specific")]:
                for excluded in [False, True]:
                    samples, k = main_draws(scope, size, family, excluded)
                    for base, comp in [("constant", "hypsometric"), ("constant", "xgb"), ("hypsometric", "xgb")]:
                        for metric in samples[base]:
                            lo, hi = interval(samples[comp][metric] - samples[base][metric])
                            cis.append(dict(size_km=size, scope=scope, kind=kind, excluded=excluded, baseline=base, comparison=comp,
                                            metric=metric, ci_low=lo, ci_high=hi, n_blocks=k))
    cis = pd.DataFrame(cis)
    cis.to_csv(out / "main_paired_intervals.csv", index=False)
    ref = pd.read_csv(tree / "spatial_block_bootstrap/paired_method_differences_ci.csv")
    ref["reference_method"], ref["comparison_method"] = ref.reference_method.map(norm_method), ref.comparison_method.map(norm_method)
    names = {("pixel", "mae"): "pixel_mae_m_per_yr", ("pixel", "abs_bias"): "pixel_bias_m_per_yr", ("specific", "mae"): "glacier_specific_mae_m_per_yr",
             ("specific", "abs_bias"): "glacier_specific_bias_m_per_yr", ("specific", "nmad"): "glacier_specific_nmad_m_per_yr"}
    delta = 0.0
    for row in cis.loc[~cis.excluded].itertuples():
        e = ref.loc[ref.block_size_km.eq(row.size_km) & ref.region.eq(row.scope) & ref.metric.eq(names[(row.kind, row.metric)])
                    & ref.reference_method.eq(row.baseline) & ref.comparison_method.eq(row.comparison)].iloc[0]
        delta = max(delta, abs(row.ci_low - e.ci_low), abs(row.ci_high - e.ci_high))
    assert delta < 1e-10, delta
    checks["original_main_intervals_max_abs_diff"] = delta
    print("Original main intervals reproduced, maximum difference", delta, flush=True)

    def block_interval(v, size=100, seed=20261018):
        v = v.dropna()
        b = assign.loc[assign.block_size_km.eq(size)].set_index("rgiid").block_id.reindex(v.index)
        assert b.notna().all()
        codes, uniq = pd.factorize(b)
        k = len(uniq)
        mult = mults(k, seed)
        den = mult @ np.bincount(codes, minlength=k)
        lo, hi = interval((mult @ np.bincount(codes, weights=v.to_numpy(), minlength=k)) / den)
        return dict(estimate=v.mean(), ci_low=lo, ci_high=hi, n_glaciers=len(v), n_blocks=k)

    print("Recalculating elevation-band diagnostics", flush=True)
    band = pd.read_csv(rev / "band_diagnostics_full/band_observed_void.csv")
    err = pd.read_csv(rev / "band_diagnostics_full/band_method_errors.csv")
    refband = pd.read_csv(rev / "band_diagnostics_full/band_reference_decomposition.csv")
    band, err, refband = (d.loc[~d.rgiid.isin(X)] for d in (band, err, refband))

    def balanced(df, col, weight=None):
        t = df.dropna(subset=[col]).copy()
        keys = ["region", "rgiid", "realization"]
        if weight:
            t["_sum"] = t[col] * t[weight]
            gr = t.groupby(keys)[["_sum", weight]].sum()
            gr = gr["_sum"] / gr[weight]
        else:
            gr = t.groupby(keys)[col].mean()
        return gr.groupby(level=["region", "rgiid"]).mean()

    bands = []
    for scope, s in refband.groupby("region"):
        bands.append(dict(scope=scope, metric="within_fraction", value=s.loc[~s.zero_variance].within_fraction.mean()))
    band = band.assign(contrast_mean=band.mean_obs - band.mean_void)
    band["abs_contrast_mean"] = band.contrast_mean.abs()
    band["contrast_sd"] = np.sqrt(band.var_obs) - np.sqrt(band.var_void)
    for col in ["contrast_mean", "abs_contrast_mean", "contrast_sd"]:
        values = balanced(band.loc[band.eligible], col, "n_void")
        for scope in ["whim", "ehim"]:
            v = values.loc[scope]
            bands.append(dict(scope=scope, metric=col, value=v.mean(), **block_interval(v)))
    t = band.assign(eligible_void=np.where(band.eligible, band.n_void, 0), empty_void=np.where(band.observed_band_empty, band.n_void, 0)
                    ).groupby(["region", "rgiid", "realization"])[["n_void", "eligible_void", "empty_void"]].sum()
    for col in ["eligible_void", "empty_void"]:
        v = (t[col] / t.n_void).groupby(level=["region", "rgiid"]).mean()
        for scope in ["whim", "ehim"]:
            bands.append(dict(scope=scope, metric=col + "_share", value=v.loc[scope].mean()))
    for stratum, s in err.groupby("stratum"):
        for col in ["mae_constant", "mae_hypsometric", "mae_hybrid", "bias_hypsometric", "bias_hybrid"]:
            v = balanced(s, col)
            for scope in ["whim", "ehim"]:
                bands.append(dict(scope=scope, metric=stratum + "_" + col, value=v.loc[scope].mean()))
    pd.DataFrame(bands).to_csv(out / "band_summaries.csv", index=False)

    print("Recalculating realization sensitivity", flush=True)
    S = pd.read_csv(rev / "realization_subsets_per_glacier.csv")
    S = S.loc[~S.rgiid.isin(X)].copy()
    d5 = S.loc[S.k.eq(5)].set_index("rgiid").d_mae_hybrid_minus_hyp
    S["sign_flip"] = np.sign(S.d_mae_hybrid_minus_hyp) != np.sign(S.rgiid.map(d5))
    S.loc[S.k.eq(1)].groupby(["region", "rgiid"]).sign_flip.mean().groupby("region").mean().to_csv(out / "single_mask_rank_flips.csv")
    E = per.loc[~per.rgiid.isin(X)].pivot(index=["region", "rgiid", "realization_index"], columns="method",
                                          values="glacier_wide_specific_error_m_per_yr").rename(columns={"xgb": "hybrid"}).reset_index()
    A = {m: E.pivot(index="rgiid", columns="realization_index", values=m).sort_index() for m in ["constant", "hypsometric", "hybrid"]}
    region = E.groupby("rgiid").region.first().reindex(A["hybrid"].index)
    rng = np.random.default_rng(20261018)
    env = []
    for scope in ["whim", "ehim"]:
        ids = region.loc[region.eq(scope)].index
        Xm = {m: A[m].loc[ids].abs().to_numpy() for m in A}
        for k in range(1, 5):
            combos = list(itertools.combinations(range(5), k))
            masks = np.zeros((len(combos), 5), bool)
            for i, c in enumerate(combos):
                masks[i, list(c)] = True
            values = []
            for _ in range(1000):
                pick = masks[rng.integers(0, len(combos), len(ids))]
                values.append([(Xm[m] * pick).sum(1).mean() / k for m in ["constant", "hypsometric", "hybrid"]])
            values = np.asarray(values)
            for metric, v in [("constant_minus_hyp", values[:, 0] - values[:, 1]), ("hyp_minus_hybrid", values[:, 1] - values[:, 2])]:
                lo, hi = interval(v)
                env.append(dict(scope=scope, k=k, contrast=metric, low=lo, high=hi))
    pd.DataFrame(env).to_csv(out / "realization_envelopes.csv", index=False)

    # Geometry: the original tertile groups (fixed before the exclusion) are retained.
    geo = pd.read_csv(rev / "geometry_per_glacier_realization.csv")
    geo = geo.loc[~geo.rgiid.isin(X)].merge(E.drop(columns="region"), on=["rgiid", "realization_index"], validate="one_to_one")
    rows = []
    for var in ["achieved_frac", "u_raster"]:
        for (scope, label), s in geo.groupby(["region", var + "_bin"], observed=True):
            v = s.groupby("rgiid")[["constant", "hypsometric", "hybrid"]].agg(lambda x: np.mean(abs(x))).mean()
            rows.append(dict(scope=scope, variable=var, bin=label, n_glaciers=s.rgiid.nunique(), **v.to_dict()))
    pd.DataFrame(rows).to_csv(out / "geometry_fixed_groups.csv", index=False)
    write_json(out / "core_checks.json", checks)
    finalize_evaluation_outputs(out)
    print("CORE RECALCULATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()

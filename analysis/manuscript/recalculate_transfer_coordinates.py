#!/usr/bin/env python3
'Recalculate regional-transfer and coordinate-sensitivity results for the manuscript evaluation. Coordinate intervals use sequential Western then Eastern draws at seed 20261019, with a region-stratified combined interval.'
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import finalize_evaluation_outputs, Paths, Selection, base_parser, interval, write_json

METRICS = ["glacier_mae", "glacier_rmse", "glacier_bias", "pixel_mae", "pixel_rmse", "pixel_bias",
           "integrated_mae", "integrated_rmse", "integrated_bias"]


def summarize(P, w=None):
    if w is None:
        w = np.ones((1, len(P)))
    w = np.atleast_2d(w)

    def mean(col):
        return (w @ P[col].to_numpy()) / w.sum(1)
    ident = lambda x: x  # noqa: E731
    return {m: f(mean(c)) for m, c, f in [
        ("glacier_mae", "glacier_abs", ident), ("glacier_rmse", "glacier_sq", np.sqrt), ("glacier_bias", "glacier_mean", ident),
        ("pixel_mae", "pixel_abs", ident), ("pixel_rmse", "pixel_sq", np.sqrt), ("pixel_bias", "pixel_mean", ident),
        ("integrated_mae", "integrated_abs", ident), ("integrated_rmse", "integrated_sq", np.sqrt), ("integrated_bias", "integrated_mean", ident)]}


def main() -> None:
    a = base_parser(__doc__).parse_args()
    paths, sel = Paths(a.data_root, a.repo), Selection(a.repo)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    X = sel.excluded
    con = pd.read_csv(paths["sensitivity_metrics"] / "per_glacier_error_contributions.csv").set_index("rgiid")
    assign = pd.read_csv(paths["reference_results"] / "spatial_block_bootstrap/block_assignments.csv")
    blk = assign.loc[assign.block_size_km.eq(100)].set_index("rgiid").block_id
    checks = {}

    def block_interval(values, seed=20261018):
        v = values.dropna()
        b = blk.reindex(v.index)
        assert b.notna().all()
        codes, uniq = pd.factorize(b)
        k = len(uniq)
        rng = np.random.default_rng(seed)
        mult = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(2000)])
        dist = (mult @ np.bincount(codes, weights=v, minlength=k)) / (mult @ np.bincount(codes, minlength=k))
        lo, hi = interval(dist)
        return dict(estimate=v.mean(), ci_low=lo, ci_high=hi)

    print("Loading keyed reference observations", flush=True)
    f = pd.read_parquet(paths["reference_predictions"], columns=["method", "rgiid", "realization_index", "flat_index", "obs"])
    f = f.loc[f.method.eq("constant")]
    obs = f.set_index(["rgiid", "realization_index", "flat_index"]).obs
    assert obs.index.is_unique
    del f

    def per_glacier_prediction(path):
        print("Summarizing", path.name, flush=True)
        z = np.load(path, allow_pickle=False)
        rgiid, real, pred = z["rgiid"].astype(str), z["realization"], z["prediction"]
        keys = pd.MultiIndex.from_arrays([rgiid, real, z["pixel_flat_index"]])
        assert keys.is_unique
        ix = obs.index.get_indexer(keys)
        assert (ix >= 0).all(), "prediction keys outside the common target set"
        e = pred.astype(float) - obs.to_numpy()[ix]
        den = np.rint(con.valid_area_km2.reindex(rgiid).to_numpy() * 1e6 / 900)
        assert np.isfinite(den).all()
        d = pd.DataFrame(dict(rgiid=rgiid, real=real, e=e, a=abs(e), q=e * e, den=den))
        gr = d.groupby(["rgiid", "real"]).agg(s=("e", "sum"), den=("den", "first"), pm=("e", "mean"), pa=("a", "mean"), pq=("q", "mean"))
        gr["E"] = gr.s / gr.den
        gr["V"] = gr.s * 900
        fields = {"glacier_abs": abs(gr.E), "glacier_sq": gr.E ** 2, "glacier_mean": gr.E, "pixel_abs": gr.pa, "pixel_sq": gr.pq,
                  "pixel_mean": gr.pm, "integrated_abs": abs(gr.V), "integrated_sq": gr.V ** 2, "integrated_mean": gr.V}
        res = pd.DataFrame({k: v.groupby(level=0).mean() for k, v in fields.items()})
        res["region"] = con.region.reindex(res.index)
        res["n_realizations"] = gr.groupby(level=0).size()
        res["n_targets"] = d.groupby("rgiid").size()
        assert res.n_realizations.eq(5).all()
        return res

    ref = pd.read_csv(paths["sensitivity_metrics"] / "transfer_metrics.csv")
    transfer, trans_ci, PGt = [], [], {}
    for target in ["whim", "ehim"]:
        for source in ["whim", "ehim"]:
            pg = per_glacier_prediction(Path(str(paths["transfer_prediction_pattern"]).format(source=source, target=target)))
            PGt[(source, target)] = pg
            full = summarize(pg)
            e = ref.loc[ref.target_region.eq(target) & ref.method.eq("transfer_from_" + source)].iloc[0]
            for m in METRICS:
                assert np.isclose(full[m][0], e[m], rtol=2e-10, atol=1e-9), (target, source, m)
            t = pg.drop(index=X, errors="ignore")
            transfer.append(dict(target=target, source=source, n_glaciers=len(t), n_targets=int(t.n_targets.sum()),
                                 **{k: float(v[0]) for k, v in summarize(t).items()}))
        same = PGt[(target, target)].drop(index=X, errors="ignore")
        cross = PGt[("ehim" if target == "whim" else "whim", target)].drop(index=X, errors="ignore")
        trans_ci.append(dict(target=target, contrast="cross_minus_same_glacier_mae", **block_interval(cross.glacier_abs - same.glacier_abs)))
    checks["original_transfer_metrics_reproduced"] = True
    pd.DataFrame(transfer).to_csv(out / "transfer_metrics.csv", index=False)
    pd.DataFrame(trans_ci).to_csv(out / "transfer_intervals.csv", index=False)

    print("Recomputing coordinate metrics and intervals", flush=True)
    cdir = paths["coordinate_predictions"]
    PG = {arm: per_glacier_prediction(cdir / f"coords_{arm}_predictions_test.npz") for arm in ["off", "on_absolute"]}
    assert PG["on_absolute"].index.equals(PG["off"].index)
    cm = paths["coordinate_metrics"]
    refscore = pd.read_csv(cm / "coordinate_scores.csv")
    refci = pd.read_csv(cm / "coordinate_paired_intervals.csv")
    intervals, metrics, delta = [], [], 0.0
    for excluded in [False, True]:
        ids = PG["off"].index
        if excluded:
            ids = ids[~ids.isin(X)]
        region = PG["off"].region.reindex(ids)
        gb = blk.reindex(ids)
        rng = np.random.default_rng(20261019)
        W = np.zeros((2000, len(ids)))
        for scope in ["whim", "ehim"]:  # one generator, Western draws first, then Eastern
            mask = region.eq(scope).to_numpy()
            codes, uniq = pd.factorize(gb[mask])
            k = len(uniq)
            mult = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(2000)])
            W[:, mask] = mult[:, codes]
        for scope in ["whim", "ehim", "combined"]:
            mask = np.ones(len(ids), bool) if scope == "combined" else region.eq(scope).to_numpy()
            i = ids[mask]
            pt = {arm: summarize(PG[arm].loc[i]) for arm in PG}
            bs = {arm: summarize(PG[arm].loc[i], W[:, mask]) for arm in PG}
            for arm in PG:
                metrics.append(dict(excluded=excluded, scope=scope, arm=arm, n_glaciers=len(i),
                                    n_targets=int(PG[arm].loc[i].n_targets.sum()), **{k: float(v[0]) for k, v in pt[arm].items()}))
                if not excluded:
                    e = refscore.loc[refscore.scope.eq(scope) & refscore.method.eq("coords_" + arm)].iloc[0]
                    for m in METRICS:
                        col = m + "_m3" if m.startswith("integrated_") else m
                        assert np.isclose(pt[arm][m][0], e[col], rtol=2e-10, atol=1e-9), (scope, arm, m)
            for m in METRICS:
                if m.endswith("bias"):
                    est = abs(pt["off"][m][0]) - abs(pt["on_absolute"][m][0])
                    dist = abs(bs["off"][m]) - abs(bs["on_absolute"][m])
                    contrast = "|bias coords_off| - |bias coords_on_absolute|"
                else:
                    est = pt["off"][m][0] - pt["on_absolute"][m][0]
                    dist = bs["off"][m] - bs["on_absolute"][m]
                    contrast = "coords_off - coords_on_absolute"
                lo, hi = interval(dist)
                intervals.append(dict(excluded=excluded, scope=scope, metric=m, contrast=contrast, estimate=est, ci_low=lo, ci_high=hi))
                if not excluded:
                    e = refci.loc[refci.scope.eq(scope) & refci.metric.eq(m) & refci.contrast.eq(contrast)].iloc[0]
                    delta = max(delta, abs(lo - e.ci_low), abs(hi - e.ci_high))
    assert delta < 1e-6, delta
    checks["original_coordinate_intervals_max_abs_diff"] = delta
    print("Original coordinate intervals reproduced, maximum endpoint difference", delta, flush=True)
    pd.DataFrame(metrics).to_csv(out / "coordinate_all_metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "coordinate_all_intervals.csv", index=False)
    pgc = pd.concat({arm: PG[arm] for arm in PG}, names=["arm", "rgiid"]).reset_index()
    pgc["evaluated"] = ~pgc.rgiid.isin(X)  # all assigned test glaciers kept for the original-population checks
    pgc.to_csv(out / "coordinate_per_glacier_moments.csv", index=False)
    write_json(out / "transfer_coordinate_checks.json", checks)
    finalize_evaluation_outputs(out)
    print("TRANSFER AND COORDINATE RECOMPUTATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()

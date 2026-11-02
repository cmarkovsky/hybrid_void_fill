#!/usr/bin/env python3
'Draw Figures S4-S8 from the manuscript evaluation and archived experiment summaries.'
from __future__ import annotations

import itertools
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import Paths, Selection, base_parser, norm_method, write_json  # noqa: E402

SEED = 20261018
METHODS = {"constant": "constant", "hypsometric": "hypsometric", "xgb": "hybrid"}
COLORS = {"whim": "#3b6fb6", "ehim": "#c8553d", "combined": "#555555"}
LABEL = {"whim": "Western", "ehim": "Eastern", "combined": "Combined"}


def block_interval(values, block_of, draws=2000, seed=SEED):
    v = values.dropna()
    b = block_of.reindex(v.index)
    if b.isna().any():
        raise KeyError("glaciers without block assignment")
    codes, uniq = pd.factorize(b)
    k = len(uniq)
    sums = np.bincount(codes, weights=v.to_numpy(float), minlength=k)
    cnts = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    mult = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(draws)])
    est = (mult @ sums) / (mult @ cnts)
    return dict(ci_low=float(np.percentile(est, 2.5)), ci_high=float(np.percentile(est, 97.5)), n_blocks=k)


def save(fig, out, name):
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(out / f"{name}.{e}", dpi=300)
    plt.close(fig)


def balanced(df, value, weight=None):
    t = df.dropna(subset=[value])
    keys = ["region", "rgiid", "realization"]
    if weight is None:
        g = t.groupby(keys)[value].mean()
    else:
        g = t.assign(_w=t[value] * t[weight]).groupby(keys)[["_w", weight]].sum()
        g = g["_w"] / g[weight]
    return g.groupby(level=["region", "rgiid"]).mean()


def band_figure(rev, blk, X, out, checks):
    d = rev / "band_diagnostics_full"
    ref0, band0, err0 = (pd.read_csv(d / n) for n in ["band_reference_decomposition.csv", "band_observed_void.csv", "band_method_errors.csv"])
    summary = pd.read_csv(d / "regional_band_summary.csv")

    def strata(err):
        err = err.assign(d_mae_hyb_minus_hyp=err.mae_hybrid - err.mae_hypsometric)
        rows = []
        for (reg, stratum), t in err.groupby(["region", "stratum"]):
            if stratum not in ("occupied_band", "empty_band"):
                continue
            g_ = t.groupby(["rgiid", "realization"]).d_mae_hyb_minus_hyp.mean().groupby(level=0).mean()
            rows.append(dict(region=reg, stratum=stratum, mean=g_.mean(), n_glaciers=len(g_), **block_interval(g_, blk)))
        return pd.DataFrame(rows)

    s0 = strata(err0)
    want = summary[(summary.section == "2C") & (summary.quantity == "d_mae_hyb_minus_hyp") & summary.stratum.isin(["occupied_band", "empty_band"])]
    m = s0.merge(want, on=["region", "stratum"], suffixes=("", "_run"))
    checks["band_original_reproduced_max_abs_diff"] = float(np.max(np.abs(m[["mean", "ci_low", "ci_high"]].to_numpy() - m[["mean_run", "ci_low_run", "ci_high_run"]].to_numpy())))
    assert len(m) == 4 and checks["band_original_reproduced_max_abs_diff"] < 1e-12
    ref, band, err = (x[~x.rgiid.isin(X)] for x in (ref0, band0, err0))
    band = band.assign(contrast_mean=band.mean_obs - band.mean_void)
    el = band[band.eligible]
    S = strata(err)
    S.to_csv(out / "band_diagnostics_strata.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for reg, col in [("whim", "#3b6fb6"), ("ehim", "#c8553d")]:
        r_ = ref[(ref.region == reg) & ~ref.zero_variance]
        axs[0].hist(r_.within_fraction, bins=np.linspace(0, 1, 26), histtype="step", lw=1.6, color=col,
                    label=f"{'Western' if reg == 'whim' else 'Eastern'} (n={len(r_)})")
        g_ = balanced(el[el.region == reg], "contrast_mean", "n_void").loc[reg]
        axs[1].hist(g_.clip(-1, 1), bins=np.linspace(-1, 1, 41), histtype="step", lw=1.6, color=col)
    axs[0].set_xlabel("within-band share of reference variance"), axs[0].set_ylabel("glaciers"), axs[0].legend(fontsize=7, frameon=False)
    axs[1].set_xlabel("observed − withheld band mean (m a⁻¹), clipped ±1"), axs[1].axvline(0, color="#555", lw=0.8)
    yl = []
    for k, rr in enumerate(S.itertuples()):
        col = "#3b6fb6" if rr.region == "whim" else "#c8553d"
        axs[2].errorbar(rr.mean, k, xerr=[[rr.mean - rr.ci_low], [rr.ci_high - rr.mean]], fmt="o", color=col, ms=5, capsize=3)
        yl.append(f"{'W' if rr.region == 'whim' else 'E'} {rr.stratum.replace('_', ' ')}\n(n={rr.n_glaciers} glaciers)")
    axs[2].set_yticks(range(len(yl)), yl, fontsize=7), axs[2].axvline(0, color="#555", lw=0.8)
    axs[2].set_xlabel("hybrid − hypsometric void-target MAE (m a⁻¹)\n100-km block intervals (conditional)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "band_diagnostics")


def realization_figure(rev, tree, X, out, checks):
    per = pd.read_csv(tree / "glacier_wide_metrics/per_glacier_realization_area_normalized_metrics.csv",
                      usecols=["rgiid", "region", "realization_index", "method", "glacier_wide_specific_error_m_per_yr"])
    per["method"] = per.method.map(norm_method).map(METHODS)
    geo0 = pd.read_csv(rev / "geometry_per_glacier_realization.csv")

    def compute(excl):
        p = per[~per.rgiid.isin(excl)]
        E = p.pivot_table(index=["region", "rgiid", "realization_index"], columns="method", values="glacier_wide_specific_error_m_per_yr").reset_index()
        A = {m: E.pivot(index="rgiid", columns="realization_index", values=m).sort_index() for m in METHODS.values()}
        region = E.groupby("rgiid").region.first().reindex(A["hybrid"].index)
        full = {m: A[m].abs().mean(axis=1) for m in A}
        rows = []
        for s in [s for k in range(1, 6) for s in itertools.combinations(range(5), k)]:
            mae = {m: A[m][list(s)].abs().mean(axis=1) for m in A}
            rows.append(pd.DataFrame({"rgiid": A["hybrid"].index, "region": region.values, "k": len(s),
                                      **{f"absdiff_from5_{m}": (mae[m] - full[m]).abs().values for m in A}}))
        S = pd.concat(rows, ignore_index=True)
        per_g = S.groupby(["region", "k", "rgiid"])[[f"absdiff_from5_{m}" for m in A]].mean()
        reg = per_g.groupby(level=["region", "k"]).mean().rename(columns=lambda c: c.replace("absdiff_from5_", "mean_absdiff_") + "_mean").reset_index()
        rng = np.random.default_rng(SEED)
        env = []
        for reg_name in ["whim", "ehim"]:
            ids = region[region == reg_name].index
            Xa = {m: A[m].loc[ids].abs().to_numpy() for m in A}
            for k in range(1, 5):
                combos = list(itertools.combinations(range(5), k))
                masks = np.zeros((len(combos), 5), bool)
                for i, c in enumerate(combos):
                    masks[i, list(c)] = True
                d = []
                for _ in range(1000):
                    pick = masks[rng.integers(0, len(combos), len(ids))]
                    g = {m: (Xa[m] * pick).sum(1) / k for m in A}
                    d.append((g["hybrid"] - g["hypsometric"]).mean())
                env.append(dict(region=reg_name, k=k, k5_value=(full["hybrid"] - full["hypsometric"]).loc[ids].mean(),
                                envelope_lo=float(np.percentile(d, 2.5)), envelope_hi=float(np.percentile(d, 97.5))))
        geo = geo0[~geo0.rgiid.isin(excl)].merge(E.drop(columns="region"), on=["rgiid", "realization_index"], validate="one_to_one")
        # The stored tertile labels are the original (pre-exclusion) groups; ordered as pd.cut made them.
        geo["achieved_frac_bin"] = pd.Categorical(geo.achieved_frac_bin, ["low", "mid", "high"], ordered=True)
        G = []
        for (reg_name, b), t in geo.groupby(["region", "achieved_frac_bin"], observed=True):
            pg = t.groupby("rgiid")[list(METHODS.values())].agg(lambda x: float(np.mean(np.abs(x))))
            G.append(dict(variable="achieved_frac", bin=b, region=reg_name, n_glaciers=t.rgiid.nunique(),
                          d_mae_hybrid_minus_hyp=(pg["hybrid"] - pg["hypsometric"]).mean()))
        return reg, pd.DataFrame(env), pd.DataFrame(G)  # groupby(region, bin) order, as drawn originally

    reg0, env0, G0 = compute([])
    run_env = pd.read_csv(rev / "realization_subset_envelope.csv").query("quantity == 'd_mae_hybrid_minus_hyp'")
    run_reg = pd.read_csv(rev / "realization_sensitivity.csv")
    run_geo = pd.read_csv(rev / "geometry_sensitivity.csv").query("variable == 'achieved_frac'")
    e = env0.merge(run_env, on=["region", "k"], suffixes=("", "_run"))
    r = reg0.merge(run_reg, on=["region", "k"], suffixes=("", "_run"))
    g = G0.astype({"bin": str}).merge(run_geo, on=["region", "bin"], suffixes=("", "_run"))
    diffs = [np.abs(e[c] - e[c + "_run"]).max() for c in ["k5_value", "envelope_lo", "envelope_hi"]]
    diffs += [np.abs(r[c] - r[c + "_run"]).max() for c in reg0.columns if c.startswith("mean_absdiff")]
    diffs += [np.abs(g.d_mae_hybrid_minus_hyp - g.d_mae_hybrid_minus_hyp_run).max()]
    checks["realization_original_reproduced_max_abs_diff"] = float(max(diffs))
    assert len(e) == 8 and len(g) == 6 and max(diffs) < 1e-12
    out_reg, env, G = compute(X)
    out_reg.to_csv(out / "realization_sensitivity_summary.csv", index=False)
    env.to_csv(out / "realization_envelope.csv", index=False)
    G.to_csv(out / "geometry_fixed_tertiles.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6))
    colors = {"constant": "#8c8c8c", "hypsometric": "#1b9e77", "hybrid": "#7570b3"}
    for reg_name, ls in [("whim", "-"), ("ehim", "--")]:
        o = out_reg[out_reg.region == reg_name]
        for m in METHODS.values():
            axs[0].plot(o.k, o[f"mean_absdiff_{m}_mean"], ls, marker="o", ms=4, lw=1.6, color=colors[m], label=f"{m} ({'W' if reg_name == 'whim' else 'E'})")
        e_ = env[env.region == reg_name]
        k5 = e_.k5_value.iloc[0]
        axs[1].fill_between(e_.k, e_.envelope_lo, e_.envelope_hi, color="#7570b3", alpha=0.18 if ls == "-" else 0.08)
        axs[1].plot([*e_.k, 5], [*(e_.envelope_lo + e_.envelope_hi) / 2, k5], ls, color="#7570b3", lw=1.6, label=f"{'Western' if reg_name == 'whim' else 'Eastern'}")
    axs[0].set_xlabel("realizations in subset (k)"), axs[0].set_ylabel("mean |MAE_S − MAE_5| per glacier (m a⁻¹)")
    axs[0].legend(fontsize=6, frameon=False, ncol=2)
    axs[1].axhline(0, color="#555", lw=0.8), axs[1].set_xlabel("realizations per glacier (k)")
    axs[1].set_ylabel("hybrid − hypsometric regional glacier MAE"), axs[1].legend(fontsize=7, frameon=False)
    axs[1].set_title("random-subset envelope (sensitivity)", fontsize=8)
    lab = []
    for i, rr in enumerate(G.itertuples()):
        axs[2].plot(rr.d_mae_hybrid_minus_hyp, i, "o", color="#3b6fb6" if rr.region == "whim" else "#c8553d", ms=6)
        lab.append(f"{'W' if rr.region == 'whim' else 'E'} void fraction {rr.bin} (n={rr.n_glaciers})")
    axs[2].set_yticks(range(len(lab)), lab, fontsize=7), axs[2].axvline(0, color="#555", lw=0.8)
    axs[2].set_xlabel("hybrid − hypsometric glacier MAE\n(by void-fraction tertile)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "realization_geometry_sensitivity")


def transfer_map(revroot, tree, sel, out, checks):
    import json
    geo = pd.read_csv(tree / "spatial_block_bootstrap/glacier_geometry.csv").set_index("rgiid")
    src = pd.read_csv(revroot / "manifests/transfer_source_glaciers.csv")
    proto = json.loads((revroot / "protocol/transfer_protocol.json").read_text())
    inv = pd.read_csv(sel.repo / "data/metadata/evaluation_membership.csv").set_index("rgiid")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for t, col in [("whim", "#3b6fb6"), ("ehim", "#c8553d")]:
        tg = geo.loc[inv.index[(inv.region == t) & inv.evaluated]]
        ax.scatter(tg.centroid_x / 1e3, tg.centroid_y / 1e3, s=4, marker="o", facecolors="none", edgecolors=col, lw=0.5,
                   label=f"evaluated test {'W' if t == 'whim' else 'E'} (n={len(tg)})")
        s_ = src[src.source_region == t]
        ax.scatter(s_.cx / 1e3, s_.cy / 1e3, s=3, marker="x", color=col, alpha=0.6, lw=0.6, label=f"source {'W' if t == 'whim' else 'E'} (n={len(s_)})")
    ex = geo.loc[sel.excluded]
    ax.scatter(ex.centroid_x / 1e3, ex.centroid_y / 1e3, s=40, marker="D", facecolors="none", edgecolors="k", lw=0.9,
               label="excluded from evaluation; in separation rule (n=%d)" % len(ex))
    d = proto["evaluation"]["distance_checks"]
    ax.set_title(f"Regional-transfer source and test glaciers (Albers km); min cross-region separation {min(d['whim->ehim']['min_m'], d['ehim->whim']['min_m']) / 1e3:.0f} km", fontsize=8)
    ax.set_xlabel("x (km)"), ax.set_ylabel("y (km)"), ax.set_aspect("equal")
    leg = ax.legend(fontsize=7, frameon=False, markerscale=3)
    leg.legend_handles[-1].set_sizes([40])  # the excluded-glacier marker is already large
    ax.spines[["top", "right"]].set_visible(False)
    checks["transfer_map_counts"] = {"sources": src.source_region.value_counts().to_dict(),
                                     "evaluated_test": inv[inv.evaluated].region.value_counts().to_dict(),
                                     "assigned_test_in_separation_rule": proto["evaluation"]["test_glaciers"]}
    save(fig, out, "transfer_separation_map")


def coordinate_figures(cm, recalc, X, out, checks):
    m = pd.read_csv(recalc / "coordinate_per_glacier_moments.csv")
    pg = m.pivot(index="rgiid", columns="arm", values="glacier_abs")
    pg.columns = ["glacier_mae_off", "glacier_mae_on"]
    pg["region"] = m.drop_duplicates("rgiid").set_index("rgiid").region.reindex(pg.index)
    run = pd.read_csv(cm / "coordinate_per_glacier.csv").set_index("rgiid")
    checks["coordinate_per_glacier_original_max_abs_diff"] = float(np.abs(pg[["glacier_mae_off", "glacier_mae_on"]].to_numpy()
                                                                          - run.loc[pg.index, ["glacier_mae_off", "glacier_mae_on"]].to_numpy()).max())
    assert checks["coordinate_per_glacier_original_max_abs_diff"] < 1e-12
    pg = pg[~pg.index.isin(X)]
    pg["glacier_mae_off_minus_on"] = pg.glacier_mae_off - pg.glacier_mae_on
    pg.to_csv(out / "coordinate_per_glacier_subset.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    lim = float(np.r_[pg.glacier_mae_off, pg.glacier_mae_on].max()) * 1.02
    for t in ["whim", "ehim"]:
        d = pg[pg.region == t]
        ax[0].scatter(d.glacier_mae_off, d.glacier_mae_on, s=5, alpha=0.5, color=COLORS[t], lw=0, label=f"{LABEL[t]} (n={len(d)})")
    ax[0].plot([0, lim], [0, lim], color="#888", lw=0.8)
    ax[0].set_xlim(0, lim), ax[0].set_ylim(0, lim), ax[0].set_aspect("equal")
    ax[0].set_xlabel("coords_off glacier MAE (m a$^{-1}$)"), ax[0].set_ylabel("coords_on_absolute glacier MAE (m a$^{-1}$)")
    ax[0].legend(frameon=False, fontsize=8, markerscale=2.5)
    q = float(np.abs(pg.glacier_mae_off_minus_on).max())
    bins = np.linspace(-q, q, 61)
    for t in ["whim", "ehim"]:
        d = pg[pg.region == t].glacier_mae_off_minus_on
        ax[1].hist(d, bins=bins, histtype="step", lw=1.4, color=COLORS[t], density=True, label=f"{LABEL[t]}: mean {d.mean():+.4f}")
    ax[1].axvline(0, color="#333", lw=0.8)
    ax[1].set_xlabel("coords_off − coords_on_absolute, per-glacier MAE (m a$^{-1}$)\n< 0: excluding coordinates is better"), ax[1].set_ylabel("density")
    ax[1].legend(frameon=False, fontsize=8)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    checks["coordinate_per_glacier_display"] = {"axis_limit": lim, "histogram_half_range": q, "points_outside_axes": 0}
    save(fig, out, "coordinate_per_glacier_comparison")

    I = pd.read_csv(recalc / "coordinate_all_intervals.csv").query("excluded")
    panels = [("m a$^{-1}$", ["glacier_mae", "glacier_rmse", "glacier_bias", "pixel_mae", "pixel_rmse", "pixel_bias"], 1.0),
              ("10$^3$ m$^3$ a$^{-1}$", ["integrated_mae", "integrated_rmse", "integrated_bias"], 1e-3)]
    nice = {"glacier_mae": "glacier MAE", "glacier_rmse": "glacier RMSE", "glacier_bias": "|glacier bias|", "pixel_mae": "pixel MAE",
            "pixel_rmse": "pixel RMSE", "pixel_bias": "|pixel bias|", "integrated_mae": "integrated MAE", "integrated_rmse": "integrated RMSE",
            "integrated_bias": "|integrated bias|"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2, 1.2]})
    for ax, (unit, mets, sc) in zip(axes, panels):
        ylab, ypos = [], 0
        for k in mets:
            for scope in ["whim", "ehim", "combined"]:
                r = I[(I.metric == k) & (I.scope == scope)].iloc[0]
                ax.errorbar(r.estimate * sc, -ypos, xerr=[[(r.estimate - r.ci_low) * sc], [(r.ci_high - r.estimate) * sc]], fmt="o", ms=4,
                            color=COLORS[scope], capsize=2, lw=1.2)
                ylab.append(f"{nice[k]} · {LABEL[scope][0]}")
                ypos += 1
            ypos += 0.6
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks([-(i + 0.6 * (i // 3)) for i in range(len(ylab))], ylab, fontsize=7)
        ax.set_xlabel(f"coords_off − coords_on_absolute ({unit})\n< 0: excluding coordinates is better", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Paired 100-km block intervals (2,000 draws; conditional on this fitted pair)", fontsize=8, loc="left")
    save(fig, out, "coordinate_regional_intervals")


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--recalc", type=__import__("pathlib").Path, required=True)
    ap.add_argument("--subset-tree", type=__import__("pathlib").Path, required=True)
    a = ap.parse_args()
    paths, sel = Paths(a.data_root, a.repo), Selection(a.repo)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    tree, rev = paths["reference_results"], paths["sensitivity_metrics"]
    blk = pd.read_csv(tree / "spatial_block_bootstrap/block_assignments.csv").query("block_size_km == 100").set_index("rgiid").block_id
    checks = {}
    subprocess.run([sys.executable, str(a.repo / "analysis/manuscript/figures/figS01_relative_improvement.py"),
                    str(a.subset_tree / "spatial_block_bootstrap/paired_method_differences_ci.csv"),
                    str(out / "fig_04_relative_improvement.png")], check=True)
    band_figure(rev, blk, sel.excluded, out, checks)
    realization_figure(rev, tree, sel.excluded, out, checks)
    transfer_map(rev.parent, tree, sel, out, checks)
    coordinate_figures(paths["coordinate_metrics"], a.recalc, sel.excluded, out, checks)
    write_json(out / "SUPPLEMENT_FIGURE_CHECKS.json", checks)
    print("supplement figures complete", checks)


if __name__ == "__main__":
    main()

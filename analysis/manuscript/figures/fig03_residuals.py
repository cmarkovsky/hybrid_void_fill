
# Changes: --input required; the glacier-ID check uses the repository evaluation-membership file;
# format_sig3 uses Decimal rounding so trailing zeros are kept (0.004697 -> 0.00470).
# 02_plot_kde_grid_with_rightside_exceedance_common_xy_tc.py
#
# 2x4 figure using balanced common-support residuals:
#   rows = region
#   cols 1-3 = KDE by method
#   col 4 = exceedance probability with all methods overlaid
#
# Exceedance panel shows:
#   P(|residual| >= x)
#
# All methods are compared on the exact same pixels. Each glacier has equal
# total weight, divided equally among its realizations and pixels.

from __future__ import annotations


import argparse
from pathlib import Path
import colorsys
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
import seaborn as sns


# =========================
# USER SETTINGS
# =========================
COMMON_XY_CACHE = None  # --input is required
EVALUATED_SPLIT = Path(__file__).resolve().parents[3] / 'data/metadata/splits/evaluation_split.json'

CLIP_PERCENTILES = (0.5, 99.5)

KDE_GRID_SIZE = 500
KDE_BW = 0.22

TAIL_THRESHOLD = 1.0
TAIL_ALPHA = 0.26
RIDGE_ALPHA = 0.76
LINE_ALPHA = 0.98
KDE_HEIGHT_SCALE = 1.0

METHOD_ORDER = ["constant", "hypsometric", "xgb"]

METHOD_BASE = {
    "constant": "#5C6F91",
    "hypsometric": "#B5651D",
    "xgb": "#3A7D44",
}

METHOD_NAMES = {
    "constant": "Constant local mean",
    "hypsometric": "Hypsometric local mean",
    "xgb": "Hybrid XGBoost",
}

REGIONS_ORDER = ["whim", "ehim"]

REG_NAMES = {
    "whim": r"Western\ Himalaya", # type: ignore
    "ehim": r"Eastern\ Himalaya", # type: ignore
}

SAVE_FIG = True
SAVE_BASE = Path('figures/updated') / 'fig_03'

TICK_LOCS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
ABS_TICK_LOCS = [0.0, 0.5, 1.0, 1.5]

FIGSIZE = (12, 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot balanced common-support residual distributions for the "
            "coordinate-free XGBoost model and paired baseline methods."
        )
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Exact-common-support residual table of the evaluated population (stage 02 output).")
    parser.add_argument("--output-base", type=Path, default=SAVE_BASE)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


# =========================
# STYLE
# =========================
sns.set_theme(
    style="whitegrid",
    context="paper",
    font="DejaVu Sans",
    rc={
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": "#d0d0d0",
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    },
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "legend.title_fontsize": 8,
    "lines.linewidth": 1.0,
    "axes.titlepad": 4,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
})


# =========================
# Helpers
# =========================
def adjust_lightness(color, factor):
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)


def method_shades(base_hex):
    return {
        "density": base_hex,
        "tail": adjust_lightness(base_hex, 0.90),
        "range95": adjust_lightness(base_hex, 1.55),
        "IQR": adjust_lightness(base_hex, 0.72),
        "median": adjust_lightness(base_hex, 0.52),
    }


METHOD_COLORS = {m: method_shades(c) for m, c in METHOD_BASE.items()}


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return np.nan
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(quantile) * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    return float(values[index])


def density_kde(x, weights, xlim, grid_size=500, bw_method="scott"):
    lo, hi = xlim
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(x)
        & np.isfinite(weights)
        & (weights > 0)
        & (x >= lo)
        & (x <= hi)
    )
    x = x[valid]
    weights = weights[valid]

    grid = np.linspace(lo, hi, grid_size)

    if x.size < 2 or np.allclose(np.std(x), 0.0):
        d = np.zeros_like(grid)
        if x.size >= 1:
            idx = np.argmin(np.abs(grid - np.median(x)))
            d[idx] = 1.0
        return grid, d

    kde = gaussian_kde(x, bw_method=bw_method, weights=weights)
    d = kde(grid)
    return grid, d


def exceedance_xy(x, weights, xmax=None):
    """
    Exceedance curve for absolute residuals:
        P(|x| >= t)
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    x = x[valid]
    weights = weights[valid]
    x = np.abs(x)

    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    sorted_weights = weights[order]
    ys = np.cumsum(sorted_weights[::-1])[::-1] / np.sum(sorted_weights)
    if xmax is not None:
        keep = xs <= xmax
        xs = xs[keep]
        ys = ys[keep]
    return xs, ys


def compute_global_density_scale(
    residuals: dict[tuple[str, str], np.ndarray],
    weights: dict[tuple[str, str], np.ndarray],
    regions: list[str],
    methods: list[str],
    xlim: tuple[float, float],
    grid_size: int,
    bw_method,
) -> float:
    max_density = 0.0
    for region in regions:
        for method in methods:
            r = residuals.get((region, method), np.array([], dtype=np.float64))
            w = weights.get((region, method), np.array([], dtype=np.float64))
            if r.size == 0:
                continue
            _, d = density_kde(
                r,
                w,
                xlim=xlim,
                grid_size=grid_size,
                bw_method=bw_method,
            )
            local_max = float(np.nanmax(d)) if d.size else 0.0
            max_density = max(max_density, local_max)
    return max_density if max_density > 0 else 1.0


def draw_quantile_glyph(ax, r, weights, y, method=None):
    med = weighted_quantile(r, weights, 0.5)
    q25 = weighted_quantile(r, weights, 0.25)
    q75 = weighted_quantile(r, weights, 0.75)
    q025 = weighted_quantile(r, weights, 0.025)
    q975 = weighted_quantile(r, weights, 0.975)

    ax.plot(
        [q025, q975], [y, y],
        lw=1.5, alpha=0.62, solid_capstyle="round",
        color=METHOD_COLORS[method]["range95"], zorder=8, # type: ignore
    )

    cap_height = 0.015
    ax.plot([q025, q025], [y - cap_height, y + cap_height],
            lw=1.5, alpha=0.55, color=METHOD_COLORS[method]["range95"], zorder=8) # type: ignore
    ax.plot([q975, q975], [y - cap_height, y + cap_height],
            lw=1.5, alpha=0.55, color=METHOD_COLORS[method]["range95"], zorder=8) # type: ignore

    ax.plot(
        [q25, q75], [y, y],
        lw=6.5, solid_capstyle="round",
        color=METHOD_COLORS[method]["IQR"], zorder=9, # type: ignore
    )

    ax.scatter(
        [med], [y],
        s=36, facecolor=METHOD_COLORS[method]["median"], # type: ignore
        edgecolor="white", linewidth=0.9, zorder=10,
    )


def draw_tail_highlights(ax, x, y0, d, threshold, color):
    left = x <= -threshold
    right = x >= threshold

    if np.any(left):
        ax.fill_between(x[left], y0, y0 + d[left], color=color, alpha=TAIL_ALPHA, linewidth=0, zorder=4)
    if np.any(right):
        ax.fill_between(x[right], y0, y0 + d[right], color=color, alpha=TAIL_ALPHA, linewidth=0, zorder=4)


def summary_stats(r, weights):
    r = np.asarray(r, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(r) & np.isfinite(weights) & (weights > 0)
    r = r[valid]
    weights = weights[valid]
    if r.size == 0:
        return {"bias": np.nan, "nmad": np.nan, "rmse": np.nan, "n": 0}

    med = weighted_quantile(r, weights, 0.5)
    nm = 1.4826 * weighted_quantile(np.abs(r - med), weights, 0.5)
    bi = np.average(r, weights=weights)
    rs = np.sqrt(np.average(r ** 2, weights=weights))
    return {"bias": bi, "nmad": nm, "rmse": rs, "n": int(r.size)}


def format_sig3(value, signed=False):
    # Decimal rounding as in build_tables.fmt. np.format_float_positional dropped the trailing
    # zero when rounding carried (0.004697 -> "0.0047" instead of "0.00470").
    from decimal import Decimal, ROUND_HALF_EVEN
    d=Decimal(str(float(value)))
    if d==0:
        return "+0.000" if signed else "0.000"
    q=d.quantize(Decimal(1).scaleb(d.copy_abs().adjusted()-2),rounding=ROUND_HALF_EVEN)
    text=f"{q:.{max(0,2-q.copy_abs().adjusted())}f}"
    return "+"+text if signed and value>=0 else text


def load_common_xy_residuals_by_region_method(
    cache_path: Path,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[str, int],
]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Common XY cache not found: {cache_path}")

    if cache_path.suffix.lower() == ".csv":
        df = pd.read_csv(cache_path)
    elif cache_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(cache_path)
    else:
        raise ValueError("--input must be a CSV or parquet file")

    required = {"method", "rgiid", "region", "realization"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cache missing required columns: {sorted(missing)}")

    if "residual" not in df.columns:
        if not {"obs", "pred"}.issubset(df.columns):
            raise ValueError(
                "Input must contain either 'residual' or both 'obs' and 'pred'."
            )
        df["residual"] = (
            pd.to_numeric(df["pred"], errors="coerce")
            - pd.to_numeric(df["obs"], errors="coerce")
        )

    df = df.copy()
    df["rgiid"] = df["rgiid"].astype(str)
    split_path=EVALUATED_SPLIT
    if split_path.is_file():
        split=json.loads(split_path.read_text())
        expected_ids={str(rid) for rid,group in split.items() if group=="test"}
        if set(df["rgiid"].unique())!=expected_ids:
            raise ValueError("Residual glacier IDs do not match the evaluated test population")
    df["method"] = df["method"].astype(str).str.lower().str.strip()
    df["region"] = df["region"].astype(str).str.lower().str.strip().replace(
        {
            "western himalaya": "whim",
            "western_himalaya": "whim",
            "eastern himalaya": "ehim",
            "eastern_himalaya": "ehim",
        }
    )
    df["residual"] = pd.to_numeric(df["residual"], errors="coerce")
    if not np.isfinite(df["residual"]).all():
        raise ValueError(
            "Common-support input contains non-finite residuals. Regenerate it "
            "with 02_pixel_summary_stats.py."
        )

    observed_methods = set(df["method"].unique())
    if observed_methods != set(METHOD_ORDER):
        raise ValueError(
            f"Expected methods {METHOD_ORDER}, found {sorted(observed_methods)}"
        )

    if "pixel_flat_index" in df.columns and "flat_index" not in df.columns:
        df = df.rename(columns={"pixel_flat_index": "flat_index"})
    if "flat_index" in df.columns:
        pixel_columns = ["flat_index"]
    elif {"x", "y"}.issubset(df.columns):
        pixel_columns = ["x", "y"]
    else:
        raise ValueError("Exact support validation requires flat_index or x and y")

    realization_column = (
        "realization_index" if "realization_index" in df.columns else "realization"
    )
    physical_key = ["rgiid", realization_column, *pixel_columns]
    method_key = ["method", *physical_key]
    if df.duplicated(method_key).any():
        raise ValueError("Duplicate method/glacier/realization/pixel rows found")
    support = df.groupby(physical_key, observed=True)["method"].nunique()
    if (support != len(METHOD_ORDER)).any():
        raise ValueError("Methods are not evaluated on identical physical pixels")

    if "evaluation_weight" in df.columns:
        df["evaluation_weight"] = pd.to_numeric(
            df["evaluation_weight"], errors="coerce"
        )
        if (
            not np.isfinite(df["evaluation_weight"]).all()
            or (df["evaluation_weight"] <= 0).any()
        ):
            raise ValueError("evaluation_weight must contain finite positive values")
    else:
        method_glacier = ["method", "rgiid"]
        n_realizations = df.groupby(method_glacier, observed=True)[
            realization_column
        ].transform("nunique")
        stratum = ["method", "rgiid", realization_column]
        n_pixels = df.groupby(stratum, observed=True)["residual"].transform("size")
        df["evaluation_weight"] = 1.0 / (n_realizations * n_pixels)

    totals = df.groupby(["method", "rgiid"], observed=True)[
        "evaluation_weight"
    ].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            "Evaluation weights do not sum to one per method/glacier. "
            "Regenerate the weighted residual table."
        )

    residuals: dict[tuple[str, str], np.ndarray] = {}
    weights: dict[tuple[str, str], np.ndarray] = {}
    n_glaciers_by_region = {}
    for region in REGIONS_ORDER:
        region_frame = df[df["region"] == region]
        n_glaciers_by_region[region] = int(region_frame["rgiid"].nunique())
        for method in METHOD_ORDER:
            sub = region_frame[region_frame["method"] == method]
            arr = sub["residual"].to_numpy(dtype=np.float64)
            residuals[(region, method)] = arr
            weights[(region, method)] = sub["evaluation_weight"].to_numpy(
                dtype=np.float64
            )

    return residuals, weights, n_glaciers_by_region


def verify_equal_region_counts(residuals: dict[tuple[str, str], np.ndarray]) -> dict[str, int]:
    out = {}
    for region in REGIONS_ORDER:
        counts = {method: len(residuals.get((region, method), np.array([]))) for method in METHOD_ORDER}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Pixel mismatch in region {region}: {counts}")
        out[region] = next(iter(counts.values())) if counts else 0
        print(f"{region}: consistent pixel count = {counts}")
    return out


# =========================
# Main
# =========================
def main():
    args = parse_args()

    residuals, weights, n_glaciers_by_region = (
        load_common_xy_residuals_by_region_method(cache_path=args.input)
    )

    print("Test glaciers by region:")
    for region in REGIONS_ORDER:
        if region in n_glaciers_by_region:
            print(f"  {region}: {n_glaciers_by_region[region]:,}")

    n_pixels_by_region = verify_equal_region_counts(residuals)

    pooled = []
    pooled_weights = []
    for region in REGIONS_ORDER:
        for method in METHOD_ORDER:
            arr = residuals.get((region, method), np.array([], dtype=np.float64))
            weight = weights.get((region, method), np.array([], dtype=np.float64))
            if arr.size:
                pooled.append(arr)
                pooled_weights.append(weight)

    if len(pooled) == 0:
        raise RuntimeError("No residuals available to plot.")

    pooled = np.concatenate(pooled)
    pooled_weights_array = np.concatenate(pooled_weights)
    lo = weighted_quantile(pooled, pooled_weights_array, CLIP_PERCENTILES[0] / 100)
    hi = weighted_quantile(pooled, pooled_weights_array, CLIP_PERCENTILES[1] / 100)
    max_abs = float(max(abs(lo), abs(hi)))
    xlim = (-max_abs, max_abs)
    abs_xlim = (0.0, max_abs)

    global_density_max = compute_global_density_scale(
        residuals=residuals,
        weights=weights,
        regions=REGIONS_ORDER,
        methods=METHOD_ORDER,
        xlim=xlim,
        grid_size=KDE_GRID_SIZE,
        bw_method=KDE_BW,
    )

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 1.0],
        # hspace=0.15,
    )

    subfigs = [fig.add_subfigure(gs[i]) for i in range(2)]
    
    axes = np.empty((2, 4), dtype=object)
    for i, subfig in enumerate(subfigs):
        gs_sub = subfig.add_gridspec(
            nrows=1,
            ncols=4,
            width_ratios=[1.0, 1.0, 1.0, 1.0],
        )
        subfig.suptitle(r"$\mathbf{" + REG_NAMES[REGIONS_ORDER[i]] + "}$\n", fontsize=10, fontweight="bold")
        for j in range(4):
            sharex = axes[0, 0] if (i > 0 and j < 3) else None
            axes[i, j] = subfig.add_subplot(gs_sub[0, j], sharex=sharex)
    

    pos_left = axes[0, 0].get_position()
    pos_right = axes[0, 3].get_position()

    x_center = 0.5 * (pos_left.x0 + pos_right.x1)
    y_whim = axes[0, 0].get_position().y1 + 0.1


    #     x_center,
    #     y_whim,
    #     REG_NAMES[REGIONS_ORDER[0]],




    # )

    # y_ehim = axes[1, 0].get_position().y1 + 0.1


    #     x_center,
    #     y_ehim,
    #     REG_NAMES[REGIONS_ORDER[1]],




    # )
    
    # KDE columns
    for i, region in enumerate(REGIONS_ORDER):
        for j, method in enumerate(METHOD_ORDER):
            ax = axes[i, j]

            ax.axvline(0.0, ls="--", lw=0.9, alpha=0.85, color="k", zorder=10)
            ax.axvline(-TAIL_THRESHOLD, ls=":", lw=0.8, alpha=0.55, color="0.35", zorder=1)
            ax.axvline(+TAIL_THRESHOLD, ls=":", lw=0.8, alpha=0.55, color="0.35", zorder=1)

            r = residuals.get((region, method), np.array([], dtype=np.float64))
            w = weights.get((region, method), np.array([], dtype=np.float64))
            if r.size == 0:
                ax.set_visible(False)
                continue

            x, d = density_kde(
                r,
                w,
                xlim=xlim,
                grid_size=KDE_GRID_SIZE,
                bw_method=KDE_BW,
            )
            d = d / (global_density_max + 1e-12)
            d *= KDE_HEIGHT_SCALE

            color = METHOD_COLORS[method]["density"]
            tail_color = METHOD_COLORS[method]["tail"]

            ax.fill_between(x, 0, d, color=color, alpha=RIDGE_ALPHA, linewidth=0.6, zorder=2)
            ax.plot(x, d, color=color, linewidth=1.0, alpha=LINE_ALPHA, zorder=3)

            draw_tail_highlights(ax=ax, x=x, y0=0, d=d, threshold=TAIL_THRESHOLD, color=tail_color)

            ymax = float(np.nanmax(d)) if d.size else 0.0
            glyph_y = ymax * 1.08 if ymax > 0 else 0.08
            draw_quantile_glyph(ax, r, w, y=glyph_y, method=method)

            stats = summary_stats(r, w)
            ax.text(
                0.97, 0.97,
                f"NMAD = {format_sig3(stats['nmad'])}\nBias = {format_sig3(stats['bias'],signed=True)}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=7.2,
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    fc="white", ec="0.85", alpha=0.88,
                ),
                zorder=20,
            )


            ax.set_title(METHOD_NAMES[method], pad=5)

            if j == 0:
                n_pix = n_pixels_by_region.get(region, 0)
                ax.set_ylabel(
                    f"Balanced density\n($N = {format(n_pix, ',').replace(',', '{,}')}$ target instances)",
                    fontsize=8,
                    labelpad=5,
                )
            else:
                ax.set_ylabel("")

            ax.set_xlim(*xlim)
            ax.set_ylim(0, max(1.10, 1.08 * KDE_HEIGHT_SCALE * 1.12))
            ax.grid(axis="x", alpha=0.20)
            ax.grid(axis="y", alpha=0.10)
            ax.tick_params(axis="y", labelleft=False)
            ax.xaxis.set_major_locator(mticker.FixedLocator(TICK_LOCS))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Exceedance column
    for i, region in enumerate(REGIONS_ORDER):
        ax = axes[i, 3]

        ax.axvline(TAIL_THRESHOLD, ls=":", lw=0.8, alpha=0.55, color="0.35", zorder=1)

        for method in METHOD_ORDER:
            r = residuals.get((region, method), np.array([], dtype=np.float64))
            w = weights.get((region, method), np.array([], dtype=np.float64))

            xs, ys = exceedance_xy(r, w, xmax=abs_xlim[1])
            if xs.size == 0:
                continue

            ax.plot(
                xs, ys,
                color=METHOD_COLORS[method]["density"],
                lw=1.4,
                alpha=0.98,
                label=METHOD_NAMES[method],
            )

        
        ax.set_title("Tail exceedance", pad=5)

        ax.set_xlim(*abs_xlim)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", alpha=0.20)
        ax.grid(axis="y", alpha=0.16)

        ax.xaxis.set_major_locator(mticker.FixedLocator(ABS_TICK_LOCS))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.tick_params(axis="x", which="major", labelsize=7.5, length=3.5, width=0.7)

        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        if i == 0:
            ax.set_ylabel(r"$P(|\mathrm{residual}| \geq x)$", fontsize=8, rotation=-90, labelpad=12)
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

        else:
            ax.set_ylabel(r"$P(|\mathrm{residual}| \geq x)$", fontsize=8, rotation=-90, labelpad=12)
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis="y", labelleft=False, labelright=True, length=3.5, width=0.7)
        ax.yaxis.set_label_position("right")
    # bottom-row x labels
    for j in range(3):
        axes[1, j].set_xlabel(r"$\mathrm{d}h/\mathrm{d}t$ residual (m a$^{-1}$)", fontsize=8)
    axes[1, 3].set_xlabel(r"$|\mathrm{d}h/\mathrm{d}t \ \mathrm{residual}|$ (m a$^{-1}$)", fontsize=8)

    # suppress top-row x tick labels
    for j in range(4):
        axes[0, j].tick_params(axis="x", labelbottom=False)

    # panel labels
    panel_labels = list("abcdefgh")
    for label, ax in zip(panel_labels, axes.flatten()):
        ax.text(
            0.02, 0.98, f"({label})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8,
            fontweight="bold",
        )

    rows=[]
    for region in REGIONS_ORDER:
        for method in METHOD_ORDER:
            r=residuals[(region,method)]
            w=weights[(region,method)]
            info=summary_stats(r,w)
            info.update(region=region,method=method,mae=float(np.average(np.abs(r),weights=w)),
                n_glaciers=n_glaciers_by_region[region])
            rows.append(info)
    args.output_base.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(str(args.output_base)+"_metrics.csv",index=False)
    audit={"input":str(args.input),"input_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "n_glaciers":n_glaciers_by_region,"target_instances_per_method":n_pixels_by_region,
        "methods_share_identical_targets":True,"weight_per_method_glacier":1.0,
        "density_xlim":list(xlim),"display_percentiles":list(CLIP_PERCENTILES),
        "density_ylim":[0,max(1.10,1.08*KDE_HEIGHT_SCALE*1.12)],
        "whisker_percentiles":[2.5,97.5],"annotation_significant_figures":3,
        "metrics":rows}
    Path(str(args.output_base)+"_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    # plt.tight_layout()

    if SAVE_FIG:
        args.output_base.parent.mkdir(parents=True, exist_ok=True)
        pdf_path = args.output_base.with_suffix(".pdf")
        png_path = args.output_base.with_suffix(".png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
        print(f"Saved: {png_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

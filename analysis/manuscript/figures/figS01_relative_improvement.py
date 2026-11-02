#!/usr/bin/env python3
"""Figure S1: relative improvement (%) with spatial-block CIs.

Reads the production paired-difference table (relative improvement and its percentile
interval are computed within each bootstrap draw). Positive = the comparison method
has lower error than the reference. One marker per block size (50/100/200 km), so
the block-size sensitivity is visible directly. |Bias| is omitted: ratios of biases
near zero are not meaningful.

  python figS01_relative_improvement.py <paired_method_differences_ci.csv> <output.png>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("paired_table", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
paired = pd.read_csv(args.paired_table)
import matplotlib.transforms as mtransforms

METRICS = [("pixel_mae_m_per_yr", "MAE", "Pixel scale"), ("pixel_nmad_m_per_yr", "NMAD", "Pixel scale"),
           ("glacier_specific_mae_m_per_yr", "MAE", "Glacier-wide"), ("glacier_specific_nmad_m_per_yr", "NMAD", "Glacier-wide")]
COMPARISONS = [("Constant", "Hypsometric", "Hypsometric vs Constant", "#1B9E77", "D"),
               ("Constant", "XGBoost", "Hybrid XGBoost vs Constant", "#5E3C99", "o"),
               ("Hypsometric", "XGBoost", "Hybrid XGBoost vs Hypsometric", "#D95F02", "s")]
REGIONS = [("whim", "Western Himalaya"), ("ehim", "Eastern Himalaya")]
SIZES = {50: 0.30, 100: 0.0, 200: -0.30}  # top to bottom: 50, 100, 200 km
# Block size is shown by whisker line style; 100 km (primary) is solid and slightly heavier.
STYLES = {50: dict(ls="--", ms=3.6, lw=1.1, alpha=0.85, label="50 km"),
          100: dict(ls="-", ms=5.5, lw=1.5, alpha=1.0, label="100 km (primary)"),
          200: dict(ls=":", ms=3.6, lw=1.4, alpha=0.85, label="200 km")}
ROW, GROUP_GAP, SECTION_GAP, PAD = 1.15, 0.9, 0.9, 0.6

# Vertical layout: three comparison rows per metric group, a gap between groups and an
# extra gap between the pixel-scale and glacier-wide sections.
centers, tops, bottoms, y = [], [], [], 0.0
for i, (_, _, section) in enumerate(METRICS):
    if i and section != METRICS[i - 1][2]:
        y -= SECTION_GAP
    rows = [y - j * ROW for j in range(len(COMPARISONS))]
    centers.append(rows)
    tops.append(rows[0] + PAD)
    bottoms.append(rows[-1] - PAD)
    y = rows[-1] - ROW - GROUP_GAP

fig, axes = plt.subplots(1, 2, figsize=(11.5, 8.4), sharey=True, sharex=True)  # same x-range: West and East comparable
for k, (ax, (region, rlabel)) in enumerate(zip(axes, REGIONS)):
    for i, (metric, mlabel, section) in enumerate(METRICS):
        ax.axhspan(bottoms[i], tops[i], color="#F1F3F5" if i % 2 == 0 else "#E6EAEE", zorder=0, lw=0)
        for j, (ref, comp, clabel, color, marker) in enumerate(COMPARISONS):
            for km, dy in SIZES.items():
                r = paired[(paired.region == region) & (paired.metric == metric) & (paired.reference_method == ref)
                           & (paired.comparison_method == comp) & (paired.block_size_km == km)]
                if r.empty:
                    continue
                r = r.iloc[0]
                st = STYLES[km]
                # Every block size shows its point estimate; the 100 km (primary) marker is larger.
                primary = km == 100
                bars = ax.errorbar(r.relative_improvement_pct, centers[i][j] + dy,
                                   xerr=[[r.relative_improvement_pct - r.relative_ci_low], [r.relative_ci_high - r.relative_improvement_pct]],
                                   fmt=marker, color=color, ms=6.5 if primary else 4.6, mec="white", mew=0.9 if primary else 0.7,
                                   alpha=st["alpha"], capsize=2.2, lw=st["lw"], zorder=4 if primary else 3)
                bars.lines[2][0].set_linestyle(st["ls"])
    # Darker rule between the pixel-scale and glacier-wide sections.
    for i in range(1, len(METRICS)):
        if METRICS[i][2] != METRICS[i - 1][2]:
            ax.axhline((bottoms[i - 1] + tops[i]) / 2, color="#4A4A4A", lw=1.1, zorder=1)
    ax.axvline(0, color="#555", lw=0.8, ls="--", zorder=2)
    ax.set_title(rlabel)
    ax.set_xlabel("Relative improvement (%; positive = lower error)")
    ax.grid(axis="x", color="#D0D4D8", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.set_yticks([c[1] for c in centers], [m[1] for m in METRICS], fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(bottoms[-1] - 0.2, tops[0] + 0.2)
    if k == 0:  # section labels left of the metric labels
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for section in dict.fromkeys(m[2] for m in METRICS):
            rows = [i for i, m in enumerate(METRICS) if m[2] == section]
            mid = (tops[rows[0]] + bottoms[rows[-1]]) / 2
            ax.text(-0.13, mid, section, transform=trans, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
method_handles = [plt.Line2D([], [], marker=m, color=c, ls="none", ms=6, label=l) for _, _, l, c, m in COMPARISONS]
size_handles = [plt.Line2D([], [], color="#4D4D4D", marker="o", ms=6.5 if km == 100 else 4.6, mec="white", mew=0.9,
                           ls=st["ls"], lw=st["lw"], alpha=st["alpha"], label=st["label"]) for km, st in STYLES.items()]
fig.legend(handles=method_handles, title="Comparison", loc="upper left", bbox_to_anchor=(0.10, 1.03), frameon=False,
           alignment="left", title_fontproperties={"weight": "bold"})
fig.legend(handles=size_handles, title="Spatial block size (95% interval line)", loc="upper left", bbox_to_anchor=(0.58, 1.03), frameon=False,
           handlelength=3.4, alignment="left", title_fontproperties={"weight": "bold"})
fig.text(0.5, -0.02, "Paired spatial-block bootstrap, 2,000 draws; whiskers are 95% percentile intervals of the relative improvement. "
         "Block sizes are stacked in legend order; point estimates are identical across block sizes (only the intervals change).",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0.02, 0, 1, 0.90))
fig.savefig(args.output, dpi=250, bbox_inches="tight")
print("saved", args.output)

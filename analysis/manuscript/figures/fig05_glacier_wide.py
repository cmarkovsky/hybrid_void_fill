#!/usr/bin/env python3
# Copy of workflow/src/void_fill/06_figures/05_fig5_glacier_wide_comparison.py (release 14a87a4).
# Changes: workflow_support import path; panel annotations use three significant figures (sig3)
# instead of three decimals. Plotted data and layout are unchanged.
"""Plot original and reconstructed glacier-wide estimates.

This script reads the glacier-level CSV written by
``04_calculate_glacier_wide_comparison.py``. It performs no statistical
aggregation and writes no analysis tables.
"""

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3] / 'workflow'))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("constant", "hypsometric", "xgb")
METHOD_LABELS = {
    "constant": "Constant",
    "hypsometric": "Hypsometric",
    "xgb": "Hybrid XGBoost",
}
METHOD_COLORS = {
    "constant": "#5C6F91",
    "hypsometric": "#B5651D",
    "xgb": "#3A7D44",
}
REGION_LABELS = {
    "whim": "Western Himalaya",
    "ehim": "Eastern Himalaya",
}



def sig3(value, signed=False):
    """Three significant figures with trailing zeros (manuscript display rule)."""
    from decimal import Decimal, ROUND_HALF_EVEN
    d = Decimal(str(float(value)))
    if d == 0:
        return "+0.000" if signed else "0.000"
    q = d.quantize(Decimal(1).scaleb(d.copy_abs().adjusted() - 2), rounding=ROUND_HALF_EVEN)
    text = f"{q:.{max(0, 2 - q.copy_abs().adjusted())}f}"
    return "+" + text if signed and value >= 0 else text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinate-mode", choices=("include", "exclude"), default="exclude"
    )
    parser.add_argument(
        "--quantity", choices=("specific", "volume"), default="specific"
    )
    parser.add_argument("--region", default="all")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=workflow_path('comparison'),
    )
    parser.add_argument(
        "--glaciers",
        type=Path,
        help="Glacier-level CSV overriding the path derived from other options.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workflow_path('figures') / 'fig_05.png',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=400)
    return parse_stage_args(parser)


def axis_label(quantity: str) -> str:
    if quantity == "specific":
        return r"Glacier-wide specific elevation change (m a$^{-1}$)"
    return r"Volume-change rate ($10^6$ m$^3$ a$^{-1}$)"


def default_glacier_path(
    input_dir: Path,
    quantity: str,
    coordinate_mode: str,
    region: str,
) -> Path:
    suffix = f"{quantity}_{coordinate_mode}_{region.lower()}"
    return input_dir / f"glacier_wide_comparison_{suffix}_glaciers.csv"


def load_glaciers(path: Path, region: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "rgiid",
        "region",
        "method",
        "original",
        "reconstructed",
        "mean_signed_error",
        "mean_absolute_error",
        "n_realizations",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["rgiid"] = frame["rgiid"].astype(str)
    frame["region"] = frame["region"].astype(str).str.lower().str.strip()
    frame["method"] = frame["method"].astype(str).str.lower().str.strip()
    frame = frame[frame["method"].isin(METHODS)].copy()
    if region.lower() != "all":
        frame = frame[frame["region"] == region.lower()].copy()
    for column in (
        "original",
        "reconstructed",
        "mean_signed_error",
        "mean_absolute_error",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        np.isfinite(frame["original"])
        & np.isfinite(frame["reconstructed"])
        & np.isfinite(frame["mean_signed_error"])
        & np.isfinite(frame["mean_absolute_error"])
    ].copy()
    if frame.empty:
        raise RuntimeError(f"No finite glacier results remain after loading {path}")

    method_sets = frame.groupby(["rgiid", "region"], observed=True)["method"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    expected = tuple(sorted(METHODS))
    bad = method_sets[method_sets != expected]
    if not bad.empty:
        raise ValueError(
            "Glacier table is not exactly paired across methods. "
            f"First problems:\n{bad.head(12).to_string()}"
        )
    if frame.duplicated(["rgiid", "region", "method"]).any():
        raise ValueError("Glacier table contains duplicate glacier-method rows")
    return frame.sort_values(["region", "rgiid", "method"])


def draw_raincloud(
    axis: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    rng: np.random.Generator,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    width = 0.72
    if len(values) > 1 and not np.allclose(values, values[0]):
        violin = axis.violinplot(
            [values],
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.minimum(vertices[:, 0], position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)
            body.set_linewidth(0.8)

    jitter = rng.uniform(0.06, width / 2, size=len(values))
    axis.scatter(
        position + jitter,
        values,
        s=7,
        color=color,
        alpha=min(0.32, max(0.07, 80 / len(values))),
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    axis.boxplot(
        [values],
        positions=[position],
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.4},
        boxprops={"facecolor": color, "edgecolor": color},
        whiskerprops={"color": color},
        capprops={"color": color},
        zorder=3,
    )


def focused_limits(
    values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("Cannot calculate plot limits without finite values")
    lower, upper = float(np.min(finite)), float(np.max(finite))
    span = upper - lower
    padding = 0.08 * span if span > 0 else max(1.0, abs(lower) * 0.08)
    return float(lower - padding), float(upper + padding)


def ordered_regions(glaciers: pd.DataFrame) -> list[str]:
    available = set(glaciers["region"].unique())
    regions = [code for code in REGION_LABELS if code in available]
    regions.extend(sorted(available.difference(regions)))
    return regions


def plot_figure(
    glaciers: pd.DataFrame,
    label: str,
    output: Path,
    seed: int,
    dpi: int,
) -> None:
    rng = np.random.default_rng(seed)
    regions = ordered_regions(glaciers)
    if not regions:
        raise ValueError("No regions are available to plot")

    figure, axes = plt.subplots(
        len(regions),
        4,
        figsize=(15.8, 4.4 * len(regions)),
        dpi=dpi,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.94,
        bottom=0.12 if len(regions) == 1 else 0.09,
        top=0.88,
        wspace=0.20,
        hspace=0.28,
    )

    scatter_limits: dict[str, tuple[float, float]] = {}
    for region_code in regions:
        regional = glaciers[glaciers["region"] == region_code]
        scatter_limits[region_code] = focused_limits(
            np.concatenate(
                [
                    regional["original"].to_numpy(dtype=float),
                    regional["reconstructed"].to_numpy(dtype=float),
                ]
            ),
            0.05,
            0.95,
        )

    distribution_limits = focused_limits(
        np.concatenate(
            [
                glaciers["original"].to_numpy(dtype=float),
                glaciers["reconstructed"].to_numpy(dtype=float),
            ]
        ),
        0.025,
        0.99,
    )

    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for row, region_code in enumerate(regions):
        for column, method in enumerate(METHODS):
            axis = axes[row, column]
            subset = glaciers[
                (glaciers["method"] == method)
                & (glaciers["region"] == region_code)
            ]
            axis.scatter(
                subset["original"],
                subset["reconstructed"],
                s=12,
                alpha=0.25,
                color=METHOD_COLORS[method],
                linewidths=0,
                rasterized=True,
            )
            limits = scatter_limits[region_code]
            axis.plot(limits, limits, color="0.2", linestyle="--", linewidth=1.0)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            ticks = axis.yaxis.get_majorticklocs()
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.tick_params(axis="x", labelsize=8, labelrotation=45)
            for tick_label in axis.get_xticklabels():
                tick_label.set_horizontalalignment("right")
            if row == len(regions) - 1:
                axis.set_xlabel(f"Original complete\n{label}")
            if column == 0:
                axis.set_ylabel(f"Reconstructed\n{label}")
            else:
                axis.set_yticklabels([])

            letter = panel_letters[row * 4 + column]
            title = f"({letter})"
            if row == 0:
                title += f"  {METHOD_LABELS[method]}"
            axis.set_title(title, loc="left")

            mae = float(subset["mean_absolute_error"].mean())
            bias = float(subset["mean_signed_error"].mean())
            axis.text(
                0.04,
                0.96,
                f"MAE = {sig3(mae)}\nBias = {sig3(bias, True)}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
            )

        axis = axes[row, 3]
        regional = glaciers[glaciers["region"] == region_code]
        original = (
            regional[["rgiid", "original"]]
            .drop_duplicates("rgiid")["original"]
            .to_numpy(dtype=float)
        )
        draw_raincloud(axis, original, 0, "#242424", rng)
        for position, method in enumerate(METHODS, start=1):
            estimates = regional.loc[
                regional["method"] == method, "reconstructed"
            ].to_numpy(dtype=float)
            draw_raincloud(axis, estimates, position, METHOD_COLORS[method], rng)
        axis.set_ylim(distribution_limits)
        axis.set_xticks(range(4))
        axis.set_xticklabels(
            ["Original\ncomplete", "Constant", "Hypsometric", "Hybrid\nXGBoost"],
            fontsize=8,
            rotation=20,
            ha="right",
            rotation_mode="anchor",
        )
        if row != len(regions) - 1:
            axis.tick_params(axis="x", labelbottom=False)
        axis.set_ylabel(label, rotation=270, labelpad=20)
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()
        letter = panel_letters[row * 4 + 3]
        title = f"({letter})"
        if row == 0:
            title += "  Glacier-wide estimates"
        axis.set_title(title, loc="left")

    for row, region_code in enumerate(regions):
        row_top = axes[row, 0].get_position().y1
        figure.text(
            0.5,
            row_top + 0.035,
            REGION_LABELS.get(region_code, region_code.upper()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(True)
        axis.grid(color="0.9", linewidth=0.6, zorder=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {output}")


def main() -> None:
    args = parse_args()
    glacier_path = args.glaciers
    if glacier_path is None:
        glacier_path = default_glacier_path(
            args.input_dir,
            args.quantity,
            args.coordinate_mode,
            args.region,
        )
    glaciers = load_glaciers(glacier_path, args.region)
    plot_figure(
        glaciers,
        axis_label(args.quantity),
        args.output,
        args.seed,
        args.dpi,
    )


if __name__ == "__main__":
    main()

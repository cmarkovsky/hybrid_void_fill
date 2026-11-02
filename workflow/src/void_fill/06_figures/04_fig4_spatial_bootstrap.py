#!/usr/bin/env python3
'Plot spatial-block-bootstrap confidence intervals for void filling.'

from __future__ import annotations

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


METHOD_ORDER = ("Constant", "Hypsometric", "XGBoost")
METHOD_LABELS = {
    "Constant": "Constant local mean",
    "Hypsometric": "Hypsometric local mean",
    "XGBoost": "Hybrid XGBoost",
}
METHOD_COLORS = {
    "Constant": "#4C78A8",
    "Hypsometric": "#F58518",
    "XGBoost": "#2A9D8F",
}
METHOD_ALIASES = {
    "constant": "Constant",
    "constant local mean": "Constant",
    "hypsometric": "Hypsometric",
    "hypsometric local mean": "Hypsometric",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
    "hybrid xgboost": "XGBoost",
}

COMPARISON_ORDER = (
    ("Constant", "XGBoost"),
    ("Hypsometric", "XGBoost"),
    ("Constant", "Hypsometric"),
)
COMPARISON_COLORS = {
    ("Constant", "Hypsometric"): "#1B9E77",
    ("Constant", "XGBoost"): "#5E3C99",
    ("Hypsometric", "XGBoost"): "#E66101",
}

COMPARISON_MARKERS = {
    ("Constant", "Hypsometric"): "D",
    ("Constant", "XGBoost"): "o",
    ("Hypsometric", "XGBoost"): "s",
}

REGION_ORDER = ("whim", "ehim")
REGION_LABELS = {
    "whim": "Western Himalaya",
    "ehim": "Eastern Himalaya",
}
FAMILY_SPECS = {
    "pixel": {
        "title": "Pixel-scale error",
        "metrics": (
            "pixel_mae_m_per_yr",
            "pixel_nmad_m_per_yr",
            "pixel_bias_m_per_yr",
        ),
        "metric_labels": ("MAE", "NMAD", "Bias"),
        "paired_metric_labels": ("MAE", "NMAD", "|Bias|"),
        "scale": 1.0,
        "x_label": r"Error statistic (m yr$^{-1}$)",
        "difference_x_label": r"Paired difference (m yr$^{-1}$)",
    },
    "volume": {
        "title": "Void-integrated volume error",
        "metrics": (
            "void_volume_mae_m3_per_yr",
            "void_volume_nmad_m3_per_yr",
            "void_volume_bias_m3_per_yr",
        ),
        "metric_labels": ("MAE", "NMAD", "Bias"),
        "paired_metric_labels": ("MAE", "NMAD", "|Bias|"),
        "scale": 1e-6,
        "x_label": r"Error statistic ($10^6$ m$^3$ yr$^{-1}$)",
        "difference_x_label": r"Paired difference ($10^6$ m$^3$ yr$^{-1}$)",
    },
    "specific": {
        "title": "Glacier-wide specific error",
        "metrics": (
            "glacier_specific_mae_m_per_yr",
            "glacier_specific_nmad_m_per_yr",
            "glacier_specific_bias_m_per_yr",
        ),
        "metric_labels": ("MAE", "NMAD", "Bias"),
        "paired_metric_labels": ("MAE", "NMAD", "|Bias|"),
        "scale": 1.0,
        "x_label": r"Error statistic (m yr$^{-1}$)",
        "difference_x_label": r"Paired difference (m yr$^{-1}$)",
    },
}

REQUIRED_COLUMNS = {
    "block_size_km",
    "region",
    "method",
    "metric",
    "estimate",
    "ci_low",
    "ci_high",
    "n_blocks",
}

REQUIRED_PAIRED_COLUMNS = {
    "block_size_km",
    "region",
    "metric",
    "reference_method",
    "comparison_method",
    "difference",
    "ci_low",
    "ci_high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication and sensitivity figures from spatial-block "
            "bootstrap confidence intervals."
        )
    )
    parser.add_argument(
        "--estimates-file",
        type=Path,
        default=workflow_path('bootstrap') / 'performance_estimates_ci.csv',
        help="CSV written by 05_spatial_block_bootstrap.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workflow_path('figures'),
        help=(
            "Directory for figures. By default, figures are written next to "
            "the estimates CSV."
        ),
    )
    parser.add_argument(
        "--paired-file",
        type=Path,
        default=None,
        help=(
            "Paired-difference CSV written by 05_spatial_block_bootstrap.py. "
            "By default, paired_method_differences_ci.csv next to the "
            "estimates CSV is used."
        ),
    )
    parser.add_argument(
        "--primary-block-size-km",
        type=int,
        default=100,
        help="Block size used in the manuscript-ready figure.",
    )
    parser.add_argument(
        "--sensitivity-family",
        choices=tuple(FAMILY_SPECS),
        default="specific",
        help="Metric family shown in the block-size sensitivity figure.",
    )
    parser.add_argument(
        "--plot-void-volume-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include the void-integrated volume-error row in the performance "
            "and paired figures (use --no-plot-void-volume-error to omit it)."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
        help="One or more output formats.",
    )
    parser.add_argument(
        "--xlim-scope",
        choices=("figure", "row"),
        default="figure",
        help=(
            "x-axis range of the paired-difference figure: one range for every panel (default, as published) "
            "or one per metric row, so glacier-wide differences are not compressed by pixel-scale ones."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parse_stage_args(parser)


def configure_style() -> None:
    """Use a restrained style that remains legible in print."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def canonical_method(value: object) -> str:
    text = str(value).strip()
    return METHOD_ALIASES.get(text.lower(), text)


def canonical_region(value: object) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "whim": "whim",
        "western himalaya": "whim",
        "western": "whim",
        "ehim": "ehim",
        "eastern himalaya": "ehim",
        "eastern": "ehim",
    }
    return aliases.get(text, text)


def load_estimates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Bootstrap summary not found: {path}\n"
            "Run 05_spatial_block_bootstrap.py first, or provide the correct "
            "path with --estimates-file."
        )

    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    numeric_columns = (
        "block_size_km",
        "estimate",
        "ci_low",
        "ci_high",
        "n_blocks",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["method"] = frame["method"].map(canonical_method)
    frame["region"] = frame["region"].map(canonical_region)

    invalid = frame[list(numeric_columns)].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            "The estimates CSV contains missing or non-numeric plotting "
            f"values in {int(invalid.sum())} row(s)."
        )
    if (frame["ci_low"] > frame["ci_high"]).any():
        raise ValueError("At least one row has ci_low greater than ci_high.")
    if (frame["n_blocks"] < 2).any():
        raise ValueError("Every region/block-size combination needs >=2 blocks.")

    key = ["block_size_km", "region", "method", "metric"]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate interval rows found for these keys:\n"
            + frame.loc[duplicates, key].head(20).to_string(index=False)
        )

    known_metrics = {
        metric
        for spec in FAMILY_SPECS.values()
        for metric in spec["metrics"]
    }
    frame = frame.loc[frame["metric"].isin(known_metrics)].copy()
    if frame.empty:
        raise ValueError("No recognized performance metrics were found.")

    return frame


def load_paired_differences(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Paired bootstrap summary not found: {path}\n"
            "Run 05_spatial_block_bootstrap.py first, or provide the correct "
            "path with --paired-file."
        )

    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_PAIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    numeric_columns = ("block_size_km", "difference", "ci_low", "ci_high")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["reference_method"] = frame["reference_method"].map(canonical_method)
    frame["comparison_method"] = frame["comparison_method"].map(canonical_method)
    frame["region"] = frame["region"].map(canonical_region)
    invalid = frame[list(numeric_columns)].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            "The paired-difference CSV contains missing or non-numeric "
            f"plotting values in {int(invalid.sum())} row(s)."
        )
    if (frame["ci_low"] > frame["ci_high"]).any():
        raise ValueError(
            "At least one paired-difference row has ci_low greater than ci_high."
        )

    key = [
        "block_size_km",
        "region",
        "metric",
        "reference_method",
        "comparison_method",
    ]
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate paired interval rows found for these keys:\n"
            + frame.loc[duplicates, key].head(20).to_string(index=False)
        )

    known_metrics = {
        metric
        for spec in FAMILY_SPECS.values()
        for metric in spec["metrics"]
    }
    frame = frame.loc[frame["metric"].isin(known_metrics)].copy()
    if frame.empty:
        raise ValueError("No recognized paired performance metrics were found.")
    return frame


def ordered_present(values: pd.Series, preferred: tuple[str, ...]) -> list[str]:
    present = list(pd.unique(values.astype(str)))
    return [value for value in preferred if value in present] + sorted(
        value for value in present if value not in preferred
    )


def method_style(method: str, fallback_index: int) -> tuple[str, str]:
    colors = tuple(METHOD_COLORS.values())
    color = METHOD_COLORS.get(method, colors[fallback_index % len(colors)])
    label = METHOD_LABELS.get(method, method)
    return color, label


def comparison_label(reference: str, comparison: str) -> str:
    return f"{comparison} − {reference}"


def comparison_color(
    reference: str,
    comparison: str,
    fallback_index: int,
) -> str:
    fallback = ("#5E3C99", "#E66101", "#1B9E77", "#7570B3")
    return COMPARISON_COLORS.get(
        (reference, comparison), fallback[fallback_index % len(fallback)]
    )


def add_method_legend(fig: plt.Figure, methods: list[str]) -> None:
    handles = []
    # Positive offsets sit above negative offsets, so list the plotted
    # methods from top to bottom in the same order they appear in each panel.
    for index, method in enumerate(reversed(methods)):
        color, label = method_style(method, len(methods) - index - 1)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label=label,
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(1, len(handles)),
        frameon=False,
    )


def horizontal_interval(
    ax: plt.Axes,
    row: pd.Series,
    *,
    y: float,
    scale: float,
    color: str,
    marker: str = "o",
    zorder: int = 3,
    estimate_column: str = "estimate",
) -> None:
    estimate = float(row[estimate_column]) * scale
    low = float(row["ci_low"]) * scale
    high = float(row["ci_high"]) * scale
    cap_half_height = 0.045
    ax.hlines(y, low, high, color=color, linewidth=1.45, zorder=zorder)
    ax.vlines(
        [low, high],
        y - cap_half_height,
        y + cap_half_height,
        color=color,
        linewidth=1.2,
        zorder=zorder,
    )
    ax.plot(
        estimate,
        y,
        marker=marker,
        markersize=5.5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        linestyle="none",
        zorder=zorder + 1,
    )


def draw_reference_lines(ax: plt.Axes) -> None:
    ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.65)
    # Steps of 1/2/5 keep every tick exactly representable at two decimals (0.025 steps were shown as 0.02/0.07).
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    """Return limits that include every finite value and a small margin."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    low = min(float(np.min(finite)), 0.0)
    high = max(float(np.max(finite)), 0.0)
    span = high - low
    padding = 0.05 * span if span > 0 else max(0.05, abs(low) * 0.05)
    return low - padding, high + padding


def label_subplots(axes: np.ndarray) -> None:
    """Add (a), (b), ... labels in row-major reading order."""
    for index, ax in enumerate(axes.flat):
        ax.text(
            0.02,
            0.98,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="normal",
        )


def plot_primary_intervals(
    frame: pd.DataFrame,
    *,
    primary_block_size_km: int,
    plot_void_volume_error: bool,
) -> plt.Figure:
    primary = frame.loc[frame["block_size_km"] == primary_block_size_km].copy()
    if primary.empty:
        available = sorted(frame["block_size_km"].unique().astype(int).tolist())
        raise ValueError(
            f"No rows found for the primary {primary_block_size_km}-km block "
            f"size. Available sizes: {available}"
        )

    regions = ordered_present(primary["region"], REGION_ORDER)
    methods = ordered_present(primary["method"], METHOD_ORDER)
    if not regions or not methods:
        raise ValueError("No regions or methods are available for plotting.")

    family_specs = {
        family: spec
        for family, spec in FAMILY_SPECS.items()
        if plot_void_volume_error or family != "volume"
    }
    limit_values = [
        primary.loc[primary["metric"].isin(spec["metrics"]),
                    ["estimate", "ci_low", "ci_high"]].to_numpy()
        * float(spec["scale"])
        for spec in family_specs.values()
    ]
    common_xlim = padded_limits(np.concatenate(limit_values, axis=0))

    fig, axes = plt.subplots(
        len(family_specs),
        len(regions),
        figsize=(5.7 * len(regions), 9.2),
        sharex="row",
        squeeze=False,
    )
    offsets = (
        np.array([0.0])
        if len(methods) == 1
        else np.linspace(-0.18, 0.18, len(methods))
    )
    y_base = np.arange(3)[::-1]

    for row_index, (family, spec) in enumerate(family_specs.items()):
        for column_index, region in enumerate(regions):
            ax = axes[row_index, column_index]
            subset = primary.loc[
                (primary["region"] == region)
                & (primary["metric"].isin(spec["metrics"]))
            ]

            for method_index, method in enumerate(methods):
                color, _ = method_style(method, method_index)
                for metric_index, metric in enumerate(spec["metrics"]):
                    selected = subset.loc[
                        (subset["method"] == method)
                        & (subset["metric"] == metric)
                    ]
                    if selected.empty:
                        warnings.warn(
                            f"Missing {region}/{method}/{metric} at "
                            f"{primary_block_size_km} km; leaving it blank."
                        )
                        continue
                    horizontal_interval(
                        ax,
                        selected.iloc[0],
                        y=float(y_base[metric_index] + offsets[method_index]),
                        scale=float(spec["scale"]),
                        color=color,
                    )

            draw_reference_lines(ax)
            ax.set_yticks(y_base, spec["metric_labels"])
            ax.set_ylim(-0.55, 2.55)
            ax.set_xlabel(str(spec["x_label"]))

            if row_index == 0:
                block_counts = subset["n_blocks"].dropna().astype(int).unique()
                block_text = (
                    f"; {block_counts[0]} occupied blocks"
                    if len(block_counts) == 1
                    else ""
                )
                ax.set_title(
                    f"{REGION_LABELS.get(region, region)}{block_text}", pad=8
                )
            if column_index == 0:
                ax.set_ylabel(str(spec["title"]), labelpad=8)
            else:
                ax.tick_params(axis="y", labelleft=False)


    for ax in axes.flat:
        ax.set_xlim(*common_xlim)

    label_subplots(axes)
    fig.suptitle(
        f"Reconstruction performance with {primary_block_size_km}-km "
        "spatial-block bootstrap intervals",
        fontsize=14,
        y=0.985,
    )
    add_method_legend(fig, methods)
    fig.text(
        0.5,
        0.012,
        "Dots are full-test-set estimates; whiskers are percentile 95% confidence intervals. "
        "Intervals quantify spatial evaluation uncertainty, not predictive uncertainty.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.925), h_pad=1.15, w_pad=1.0)
    return fig


def plot_block_size_sensitivity(
    frame: pd.DataFrame,
    *,
    family: str,
    primary_block_size_km: int,
    plot_void_volume_error: bool,
) -> plt.Figure:
    if family == "volume" and not plot_void_volume_error:
        raise ValueError(
            "The volume sensitivity figure is disabled by "
            "--no-plot-void-volume-error; choose --sensitivity-family pixel "
            "or specific."
        )
    spec = FAMILY_SPECS[family]
    subset = frame.loc[frame["metric"].isin(spec["metrics"])].copy()
    regions = ordered_present(subset["region"], REGION_ORDER)
    methods = ordered_present(subset["method"], METHOD_ORDER)
    block_sizes = sorted(subset["block_size_km"].unique().astype(int).tolist())
    if not block_sizes:
        raise ValueError(f"No rows are available for the {family!r} family.")
    common_xlim = padded_limits(
        subset[["estimate", "ci_low", "ci_high"]].to_numpy()
        * float(spec["scale"])
    )

    fig, axes = plt.subplots(
        len(spec["metrics"]),
        len(regions),
        figsize=(5.7 * len(regions), 8.8),
        sharex="row",
        squeeze=False,
    )
    method_offsets = (
        np.array([0.0])
        if len(methods) == 1
        else np.linspace(-0.18, 0.18, len(methods))
    )
    block_positions = np.arange(len(block_sizes), dtype=float)

    for metric_index, (metric, metric_label) in enumerate(
        zip(spec["metrics"], spec["metric_labels"])
    ):
        for region_index, region in enumerate(regions):
            ax = axes[metric_index, region_index]
            panel = subset.loc[
                (subset["metric"] == metric) & (subset["region"] == region)
            ]

            if primary_block_size_km in block_sizes:
                primary_position = block_positions[
                    block_sizes.index(primary_block_size_km)
                ]
                ax.axhspan(
                    primary_position - 0.42,
                    primary_position + 0.42,
                    color="#EEEEEE",
                    zorder=-2,
                )

            for method_index, method in enumerate(methods):
                color, _ = method_style(method, method_index)
                for block_index, block_size in enumerate(block_sizes):
                    selected = panel.loc[
                        (panel["method"] == method)
                        & (panel["block_size_km"] == block_size)
                    ]
                    if selected.empty:
                        warnings.warn(
                            f"Missing {region}/{method}/{metric}/{block_size} km; "
                            "leaving it blank."
                        )
                        continue
                    marker = "D" if block_size == primary_block_size_km else "o"
                    horizontal_interval(
                        ax,
                        selected.iloc[0],
                        y=float(block_positions[block_index] + method_offsets[method_index]),
                        scale=float(spec["scale"]),
                        color=color,
                        marker=marker,
                        zorder=4 if marker == "D" else 3,
                    )

            tick_labels = []
            for block_size in block_sizes:
                counts = panel.loc[
                    panel["block_size_km"] == block_size, "n_blocks"
                ].dropna().astype(int).unique()
                count_text = f" (n={counts[0]})" if len(counts) == 1 else ""
                primary_text = " *" if block_size == primary_block_size_km else ""
                tick_labels.append(f"{block_size} km{count_text}{primary_text}")

            draw_reference_lines(ax)
            ax.set_yticks(block_positions, tick_labels)
            ax.set_ylim(len(block_sizes) - 0.45, -0.55)
            ax.set_xlabel(str(spec["x_label"]))
            if region_index == 0:
                ax.set_ylabel(f"{metric_label}\nblock size", labelpad=8)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if metric_index == 0:
                ax.set_title(REGION_LABELS.get(region, region), pad=8)


    for ax in axes.flat:
        ax.set_xlim(*common_xlim)

    label_subplots(axes)
    fig.suptitle(
        f"Block-size sensitivity of {str(spec['title']).lower()} intervals",
        fontsize=14,
        y=0.985,
    )
    add_method_legend(fig, methods)
    fig.text(
        0.5,
        0.012,
        "Dots are unchanged full-test-set estimates; whiskers are percentile 95% confidence intervals. "
        f"Gray rows and diamonds mark the primary {primary_block_size_km}-km analysis; n is occupied blocks.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.055, 1.0, 0.925), h_pad=1.15, w_pad=1.0)
    return fig


def plot_paired_method_differences(
    paired: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    primary_block_size_km: int,
    plot_void_volume_error: bool,
    xlim_scope: str = "figure",
) -> plt.Figure:
    primary = paired.loc[
        paired["block_size_km"] == primary_block_size_km
    ].copy()
    if primary.empty:
        available = sorted(paired["block_size_km"].unique().astype(int).tolist())
        raise ValueError(
            f"No paired rows found for the primary {primary_block_size_km}-km "
            f"block size. Available sizes: {available}"
        )

    regions = ordered_present(primary["region"], REGION_ORDER)
    present_comparisons = set(
        zip(primary["reference_method"], primary["comparison_method"])
    )
    comparisons = [
        comparison
        for comparison in COMPARISON_ORDER
        if comparison in present_comparisons
    ]
    comparisons.extend(sorted(present_comparisons.difference(comparisons)))
    if not regions or not comparisons:
        raise ValueError("No regions or paired method comparisons are available.")

    family_specs = {
        family: spec
        for family, spec in FAMILY_SPECS.items()
        if plot_void_volume_error or family != "volume"
    }
    limit_values = [
        primary.loc[primary["metric"].isin(spec["metrics"]),
                    ["difference", "ci_low", "ci_high"]].to_numpy()
        * float(spec["scale"])
        for spec in family_specs.values()
    ]
    common_xlim = padded_limits(np.concatenate(limit_values, axis=0))
    row_xlims = [padded_limits(values) if xlim_scope == "row" else common_xlim for values in limit_values]

    fig, axes = plt.subplots(
        len(family_specs),
        len(regions),
        figsize=(5.7 * len(regions), 9.2),
        sharex="row",
        sharey="col",
        squeeze=False,
    )
    offsets = (
        np.array([0.0])
        if len(comparisons) == 1
        else np.linspace(-0.13, 0.13, len(comparisons))
    )
    y_base = np.arange(3)[::-1]

    for row_index, (family, spec) in enumerate(family_specs.items()):
        for column_index, region in enumerate(regions):
            ax = axes[row_index, column_index]
            panel = primary.loc[
                (primary["region"] == region)
                & (primary["metric"].isin(spec["metrics"]))
            ]

            for comparison_index, (reference, comparison) in enumerate(comparisons):
                color = comparison_color(reference, comparison, comparison_index)
                marker = COMPARISON_MARKERS.get((reference, comparison), "o")
                for metric_index, metric in enumerate(spec["metrics"]):
                    selected = panel.loc[
                        (panel["reference_method"] == reference)
                        & (panel["comparison_method"] == comparison)
                        & (panel["metric"] == metric)
                    ]
                    if selected.empty:
                        warnings.warn(
                            f"Missing paired result for {region}/{comparison} "
                            f"minus {reference}/{metric} at "
                            f"{primary_block_size_km} km; leaving it blank."
                        )
                        continue
                    horizontal_interval(
                        ax,
                        selected.iloc[0],
                        y=float(y_base[metric_index] + offsets[comparison_index]),
                        scale=float(spec["scale"]),
                        color=color,
                        marker=marker,
                        estimate_column="difference",
                    )

            draw_reference_lines(ax)
            ax.set_yticks(y_base, spec["paired_metric_labels"])
            ax.set_ylim(-0.55, 2.55)
            ax.set_xlabel(str(spec["difference_x_label"]))

            if row_index == 0:
                estimate_panel = estimates.loc[
                    (estimates["block_size_km"] == primary_block_size_km)
                    & (estimates["region"] == region)
                ]
                block_counts = (
                    estimate_panel["n_blocks"].dropna().astype(int).unique()
                )
                block_text = (
                    f"; {block_counts[0]} occupied blocks"
                    if len(block_counts) == 1
                    else ""
                )
                ax.set_title(f"{REGION_LABELS[region]}{block_text}", pad=8)
            if column_index == 0:
                ax.set_ylabel(str(spec["title"]), labelpad=8)
            else:
                ax.tick_params(axis="y", labelleft=False)

    for ax, xlim in zip(axes.flat, [lim for lim in row_xlims for _ in range(axes.shape[1])]):
        ax.set_xlim(*xlim)
        if xlim[0] < 0:
            ax.axvspan(
                xlim[0],
                min(0.0, xlim[1]),
                color="#EAF4EE",
                alpha=0.75,
                zorder=-3,
            )

    label_subplots(axes)
    handles = []
    # Match the top-to-bottom order of the vertically offset comparisons.
    for legend_index, (reference, comparison) in enumerate(reversed(comparisons)):
        plot_index = len(comparisons) - legend_index - 1
        color = comparison_color(reference, comparison, plot_index)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=COMPARISON_MARKERS.get((reference, comparison), "o"),
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label=comparison_label(reference, comparison),
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(1, len(handles)),
        frameon=True,
    )
    fig.suptitle(
        f"Paired method differences with {primary_block_size_km}-km "
        "spatial-block bootstrap intervals",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Differences are comparison minus reference: negative values favor the "
        "comparison method. Bias is compared as |bias|; whiskers are paired "
        "percentile 95% confidence intervals.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.925), h_pad=1.15, w_pad=1.0)
    return fig


def save_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    formats: tuple[str, ...] | list[str],
    dpi: int,
) -> list[Path]:
    saved = []
    for output_format in formats:
        path = stem.with_suffix(f".{output_format}")
        fig.savefig(
            path,
            dpi=dpi if output_format == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    if args.primary_block_size_km <= 0:
        raise ValueError("--primary-block-size-km must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    output_dir = args.output_dir or args.estimates_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    estimates = load_estimates(args.estimates_file)
    paired_path = args.paired_file or args.estimates_file.with_name(
        "paired_method_differences_ci.csv"
    )
    paired = load_paired_differences(paired_path)

    primary = plot_primary_intervals(
        estimates,
        primary_block_size_km=args.primary_block_size_km,
        plot_void_volume_error=args.plot_void_volume_error,
    )
    saved = save_figure(
        primary,
        output_dir / "spatial_bootstrap_performance_intervals",
        formats=args.formats,
        dpi=args.dpi,
    )

    sensitivity = plot_block_size_sensitivity(
        estimates,
        family=args.sensitivity_family,
        primary_block_size_km=args.primary_block_size_km,
        plot_void_volume_error=args.plot_void_volume_error,
    )
    saved.extend(
        save_figure(
            sensitivity,
            output_dir
            / f"spatial_bootstrap_block_size_sensitivity_{args.sensitivity_family}",
            formats=args.formats,
            dpi=args.dpi,
        )
    )

    comparison = plot_paired_method_differences(
        paired,
        estimates,
        primary_block_size_km=args.primary_block_size_km,
        plot_void_volume_error=args.plot_void_volume_error,
        xlim_scope=args.xlim_scope,
    )
    saved.extend(
        save_figure(
            comparison,
            output_dir / "fig_04",
            formats=args.formats,
            dpi=args.dpi,
        )
    )

    print("Saved figures:")
    for path in saved:
        print(f"  {path}")
    print(
        "Interpretation: the whiskers are confidence intervals on aggregate "
        "test-set performance, not predictive uncertainty intervals."
    )


if __name__ == "__main__":
    main()

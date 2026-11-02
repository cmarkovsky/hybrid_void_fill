#!/usr/bin/env python3

# Changes: default paths only (repository metadata; --chip-dir and --output-base required).
'Draw Figure 2 for the training, validation, and manuscript test populations.'

from __future__ import annotations


import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPS = ("train", "val", "test")
REGIONS = ("whim", "ehim")

GROUP_TITLES = {
    "train": "Training glaciers",
    "val": "Validation glaciers",
    "test": "Test glaciers",
}

REGION_TITLES = {
    "whim": "Western Himalaya",
    "ehim": "Eastern Himalaya",
}

COL_NONVOID = "#1f77b4"
COL_VOID = "#ff7f0e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final Figure 2 from artificial-void glacier chips."
    )
    parser.add_argument(
        "--chip-dir",
        type=Path,
        required=True,
        help="Directory containing one NPZ chip per glacier.",
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path(__file__).resolve().parents[3] / 'data/metadata/glacier_attributes.csv',
        help="CSV containing RGIId and, if available, group and region.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(__file__).resolve().parents[3] / 'data/metadata/splits/evaluation_split.json',
        help="Glacier-to-group JSON. Default: the production split with evaluation exclusions removed, so test panels show the evaluated population and train/val panels are unchanged. Pass data/metadata/splits/training_split.json for the assigned population.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).resolve().parents[3] / 'data/metadata/realization_inventory.csv',
        help=(
            "Final void metadata. Supplies region and, when present, the valid "
            "realizations for each glacier. CSV and Parquet are supported."
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        required=True,
        help="Output path without extension; PNG and PDF are written.",
    )
    parser.add_argument("--bin-width-m", type=float, default=50.0)
    parser.add_argument("--x-padding-m", type=float, default=5.0)
    parser.add_argument(
        "--z-percentiles",
        nargs=2,
        type=float,
        default=(0.0, 100.0),
        help="Full data range is required. Kept for CLI compatibility.",
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--sample-chips-for-bins",
        type=int,
        default=0,
        help="Number of chips used to set elevation bins; 0 uses the full inventory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--glacier-weighting",
        choices=("pixel", "equal"),
        default="equal",
        help=(
            "equal: equal glacier and realization weight (primary); pixel: "
            "pixel-pooled distributions with only 1/R_g realization weights."
        ),
    )
    parser.add_argument(
        "--min-bin-weight",
        type=float,
        default=1.0,
        help="Do not draw dh/dt summaries for bins below this weighted count.",
    )
    return parser.parse_args()


def normalize_group(value: object) -> str | None:
    text = str(value).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(text)


def normalize_region(value: object) -> str | None:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "whim": "whim",
        "western": "whim",
        "western himalaya": "whim",
        "ehim": "ehim",
        "eastern": "ehim",
        "eastern himalaya": "ehim",
    }
    return aliases.get(text)


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def load_assignments(
    attributes_path: Path,
    split_path: Path,
    metadata_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, list[object]]]:
    """Return group map, region map, and raw valid-realization labels."""

    group_map: dict[str, str] = {}
    region_map: dict[str, str] = {}
    realization_labels: dict[str, list[object]] = {}

    if attributes_path.exists():
        attrs = pd.read_csv(attributes_path)
        id_col = find_column(attrs, ("RGIId", "rgiid"))
        if id_col is None:
            raise ValueError(f"No RGIId column in {attributes_path}")

        group_col = find_column(attrs, ("group", "split", "partition"))
        region_col = find_column(attrs, ("region", "study_region"))

        for _, row in attrs.iterrows():
            rid = str(row[id_col])
            if group_col is not None:
                group = normalize_group(row[group_col])
                if group is not None:
                    group_map[rid] = group
            if region_col is not None:
                region = normalize_region(row[region_col])
                if region is not None:
                    region_map[rid] = region

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as stream:
            split = json.load(stream)
        for rid_raw, group_raw in split.items():
            group = normalize_group(group_raw)
            if group is None:
                raise ValueError(
                    f"Unexpected group {group_raw!r} for glacier {rid_raw} in {split_path}"
                )
            group_map[str(rid_raw)] = group
        # The split is authoritative: glaciers it omits (e.g. excluded ones) are dropped
        # even when the attribute table still assigns them a group.
        group_map = {rid: group for rid, group in group_map.items() if rid in split}

    if metadata_path.exists():
        meta = read_table(metadata_path)
        id_col = find_column(meta, ("RGIId", "rgiid"))
        if id_col is None:
            raise ValueError(f"No RGIId column in {metadata_path}")

        region_col = find_column(meta, ("region", "study_region"))
        realization_col = find_column(
            meta,
            (
                "realization_index",
                "realization",
                "realisation",
                "realization_id",
                "mask_id",
            ),
        )

        meta = meta.copy()
        meta[id_col] = meta[id_col].astype(str)

        if region_col is not None:
            for rid, values in meta.groupby(id_col, sort=False)[region_col]:
                regions = {
                    region
                    for region in (normalize_region(v) for v in values.dropna())
                    if region is not None
                }
                if len(regions) > 1:
                    raise ValueError(f"Multiple regions recorded for {rid}: {regions}")
                if regions:
                    region_map[str(rid)] = next(iter(regions))

        if realization_col is not None:
            if realization_col.lower() == "realization_index":
                realization_labels = {
                    str(rid): [f"idx:{int(value)}" for value in values.dropna()]
                    for rid, values in meta.groupby(id_col, sort=False)[
                        realization_col
                    ]
                }
            else:
                realization_labels = {
                    str(rid): list(values.dropna())
                    for rid, values in meta.groupby(id_col, sort=False)[
                        realization_col
                    ]
                }

    if not group_map:
        raise ValueError(
            "No glacier group assignments found. Supply --split or an attributes "
            "CSV containing a group column."
        )
    if not region_map:
        raise ValueError(
            "No glacier region assignments found. Supply --metadata or an attributes "
            "CSV containing a region column."
        )

    return group_map, region_map, realization_labels


def realization_indices(labels: list[object], n_masks: int) -> list[int]:
    """Convert r0-r4 or r1-r5 style labels to zero-based mask indices."""

    explicit = [
        int(str(label).split(":", 1)[1])
        for label in labels
        if str(label).startswith("idx:")
    ]
    if explicit:
        return sorted({index for index in explicit if 0 <= index < n_masks})

    numbers: list[int] = []
    for label in labels:
        match = re.search(r"(\d+)$", str(label).strip())
        if match:
            numbers.append(int(match.group(1)))

    numbers = sorted(set(numbers))
    if not numbers:
        return []

    if 0 in numbers:
        indices = numbers
    elif min(numbers) >= 1 and max(numbers) <= n_masks:
        indices = [number - 1 for number in numbers]
    else:
        indices = numbers

    return [index for index in indices if 0 <= index < n_masks]


def valid_masks_for_glacier(
    rid: str,
    masks: np.ndarray,
    realization_labels: dict[str, list[object]],
    base: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected 2D or 3D void_mask for {rid}; found {masks.shape}")

    if masks.shape[1:] != base.shape:
        raise ValueError(
            f"void_mask shape mismatch for {rid}: {masks.shape[1:]} vs {base.shape}"
        )

    if rid in realization_labels:
        indices = realization_indices(realization_labels[rid], masks.shape[0])
    else:
        indices = list(range(masks.shape[0]))

    # Empty masks do not represent usable artificial-void realizations.
    indices = [index for index in indices if np.any(masks[index].astype(bool) & base)]
    return masks.astype(bool), indices


def sample_elevation_edges(
    chip_paths: list[Path],
    bin_width_m: float,
    percentiles: tuple[float, float],
    n_sample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the entire finite reference support without sampling or trimming."""
    if not chip_paths or bin_width_m <= 0:
        raise ValueError("Eligible chips and a positive bin width are required")
    if tuple(percentiles) != (0.0, 100.0) or n_sample != 0:
        raise ValueError("Use --z-percentiles 0 100 and --sample-chips-for-bins 0 to retain all data")
    low=np.inf
    high=-np.inf
    for path in chip_paths:
        with np.load(path,allow_pickle=False) as chip:
            z=np.asarray(chip["z"],dtype=float)
            y=np.asarray(chip["dhdt"],dtype=float)
            mask=np.asarray(chip["glacier_mask"],dtype=bool)
        support=mask & np.isfinite(z) & np.isfinite(y)
        if not support.any():
            raise ValueError(f"No finite reference support for {path.stem}")
        low=min(low,float(z[support].min()))
        high=max(high,float(z[support].max()))
    first=np.floor(low/bin_width_m)*bin_width_m
    last=(np.floor(high/bin_width_m)+1)*bin_width_m
    edges=np.arange(first,last+bin_width_m,bin_width_m,dtype=float)
    return edges,(edges[:-1]+edges[1:])/2


def empty_stats(n_bins: int) -> dict[str, np.ndarray]:
    return {
        "weight": np.zeros(n_bins, dtype=float),
        "sum": np.zeros(n_bins, dtype=float),
        "sumsq": np.zeros(n_bins, dtype=float),
    }


def add_values(
    stats: dict[str, np.ndarray],
    bin_indices: np.ndarray,
    values: np.ndarray,
    observation_weight: float,
    n_bins: int,
) -> None:
    if values.size == 0:
        return
    weights = np.full(values.size, observation_weight, dtype=float)
    stats["weight"] += np.bincount(bin_indices, weights=weights, minlength=n_bins)
    stats["sum"] += np.bincount(
        bin_indices, weights=weights * values, minlength=n_bins
    )
    stats["sumsq"] += np.bincount(
        bin_indices, weights=weights * values * values, minlength=n_bins
    )


def weighted_mean_std(
    stats: dict[str, np.ndarray], min_weight: float
) -> tuple[np.ndarray, np.ndarray]:
    weight = stats["weight"]
    mean = np.full(weight.shape, np.nan, dtype=float)
    std = np.full(weight.shape, np.nan, dtype=float)
    valid = weight >= min_weight
    mean[valid] = stats["sum"][valid] / weight[valid]
    variance = stats["sumsq"][valid] / weight[valid] - mean[valid] ** 2
    std[valid] = np.sqrt(np.maximum(variance, 0.0))
    return mean, std


def density_from_weight(weight: np.ndarray, bin_width: float) -> np.ndarray:
    total = np.sum(weight)
    if total <= 0:
        return np.zeros_like(weight)
    return weight / (total * bin_width)


def main() -> None:
    args = parse_args()
    if not args.split.is_file() or not args.metadata.is_file():
        raise FileNotFoundError("The final split and realization inventory are required")
    if args.x_padding_m < 0:
        raise ValueError("X-axis padding must be nonnegative")

    group_map, region_map, realization_labels = load_assignments(
        attributes_path=args.attributes,
        split_path=args.split,
        metadata_path=args.metadata,
    )
    inventory=read_table(args.metadata)
    inventory_id=find_column(inventory,("rgiid","RGIId"))
    expected_valid={}
    if "n_valid_glacier_pixels" in inventory.columns:
        support_counts=inventory.groupby(inventory_id)["n_valid_glacier_pixels"]
        if (support_counts.nunique()!=1).any():
            raise ValueError("Realizations disagree on reference support")
        expected_valid={str(k):int(v) for k,v in support_counts.first().items()}

    all_chips = sorted(args.chip_dir.glob("*.npz"))
    chip_paths = [
        path
        for path in all_chips
        if path.stem in group_map
        and group_map[path.stem] in GROUPS
        and path.stem in region_map
        and region_map[path.stem] in REGIONS
    ]
    missing_chips=sorted(set(group_map)-{path.stem for path in chip_paths})
    if missing_chips:
        raise FileNotFoundError(f"Missing chips or region assignments for {len(missing_chips)} retained glaciers. First missing IDs: {missing_chips[:5]}")
    if not chip_paths:
        raise ValueError(
            f"No eligible chips found in {args.chip_dir}. Check split and region IDs."
        )

    z_edges, z_centers = sample_elevation_edges(
        chip_paths=chip_paths,
        bin_width_m=args.bin_width_m,
        percentiles=tuple(args.z_percentiles),
        n_sample=args.sample_chips_for_bins,
        seed=args.seed,
    )
    n_bins = len(z_centers)

    stats: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for region in REGIONS:
        for group in GROUPS:
            for pixel_class in ("nonvoid", "void"):
                stats[(region, group, pixel_class)] = empty_stats(n_bins)

    glaciers_used: Counter[tuple[str, str]] = Counter()
    realization_distribution: Counter[int] = Counter()
    realization_count_by_group: Counter[tuple[str, str]] = Counter()
    skipped: Counter[str] = Counter()
    outside_elevation_bins = 0
    finite_reference_pixels = 0

    for chip_path in chip_paths:
        rid = chip_path.stem
        group = group_map[rid]
        region = region_map[rid]

        with np.load(chip_path, allow_pickle=False) as chip:
            required = {"glacier_mask", "void_mask", "z", "dhdt"}
            missing = required - set(chip.files)
            if missing:
                skipped[f"missing {','.join(sorted(missing))}"] += 1
                continue

            glacier_mask = np.asarray(chip["glacier_mask"], dtype=bool)
            masks_raw = np.asarray(chip["void_mask"], dtype=bool)
            elevation = np.asarray(chip["z"], dtype=float)
            dhdt = np.asarray(chip["dhdt"], dtype=float)

        if not (
            glacier_mask.shape == elevation.shape == dhdt.shape
        ):
            raise ValueError(f"Raster shape mismatch for {rid}")

        base = glacier_mask & np.isfinite(elevation) & np.isfinite(dhdt)
        if rid in expected_valid and int(base.sum())!=expected_valid[rid]:
            raise ValueError(f"Reference support differs from the final inventory for {rid}")
        if not np.any(base):
            skipped["no finite glacier pixels"] += 1
            continue

        masks, valid_indices = valid_masks_for_glacier(
            rid=rid,
            masks=masks_raw,
            realization_labels=realization_labels,
            base=base,
        )
        r_g = len(valid_indices)
        if r_g != 5:
            raise ValueError(f"Expected five retained realizations for {rid}, found {r_g}")
        if r_g == 0:
            skipped["no valid void realizations"] += 1
            continue

        z_flat = elevation[base]
        dhdt_flat = dhdt[base]
        positions = np.argwhere(base)
        bin_index = np.digitize(z_flat, z_edges) - 1
        inside = (bin_index >= 0) & (bin_index < n_bins)
        outside_elevation_bins += int((~inside).sum())
        finite_reference_pixels += int(inside.size)
        if not np.all(inside):
            raise ValueError(f"Full-range bins failed to contain all elevations for {rid}")
        if not np.any(inside):
            skipped["outside elevation range"] += 1
            continue

        z_bin = bin_index[inside]
        dhdt_valid = dhdt_flat[inside]
        positions = positions[inside]

        realization_weight = 1.0 / r_g
        if args.glacier_weighting == "equal":
            realization_weight /= len(dhdt_valid)

        for index in valid_indices:
            void_here = masks[index][positions[:, 0], positions[:, 1]]
            nonvoid_here = ~void_here

            add_values(
                stats[(region, group, "void")],
                z_bin[void_here],
                dhdt_valid[void_here],
                realization_weight,
                n_bins,
            )
            add_values(
                stats[(region, group, "nonvoid")],
                z_bin[nonvoid_here],
                dhdt_valid[nonvoid_here],
                realization_weight,
                n_bins,
            )

        glaciers_used[(region, group)] += 1
        realization_count_by_group[(region, group)] += r_g
        realization_distribution[r_g] += 1

    if skipped:
        raise ValueError(f"Refusing to render an incomplete final population: {dict(skipped)}")
    if sum(glaciers_used.values()) != len(group_map):
        raise ValueError("Not every retained glacier contributed")
    if not glaciers_used:
        raise ValueError("No glaciers were successfully accumulated.")

    print(f"Eligible chips: {len(chip_paths):,}")
    print(
        f"Elevation bins: {n_bins} x {args.bin_width_m:g} m "
        f"from {z_edges[0]:g} to {z_edges[-1]:g} m"
    )
    print(f"Realizations per glacier: {dict(sorted(realization_distribution.items()))}")
    print(f"Glacier weighting: {args.glacier_weighting}")
    if skipped:
        print(f"Skipped: {dict(skipped)}")
    for region in REGIONS:
        for group in GROUPS:
            n_glaciers = glaciers_used[(region, group)]
            n_realizations = realization_count_by_group[(region, group)]
            print(
                f"{region} {group}: {n_glaciers:,} glaciers, "
                f"{n_realizations:,} retained realizations"
            )

    mosaic = [
        ["whim_train_pdf", "whim_val_pdf", "whim_test_pdf"],
        ["whim_train_dhdt", "whim_val_dhdt", "whim_test_dhdt"],
        ["ehim_train_pdf", "ehim_val_pdf", "ehim_test_pdf"],
        ["ehim_train_dhdt", "ehim_val_dhdt", "ehim_test_dhdt"],
    ]
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(14.5, 8.0),
        height_ratios=(2, 1, 2, 1),
        layout="constrained",
        sharex=True,
    )
    histogram_max_density = max(
        float(
            np.max(
                density_from_weight(
                    stats[(region, group, kind)]["weight"], args.bin_width_m
                )
            )
        )
        for region in REGIONS
        for group in GROUPS
        for kind in ("nonvoid", "void")
    )
    histogram_ymax = max(
        1e-3,
        np.ceil(histogram_max_density * 1.05 / 1e-3) * 1e-3,
    )

    panel_labels = {
        "whim_train_pdf": "(a)",
        "whim_val_pdf": "(b)",
        "whim_test_pdf": "(c)",
        "ehim_train_pdf": "(d)",
        "ehim_val_pdf": "(e)",
        "ehim_test_pdf": "(f)",
    }

    for region in REGIONS:
        for group in GROUPS:
            pdf_ax = axes[f"{region}_{group}_pdf"]
            line_ax = axes[f"{region}_{group}_dhdt"]

            nonvoid = stats[(region, group, "nonvoid")]
            void = stats[(region, group, "void")]

            nonvoid_density = density_from_weight(
                nonvoid["weight"], args.bin_width_m
            )
            void_density = density_from_weight(void["weight"], args.bin_width_m)

            pdf_ax.bar(
                z_centers,
                nonvoid_density,
                width=args.bin_width_m,
                color=COL_NONVOID,
                alpha=0.45,
                linewidth=0.25,
                edgecolor="0.2",
                label="Non-void pixels",
                zorder=2,
            )
            pdf_ax.bar(
                z_centers,
                void_density,
                width=args.bin_width_m,
                color=COL_VOID,
                alpha=0.45,
                linewidth=0.25,
                edgecolor="0.2",
                label="Void pixels",
                zorder=3,
            )
            pdf_ax.grid(axis="y", color="0.85", linewidth=0.6, zorder=0)
            pdf_ax.set_xlim(z_edges[0] - args.x_padding_m, z_edges[-1] + args.x_padding_m)
            # pdf_ax.set_ylim(0, histogram_ymax)
            pdf_ax.set_ylim(0, 1.1 * histogram_max_density)
            pdf_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            pdf_ax.tick_params(axis="x", labelbottom=False)
            pdf_ax.text(
                0.02,
                0.95,
                panel_labels[f"{region}_{group}_pdf"],
                transform=pdf_ax.transAxes,
                ha="left",
                va="top",
            )

            n_glaciers = glaciers_used[(region, group)]
            if group == "val":
                title = (
                    rf"$\bf{{{REGION_TITLES[region].replace(' ', r'\ ')}}}$"
                    + "\n"
                    + f"{GROUP_TITLES[group]} ($N={format(n_glaciers, ',').replace(',', '{,}')}$)"
                )
            else:
                title = f"{GROUP_TITLES[group]} ($N={format(n_glaciers, ',').replace(',', '{,}')}$)"
            pdf_ax.set_title(title)

            nonvoid_mean, nonvoid_std = weighted_mean_std(
                nonvoid, args.min_bin_weight
            )
            void_mean, void_std = weighted_mean_std(void, args.min_bin_weight)

            line_ax.plot(
                z_centers,
                nonvoid_mean,
                color=COL_NONVOID,
                linewidth=2.3,
                label=r"Non-void mean $dh/dt$",
                zorder=4,
            )
            line_ax.fill_between(
                z_centers,
                nonvoid_mean - nonvoid_std,
                nonvoid_mean + nonvoid_std,
                color=COL_NONVOID,
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            line_ax.plot(
                z_centers,
                void_mean,
                color=COL_VOID,
                linewidth=2.3,
                linestyle="--",
                label=r"Void mean $dh/dt$",
                zorder=5,
            )
            line_ax.fill_between(
                z_centers,
                void_mean - void_std,
                void_mean + void_std,
                color=COL_VOID,
                alpha=0.13,
                linewidth=0,
                zorder=3,
            )
            line_ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0)
            line_ax.grid(axis="y", color="0.85", linewidth=0.6, zorder=0)
            line_ax.set_xlim(z_edges[0] - args.x_padding_m, z_edges[-1] + args.x_padding_m)
            line_ax.set_ylim(-2.75, 2.75)
            line_ax.set_yticks((-2, -1, 0, 1, 2))
            line_ax.tick_params(
                axis="y", labelleft=(group == "train"), labelsize=9
            )
            line_ax.set_xlabel("Elevation (m)")
            line_ax.tick_params(axis="x",labelbottom=True)

            if group == "train":
                pdf_ax.set_ylabel("Density")
                line_ax.set_ylabel(r"$dh/dt$ (m a$^{-1}$)", labelpad=10)
                pdf_ax.legend(loc="upper right", fontsize=8)
                line_ax.legend(loc="upper left", fontsize=8, ncol=2)
            else:
                pdf_ax.set_ylabel("")
                line_ax.set_ylabel("")
                pdf_ax.tick_params(axis="y", labelleft=False)

    occupied=np.any(np.stack([v["weight"]>0 for v in stats.values()]),axis=0)
    first,last=np.flatnonzero(occupied)[[0,-1]]
    shared_xlim=(float(z_edges[first]-args.x_padding_m),float(z_edges[last+1]+args.x_padding_m))
    for ax in axes.values():
        ax.set_xlim(shared_xlim)
    assert all(np.allclose(ax.get_xlim(),shared_xlim) for ax in axes.values())
    assert len(axes[next(iter(axes))].get_shared_x_axes().get_siblings(axes[next(iter(axes))]))==12
    audit={
        "status":"regenerated from supplied chip data",
        "glaciers":sum(glaciers_used.values()),
        "groups":{f"{r}_{g}":glaciers_used[(r,g)] for r in REGIONS for g in GROUPS},
        "realizations_per_glacier":dict(realization_distribution),
        "finite_reference_pixels":finite_reference_pixels,
        "pixels_outside_elevation_bins":outside_elevation_bins,
        "shared_xlim_m":shared_xlim,
        "bin_width_m":args.bin_width_m,
        "all_12_axes_share_x":True,
        "bin_range_source":"all finite elevation and reference-rate pixels in retained glaciers",
        "percentile_trimming":False,
        "split":str(args.split),"metadata":str(args.metadata),"chip_dir":str(args.chip_dir),
    }
    rows=[]
    for (region,group,pixel_class),v in stats.items():
        mean,std=weighted_mean_std(v,args.min_bin_weight)
        density=density_from_weight(v["weight"],args.bin_width_m)
        for j in range(n_bins):
            rows.append(dict(region=region,group=group,pixel_class=pixel_class,
                bin_low_m=z_edges[j],bin_high_m=z_edges[j+1],weight=v["weight"][j],
                weighted_sum=v["sum"][j],weighted_sumsq=v["sumsq"][j],
                density=density[j],mean_dhdt=mean[j],sd_dhdt=std[j]))
    args.output_base.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(str(args.output_base)+"_bin_statistics.csv",index=False)
    Path(str(args.output_base)+"_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("Shared full-data elevation limits:",shared_xlim)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_base.with_suffix(".png")
    pdf_path = args.output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

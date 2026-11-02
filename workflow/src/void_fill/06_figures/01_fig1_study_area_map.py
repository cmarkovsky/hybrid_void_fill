#!/usr/bin/env python3
"""
plot_hma_glacier_context_map.py

Map-only Figure 1 panel:
- geographic context over Asia
- glacier outlines for study glaciers
- Western / Eastern Himalaya highlighted

Requires:
    geopandas
    pandas
    matplotlib
    cartopy
"""

# Shared project-root and run-profile configuration (also supports direct CLI use).
import sys as _workflow_sys
from pathlib import Path as _WorkflowPath
_workflow_sys.path.insert(0, str(_WorkflowPath(__file__).resolve().parents[3]))
from workflow_support import workflow_path, project_path, settings, parse_stage_args, check_row_inventory, write_provenance

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

GLACIER_SHP = workflow_path('glacier_shapes')
ATTR_CSV = workflow_path('attributes')
SPLIT_JSON = workflow_path('split')
HIMAP_REGIONS = workflow_path('region_shapes')
OUT_FIG = workflow_path('figures') / 'fig_01.png'

SAVE_FIG = True
DPI = 300

RGIID_COL_SHP = "RGIId"
RGIID_COL_CSV = "RGIId"
REGION_COL = "region"   # expected values: whim, ehim

REGION_LABELS = {
    "whim": "Western Himalaya",
    "ehim": "Eastern Himalaya",
}

REGION_COLORS = {
    "whim": "#CC79A7",   # muted purple
    "ehim": "#008B8B",   # muted crimson
}

ALL_GLACIER_COLOR = "0.35"  # medium gray
ALL_GLACIER_LINEWIDTH = 0.20
REGION_LINEWIDTH = 0.45

FIGSIZE = (12, 8)

# broad Asia context extent
MAP_EXTENT = [68, 98, 24, 40]  # [lon_min, lon_max, lat_min, lat_max]
INSET_EXTENT = [40, 120, 5, 55]  # broader extent for inset map
SHOW_REGION_CIRCLES = False

# approximate centers/radii for context circles (degrees)
WHIM_CENTER = (72.5, 35.5)
WHIM_RADIUS = 5.3

EHIM_CENTER = (91.0, 28.8)
EHIM_RADIUS = 4.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glacier-shp", type=Path, default=GLACIER_SHP)
    parser.add_argument("--attributes", type=Path, default=ATTR_CSV)
    parser.add_argument(
        "--split",
        type=Path,
        default=SPLIT_JSON,
        help=(
            "Optional glacier ID selection. IDs absent from the attributes CSV "
            "are skipped; plotted glaciers must also be listed in that CSV."
        ),
    )
    parser.add_argument("--himap-regions", type=Path, default=HIMAP_REGIONS)
    parser.add_argument("--output", type=Path, default=OUT_FIG)
    parser.add_argument("--dpi", type=int, default=DPI)
    return parse_stage_args(parser)

def add_scalebar_lonlat(
    ax,
    length_km=500,
    location=(0.08, 0.06),
    linewidth=0.8,
    text_offset=0.012,
    height_frac=0.015,
    n_segments=2,
    units="km",
    fontsize=8,
):
    """
    Add a simple scale bar to a lon/lat Cartopy map.

    Parameters
    ----------
    ax : cartopy/matplotlib axis
    length_km : total scalebar length in km
    location : tuple
        (x_frac, y_frac) in axes coordinates for lower-left corner
    linewidth : outline width
    text_offset : vertical offset for labels in axes coordinates
    height_frac : height of bar in axes coordinates
    n_segments : number of alternating segments
    units : label units
    fontsize : label size
    """
    import cartopy.crs as ccrs
    import matplotlib.patches as mpatches

    # current map extent in lon/lat
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())

    # anchor position in lon/lat
    lon0 = lon_min + location[0] * (lon_max - lon_min)
    lat0 = lat_min + location[1] * (lat_max - lat_min)

    # convert km to degrees longitude at latitude lat0
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))
    deg_len = length_km / km_per_deg_lon

    seg_len = deg_len / n_segments
    bar_h = height_frac * (lat_max - lat_min)

    # draw alternating rectangles
    for i in range(n_segments):
        x = lon0 + i * seg_len
        color = "black" if i % 2 == 0 else "white"
        rect = mpatches.Rectangle(
            (x, lat0),
            seg_len,
            bar_h,
            facecolor=color,
            edgecolor="black",
            linewidth=linewidth,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.add_patch(rect)

    # labels
    ax.text(
        lon0,
        lat0 + bar_h + text_offset * (lat_max - lat_min),
        "0",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    ax.text(
        lon0 + deg_len / 2,
        lat0 + bar_h + text_offset * (lat_max - lat_min),
        f"{int(length_km/2)}",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    ax.text(
        lon0 + deg_len,
        lat0 + bar_h + text_offset * (lat_max - lat_min),
        f"{int(length_km)} {units}",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import geopandas as gpd
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise ImportError(
            "Figure 1 requires geopandas, cartopy, matplotlib, and seaborn."
        ) from exc

    gdf = gpd.read_file(args.glacier_shp)
    attrs = pd.read_csv(args.attributes)

    if gdf.empty:
        raise RuntimeError(f"No glacier features found in {args.glacier_shp}")
    if attrs.empty:
        raise RuntimeError(f"No rows found in {args.attributes}")

    if RGIID_COL_SHP not in gdf.columns:
        raise ValueError(f"Shapefile missing '{RGIID_COL_SHP}' column")
    if RGIID_COL_CSV not in attrs.columns:
        raise ValueError(f"CSV missing '{RGIID_COL_CSV}' column")
    if REGION_COL not in attrs.columns:
        raise ValueError(f"CSV missing '{REGION_COL}' column")

    gdf[RGIID_COL_SHP] = gdf[RGIID_COL_SHP].astype(str)
    attrs[RGIID_COL_CSV] = attrs[RGIID_COL_CSV].astype(str)
    attrs[REGION_COL] = attrs[REGION_COL].astype(str).str.lower()

    if args.split.exists():
        with args.split.open("r", encoding="utf-8") as stream:
            split = json.load(stream)
        selected_ids = {str(rgiid) for rgiid in split}
        attribute_ids = set(attrs[RGIID_COL_CSV])
        missing_attributes = selected_ids.difference(attribute_ids)
        if missing_attributes:
            print(
                f"WARNING: skipping {len(missing_attributes)} split glacier IDs "
                f"absent from {args.attributes}; first IDs: "
                f"{sorted(missing_attributes)[:10]}"
            )
        attrs = attrs[attrs[RGIID_COL_CSV].isin(selected_ids)].copy()
        if attrs.empty:
            raise RuntimeError(
                f"No split glaciers are listed in the attributes CSV: {args.attributes}"
            )
    else:
        print(f"WARNING: split file not found; plotting all attribute rows: {args.split}")

    attribute_ids = set(attrs[RGIID_COL_CSV])
    missing_shapes = attribute_ids.difference(gdf[RGIID_COL_SHP])
    if missing_shapes:
        print(
            f"WARNING: {len(missing_shapes)} attribute glaciers have no matching "
            f"shapefile feature and will be skipped; first IDs: "
            f"{sorted(missing_shapes)[:10]}"
        )

    gdf = gdf.merge(
        attrs[[RGIID_COL_CSV, REGION_COL]],
        left_on=RGIID_COL_SHP,
        right_on=RGIID_COL_CSV,
        how="left",
        validate="many_to_one",
    )
    gdf = gdf[gdf[RGIID_COL_SHP].isin(attribute_ids)].copy()
    if gdf.empty:
        raise RuntimeError(
            "No shapefile features match the selected glacier IDs in "
            f"{args.attributes}"
        )
    
    if gdf[REGION_COL].isna().all():
        raise RuntimeError("All merged region values are missing. Check RGIId formatting.")

    # Ensure geographic CRS for Cartopy plotting
    if gdf.crs is None:
        raise ValueError("Glacier shapefile has no CRS defined.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    proj = ccrs.PlateCarree()
    himap_gdf = gpd.read_file(args.himap_regions)
    if himap_gdf.crs is None:
        raise ValueError("HIMAP region file has no CRS defined.")
    if himap_gdf.crs.to_epsg() != 4326:
        himap_gdf = himap_gdf.to_crs(4326)
    import matplotlib.gridspec as gridspec

    himap_gdf_plot = himap_gdf[himap_gdf["himap_region"].isin(["Western Himalaya", "Eastern Himalaya"])]

    # fig = plt.figure(figsize=FIGSIZE)
    # ax = plt.axes(projection=proj)
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.08)
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_hist = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax_map.set_extent(MAP_EXTENT, crs=proj) # type: ignore
    ax_map.set_aspect("auto")
    # Basemap features
    ax_map.add_feature(cfeature.OCEAN, facecolor="#dbeaf4", zorder=0) # type: ignore
    ax_map.add_feature(cfeature.LAND, facecolor="#f4f1ed", zorder=0) # type: ignore
    ax_map.add_feature(cfeature.LAKES, facecolor="#dbeaf4", edgecolor="none", zorder=0) # type: ignore
    ax_map.add_feature(cfeature.RIVERS, edgecolor="#b9d7ea", linewidth=0.3, zorder=0) # type: ignore

    ax_map.add_feature(cfeature.BORDERS, edgecolor="0.55", linewidth=0.5, zorder=1) # type: ignore
    ax_map.add_feature(cfeature.COASTLINE, edgecolor="0.45", linewidth=0.5, zorder=1) # type: ignore
    # All glaciers in light gray
    gdf.boundary.plot(
        ax=ax_map,
        transform=proj,
        color=ALL_GLACIER_COLOR,
        linewidth=ALL_GLACIER_LINEWIDTH,
        zorder=2,
    )
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ax_inset = inset_axes(
    #     ax_map,
    #     width="30%",
    #     height="30%",
    #     loc="lower left",
    #     borderpad=1,
    #     # axes_kwargs={"projection": ccrs.PlateCarree()},
    # )

    # --- small inset: [left, bottom, width, height] in figure coordinates ---
    ax_inset = fig.add_axes([0.01, 0.2, 0.225, 0.225], projection=proj) # type: ignore

    inset_extent = [40, 120, 5, 55]
    ax_inset.set_extent(inset_extent, crs=proj) # type: ignore

    ax_inset.add_feature(cfeature.OCEAN, facecolor="#dbeaf4", zorder=0) # type: ignore
    ax_inset.add_feature(cfeature.LAND, facecolor="#f4f1ed", zorder=0) # type: ignore
    ax_inset.add_feature(cfeature.BORDERS, edgecolor="0.55", linewidth=0.4, zorder=1) # type: ignore
    ax_inset.add_feature(cfeature.COASTLINE, edgecolor="0.45", linewidth=0.4, zorder=1) # type: ignore
 
    # ax_inset = plt.axes(
    #     [0.0, 0.0, 0.3, 0.3],  # [left, bottom, width, height] in figure coordinates
    #     projection=ccrs.PlateCarree()

    # )
    ax_inset.set_extent([40, 120, 5, 55]) # type: ignore
    ax_inset.add_feature(cfeature.OCEAN, facecolor="#dbeaf4", zorder=0) # type: ignore

    # ax_inset.set_extent(INSET_EXTENT, crs=proj)
    # ax_inset = plt.axes(
    # ax_inset.get_position(),
    #     projection=ccrs.PlateCarree()
    # )
    # ax_inset.set_extent(INSET_EXTENT)
    # import matplotlib.patches as mpatches

    # rect = mpatches.Rectangle(
    #     (68, 26),
    #     98 - 68,
    #     38 - 26,
    #     fill=False,
    #     edgecolor="black",
    #     linewidth=1,
    #     transform=ccrs.PlateCarree()
    # )

    # ax_inset.add_patch(rect)
    print(len(gdf), "glacier outlines plotted in light gray for context")
    for himap_region, color in REGION_COLORS.items():
        himap_gdf_plot[himap_gdf_plot["himap_region"] == REGION_LABELS[himap_region]].plot(
            ax=ax_map,
            transform=proj,
            color=color,
            linewidth=0.5,
            alpha=0.1,
            zorder=2,
            label=REGION_LABELS[himap_region] + " boundary",
        )
        himap_gdf_plot[himap_gdf_plot["himap_region"] == REGION_LABELS[himap_region]].boundary.plot(
            ax=ax_map,
            transform=proj,
            color=color,
            linewidth=0.8,
            # alpha=0.1,
            zorder=2,
            label=REGION_LABELS[himap_region] + " boundary",
        )
        himap_gdf_plot[himap_gdf_plot["himap_region"] == REGION_LABELS[himap_region]].plot(
            ax=ax_inset,
            transform=proj,
            color=color,
            linewidth=0.5,
            alpha=0.1,
            zorder=2,
            label=REGION_LABELS[himap_region] + " boundary",
        )
        himap_gdf_plot[himap_gdf_plot["himap_region"] == REGION_LABELS[himap_region]].boundary.plot(
            ax=ax_inset,
            transform=proj,
            color=color,
            linewidth=0.8,
            # alpha=0.1,
            zorder=2,
            label=REGION_LABELS[himap_region] + " boundary",
        )

    # # Highlight each region
    # for region, color in REGION_COLORS.items():
    #     sub = gdf[gdf[REGION_COL] == region]
    #     if len(sub) == 0:
    #         continue

    #     sub.boundary.plot(
    #         ax=ax_map,
    #         transform=proj,
    #         color=color,
    #         linewidth=REGION_LINEWIDTH,
    #         zorder=3,
    #         label=REGION_LABELS.get(region, region),
    #     )

    # Optional circles showing the two study regions
    if SHOW_REGION_CIRCLES:
        whim_circle = Circle(
            WHIM_CENTER,
            radius=WHIM_RADIUS,
            transform=proj,
            fill=False,
            edgecolor=REGION_COLORS["whim"],
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
        )
        ehim_circle = Circle(
            EHIM_CENTER,
            radius=EHIM_RADIUS,
            transform=proj,
            fill=False,
            edgecolor=REGION_COLORS["ehim"],
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
        )
        ax_map.add_patch(whim_circle)
        ax_map.add_patch(ehim_circle)

    # Grid / ticks
    gl = ax_map.gridlines( # type: ignore
        crs=proj,
        draw_labels=True,
        linewidth=0.4,
        color="0.8",
        alpha=0.5,
        linestyle="--",
    ) # type: ignore

    from matplotlib.ticker import FixedLocator
    gl.top_labels = False
    gl.right_labels = False
   

    # hide every other label
    for i, label in enumerate(gl.ylabel_artists):
        if i % 2 == 1:
            label.set_visible(False)

    # Legend
    handles = [
        mpatches.Patch(color=REGION_COLORS["whim"], label="Western Himalaya"),
        mpatches.Patch(color=REGION_COLORS["ehim"], label="Eastern Himalaya"),
    ]
    ax_map.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        title_fontsize=9,
    )
    add_scalebar_lonlat(
        ax_map,
        length_km=500,
        location=(0.77, 0.02),
        n_segments=2,
        fontsize=8,
        )   

    # Panel label
    ax_map.text(
        0.02, 0.98, "(a)",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    ) # type: ignore
    ax_hist.text(
        0.02, 0.98, "(b)",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    bins = np.linspace(3500, 6500, 50)
    sns.histplot(
        data=attrs,
        x="Zmed",
        bins=bins,
        hue=REGION_COL,
        ax=ax_hist,
        palette=REGION_COLORS,
        edgecolor="0.2",
        linewidth=0.3,
        legend=False,
    )
    ax_hist.set_xlabel("Glacier median elevation (m)")
    ax_hist.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_ylabel("Count", rotation = -90, labelpad = 12)
    fig.canvas.draw()
    lat_ticks = np.arange(20, 50, 2)
    gl.ylocator = FixedLocator(lat_ticks) # type: ignore

    fig.canvas.draw()
    lat_labels = gl.ylabel_artists.copy()
    
    for txt in (gl.ylabel_artists):
        txt.set_visible(False)
        
    fig.canvas.draw()
    pos_map = ax_map.get_position()
    pos_hist = ax_hist.get_position()

    ax_hist.set_position([
        pos_hist.x0,
        pos_map.y0,
        pos_hist.width,
        pos_map.height,
    ]) # type: ignore

    # plt.tight_layout()
    if SAVE_FIG:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.output,
            dpi=args.dpi,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

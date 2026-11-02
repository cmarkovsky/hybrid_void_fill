#!/usr/bin/env python3
"""Regenerate every manuscript figure, PNG and PDF, under its final number.

Requires run_production_stages.py (subset tree) and recalculate_transfer_coordinates.py (recalc dir)
to have run. Writes <out>/main/figNN_*.{png,pdf}, <out>/supplement/figSNN_*.{png,pdf} and
<out>/FIGURE_SOURCES.json (script, inputs and population of every file).

    python make_figures.py --data-root <data_archive> --recalc <recalc dir> \
        --subset-tree <data_archive>/derived/manuscript --out <figures dir>
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import Paths, Selection, base_parser, sha256, write_json

HERE = Path(__file__).resolve().parent
POP_EVAL = "evaluated test population: 1,754 glaciers (1,422 W, 332 E), RGI60-14.17287 excluded"
POP_STUDY = "full production split: 11,685 study glaciers in all groups (exclusion not applicable)"

# number, stem, population, source script
MAIN = [
    ("01", "study_area", POP_STUDY, "workflow/src/void_fill/06_figures/01_fig1_study_area_map.py"),
    ("02", "distributions", "train 8,179 and validation 1,751 assigned glaciers; test panels evaluated 1,754", "analysis/manuscript/figures/fig02_distributions.py"),
    ("03", "pixel_residuals", POP_EVAL + "; 2,540,722 targets per method", "analysis/manuscript/figures/fig03_residuals.py"),
    ("04", "bootstrap_contrasts", POP_EVAL + "; 100 km blocks, 2,000 draws, seed 20260917", "workflow/src/void_fill/06_figures/04_fig4_spatial_bootstrap.py --xlim-scope row"),
    ("05", "glacier_wide", POP_EVAL, "analysis/manuscript/figures/fig05_glacier_wide.py (3 s.f. annotations)"),
]
SUPP = [
    ("01", "relative_improvement", POP_EVAL + "; 50/100/200 km blocks", "analysis/manuscript/figures/figS01_relative_improvement.py"),
    ("02", "void_realizations", "illustrative: RGI60-15.01282 and RGI60-14.16513 (geometry/metadata selection, protocol/case_selection.csv); unaffected by the exclusion", "archived geometry-selected void-realization artwork"),
    ("03", "reconstruction_examples", "illustrative: four geometry-selected test glaciers; unaffected by the exclusion", "analysis/manuscript/figures/figS03_reconstruction_examples.py"),
    ("04", "band_diagnostics", POP_EVAL, "analysis/manuscript/supplement_figures.py band_figure"),
    ("05", "realization_geometry", POP_EVAL + "; original geometry tertiles", "analysis/manuscript/supplement_figures.py realization_figure"),
    ("06", "transfer_separation_map", "1,000 + 1,000 source glaciers; evaluated test glaciers; excluded glacier marked", "analysis/manuscript/supplement_figures.py transfer_map"),
    ("07", "coordinate_per_glacier", POP_EVAL + "; full data range", "analysis/manuscript/supplement_figures.py coordinate_figures"),
    ("08", "coordinate_intervals", POP_EVAL + "; sequential RNG seed 20261019", "analysis/manuscript/supplement_figures.py coordinate_figures"),
]


def run(cmd, env=None):
    print(" ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--recalc", type=Path, required=True)
    ap.add_argument("--subset-tree", type=Path, required=True)
    a = ap.parse_args()
    paths, sel = Paths(a.data_root, a.repo), Selection(a.repo)
    out, tree = a.out.resolve(), a.subset_tree.resolve()
    work = out / "_work"
    for d in (out / "main", out / "supplement", work):
        d.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    wf = [py, "-B", a.repo / "workflow/workflow.py", "--root", paths.root, "--config", tree / "workflow_config.json"]
    env = dict(os.environ, VOID_FILL_PROJECT_ROOT=str(paths.root), VOID_FILL_WORKFLOW_CONFIG=str(tree / "workflow_config.json"))
    m = out / "main"
    for ext in ("png", "pdf"):
        run(wf + ["figure1", "--", "--split", a.repo / "data/metadata/splits/training_split.json", "--output", m / f"fig01_study_area.{ext}"])
        run([py, "-B", a.repo / "analysis/manuscript/figures/fig05_glacier_wide.py", "--output", m / f"fig05_glacier_wide.{ext}"], env)
    run([py, "-B", HERE / "figures/fig02_distributions.py", "--chip-dir", paths["chips"], "--output-base", work / "fig_02"])
    run([py, "-B", HERE / "figures/fig03_residuals.py", "--input", tree / "pixel_summaries/test_residuals_long_common_xy.parquet", "--output-base", work / "fig_03"])
    run(wf + ["figure4", "--", "--xlim-scope", "row", "--formats", "png", "pdf", "--output-dir", work / "fig4_row"])
    for ext in ("png", "pdf"):
        shutil.copy2(work / f"fig_02.{ext}", m / f"fig02_distributions.{ext}")
        shutil.copy2(work / f"fig_03.{ext}", m / f"fig03_pixel_residuals.{ext}")
        shutil.copy2(work / f"fig4_row/fig_04.{ext}", m / f"fig04_bootstrap_contrasts.{ext}")
    run([py, "-B", HERE / "supplement_figures.py", "--data-root", paths.root, "--repo", a.repo, "--recalc", a.recalc,
         "--subset-tree", tree, "--out", work / "supp"])
    s = out / "supplement"
    for ext in ("png", "pdf"):
        run([py, a.repo / "analysis/manuscript/figures/figS01_relative_improvement.py", tree / "spatial_block_bootstrap/paired_method_differences_ci.csv", s / f"figS01_relative_improvement.{ext}"])
        for num, stem, src in [("04", "band_diagnostics", "band_diagnostics"), ("05", "realization_geometry", "realization_geometry_sensitivity"),
                               ("06", "transfer_separation_map", "transfer_separation_map"), ("07", "coordinate_per_glacier", "coordinate_per_glacier_comparison"),
                               ("08", "coordinate_intervals", "coordinate_regional_intervals")]:
            shutil.copy2(work / f"supp/{src}.{ext}", s / f"figS{num}_{stem}.{ext}")
    rev = paths["sensitivity_metrics"].parent
    for ext in ("png", "pdf"):
        shutil.copy2(rev / f"figures/spatial_examples/figA_void_realizations.{ext}", s / f"figS02_void_realizations.{ext}")
    a3 = paths["reconstruction_examples"]
    run([py, "-B", HERE / "figures/figS03_reconstruction_examples.py", "--source-dir", a3, "--output", s / "figS03_reconstruction_examples.pdf"])
    for f in ["fig_02_audit.json", "fig_02_bin_statistics.csv", "fig_03_audit.json", "fig_03_metrics.csv"]:
        shutil.copy2(work / f, m / f)
    for f in ["band_diagnostics_strata.csv", "realization_sensitivity_summary.csv", "realization_envelope.csv", "geometry_fixed_tertiles.csv",
              "coordinate_per_glacier_subset.csv", "SUPPLEMENT_FIGURE_CHECKS.json"]:
        shutil.copy2(work / f"supp/{f}", s / f)
    shutil.rmtree(work)
    rec = {}
    for folder, items, prefix in [("main", MAIN, "fig"), ("supplement", SUPP, "figS")]:
        for num, stem, pop, script in items:
            for ext in ("png", "pdf"):
                f = out / folder / f"{prefix}{num}_{stem}.{ext}"
                rec[f"{folder}/{f.name}"] = dict(figure=("Fig. " if folder == "main" else "Fig. S") + num.lstrip("0"), population=pop,
                                                 source_script=script, sha256=sha256(f), bytes=f.stat().st_size)
    write_json(out / "FIGURE_SOURCES.json", rec)
    print(len(rec), "figure files")


if __name__ == "__main__":
    main()

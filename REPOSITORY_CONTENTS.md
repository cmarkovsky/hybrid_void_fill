# Exact repository contents

The cleaned GitHub folder contains **151 files**, including **43 Python files**, **five saved boosters**, and **26 final figure images**. All test-performance defaults exclude RGI60-14.17287. The original fitting split and full realization inventory retain the scientific experiment identity.

## Directory outline

| Location | Files | Purpose |
|---|---:|---|
| `(root)` | 10 | README, citation, environment, reproduction command, scientific model/data notes, validation, and this inventory. |
| `analysis` | 18 | Final manuscript analysis plus compact fixed-setting model fitting and prediction. |
| `config` | 3 | One default workflow, one evaluation definition, and one archive path map. |
| `data` | 11 | Scientific metadata and separate fitting/evaluation splits. |
| `figures` | 28 | Five main and eight supplementary figures (26 PDF/PNG images) and their source/population records. |
| `manuscript` | 19 | One main and one supplementary LaTeX source with required table, caption, and reference inputs. |
| `models` | 12 | Five unchanged XGBoost boosters with final feature definitions/settings and supplementary designs. |
| `results` | 23 | Final evaluated evidence, six result tables, per-glacier errors, and marginal/paired intervals. |
| `tests` | 2 | Two test files containing eight focused scientific/manuscript checks. |
| `workflow` | 25 | Raw-data/cube preparation, fixed-mask remapping, baselines, model rows, numerical results, and map/bootstrap plotting. |

## Core analysis commands

`reproduce.py` runs the complete final result sequence. `analysis/manuscript/` contains membership generation, core and transfer/coordinate recalculation, table building, numerical stage execution, figure assembly, and one comprehensive verifier. Its `figures/` directory contains the dedicated Figure 2, 3, 5, S1, and S3 sources. Figure S2 uses frozen geometry-selected artwork from the data archive.

`analysis/model/model_data.py` expands the stored rows with the manuscript weighting and stable capped sampling. `fit_predict.py` fits the primary hybrid or predicts the evaluated test targets; `fit_sensitivity.py` reproduces the four fixed-setting supplementary fits.

`workflow/src/void_fill/` follows the scientific stages: `01_datacube`, `02_voids`, `03_baselines`, `04_model`, `05_results`, and `06_figures`. The datacube stage retains six operations and shared raster/grid helpers; void handling retains remapping, metadata, and validation; model handling retains row preparation and context predictors. Obsolete tuning/training alternatives, experiment packaging, and new-experiment void generation are omitted.

## Retained numerical outputs

`results/manuscript/evidence/` holds the aggregate point metrics, main intervals, transfer metrics/intervals, coordinate metrics/intervals and per-glacier moments, band summaries, realization envelopes/rank sensitivity, and fixed geometry groups. `tables/` supplies six LaTeX tables, readable Markdown, and unrounded values. The three top-level CSV files retain per-glacier method errors, marginal bootstrap intervals, and paired/relative bootstrap intervals.

## Removed from GitHub

Revision/handoff/audit/assembly histories; obsolete project-version configurations; duplicate figure implementations; previous runners and gate/resource bookkeeping; overlapping pipeline/test suites and synthetic fixture files; stale author manuscript sources, draft proofs, reconciliation/edit logs; duplicated generated stage trees, preview tables, and extra check reports; Poetry/package-release scaffolding. Original files are preserved in the delivery-level cleanup backup, outside this GitHub folder. Bulk data and full historical experiment records remain in the separate data archive.

## Complete file list

Every file proposed for Git is listed below. There are no hidden additional test suites or model variants.

### (root)

```text
.gitignore
CITATION.cff
DATA_AVAILABILITY.md
MODEL_CARD.md
README.md
REPOSITORY_CONTENTS.md
REPRODUCE.md
VALIDATION.md
reproduce.py
requirements.txt
```

### analysis

```text
analysis/manuscript/README.md
analysis/manuscript/build_tables.py
analysis/manuscript/common.py
analysis/manuscript/figures/fig02_distributions.py
analysis/manuscript/figures/fig03_residuals.py
analysis/manuscript/figures/fig05_glacier_wide.py
analysis/manuscript/figures/figS01_relative_improvement.py
analysis/manuscript/figures/figS03_reconstruction_examples.py
analysis/manuscript/make_figures.py
analysis/manuscript/make_membership.py
analysis/manuscript/recalculate_core.py
analysis/manuscript/recalculate_transfer_coordinates.py
analysis/manuscript/run_production_stages.py
analysis/manuscript/supplement_figures.py
analysis/manuscript/verify_results.py
analysis/model/fit_predict.py
analysis/model/fit_sensitivity.py
analysis/model/model_data.py
```

### config

```text
config/data_root.example.json
config/evaluation.json
config/workflow.json
```

### data

```text
data/metadata/DATA_DICTIONARY.md
data/metadata/DATA_SOURCES.csv
data/metadata/block_assignments.csv
data/metadata/evaluation_exclusions.json
data/metadata/evaluation_membership.csv
data/metadata/excluded_glaciers.json
data/metadata/glacier_attributes.csv
data/metadata/glacier_geometry.csv
data/metadata/realization_inventory.csv
data/metadata/splits/evaluation_split.json
data/metadata/splits/training_split.json
```

### figures

```text
figures/FIGURE_POPULATIONS.md
figures/FIGURE_SOURCES.json
figures/main/fig01_study_area.pdf
figures/main/fig01_study_area.png
figures/main/fig02_distributions.pdf
figures/main/fig02_distributions.png
figures/main/fig03_pixel_residuals.pdf
figures/main/fig03_pixel_residuals.png
figures/main/fig04_bootstrap_contrasts.pdf
figures/main/fig04_bootstrap_contrasts.png
figures/main/fig05_glacier_wide.pdf
figures/main/fig05_glacier_wide.png
figures/supplement/figS01_relative_improvement.pdf
figures/supplement/figS01_relative_improvement.png
figures/supplement/figS02_void_realizations.pdf
figures/supplement/figS02_void_realizations.png
figures/supplement/figS03_reconstruction_examples.pdf
figures/supplement/figS03_reconstruction_examples.png
figures/supplement/figS04_band_diagnostics.pdf
figures/supplement/figS04_band_diagnostics.png
figures/supplement/figS05_realization_geometry.pdf
figures/supplement/figS05_realization_geometry.png
figures/supplement/figS06_transfer_separation_map.pdf
figures/supplement/figS06_transfer_separation_map.png
figures/supplement/figS07_coordinate_per_glacier.pdf
figures/supplement/figS07_coordinate_per_glacier.png
figures/supplement/figS08_coordinate_intervals.pdf
figures/supplement/figS08_coordinate_intervals.png
```

### manuscript

```text
manuscript/README.md
manuscript/main/figures_tex/fig_bootstrap.tex
manuscript/main/figures_tex/fig_distributions.tex
manuscript/main/figures_tex/fig_glacier.tex
manuscript/main/figures_tex/fig_residuals.tex
manuscript/main/figures_tex/fig_study.tex
manuscript/main/main.tex
manuscript/main/references.tex
manuscript/main/tables/tab_coordinates.tex
manuscript/main/tables/tab_data.tex
manuscript/main/tables/tab_features.tex
manuscript/main/tables/tab_glacier.tex
manuscript/main/tables/tab_metadata.tex
manuscript/main/tables/tab_paired.tex
manuscript/main/tables/tab_pixel.tex
manuscript/main/tables/tab_population.tex
manuscript/main/tables/tab_transfer.tex
manuscript/main/tables/tab_volume.tex
manuscript/supplement/supplement.tex
```

### models

```text
models/README.md
models/coordinate_ablation/coordinates_off/model.json
models/coordinate_ablation/coordinates_on/model.json
models/coordinate_ablation/design.json
models/coordinate_ablation/feature_schemas.json
models/hybrid/FEATURE_SCHEMA.json
models/hybrid/feature_importance_gain.csv
models/hybrid/model.json
models/hybrid/training_config.json
models/regional_transfer/design.json
models/regional_transfer/source_ehim.json
models/regional_transfer/source_whim.json
```

### results

```text
results/manuscript/README.md
results/manuscript/bootstrap_intervals.csv
results/manuscript/evidence/band_summaries.csv
results/manuscript/evidence/coordinate_all_intervals.csv
results/manuscript/evidence/coordinate_all_metrics.csv
results/manuscript/evidence/coordinate_per_glacier_moments.csv
results/manuscript/evidence/geometry_fixed_groups.csv
results/manuscript/evidence/main_paired_intervals.csv
results/manuscript/evidence/point_metrics.csv
results/manuscript/evidence/realization_envelopes.csv
results/manuscript/evidence/single_mask_rank_flips.csv
results/manuscript/evidence/transfer_intervals.csv
results/manuscript/evidence/transfer_metrics.csv
results/manuscript/paired_bootstrap_intervals.csv
results/manuscript/per_glacier_metrics.csv
results/manuscript/tables/tab_coordinates.tex
results/manuscript/tables/tab_glacier.tex
results/manuscript/tables/tab_paired.tex
results/manuscript/tables/tab_pixel.tex
results/manuscript/tables/tab_transfer.tex
results/manuscript/tables/tab_volume.tex
results/manuscript/tables/table_values_unrounded.json
results/manuscript/tables/tables_copy_ready.md
```

### tests

```text
tests/test_manuscript_results.py
tests/test_scientific_invariants.py
```

### workflow

```text
workflow/README.md
workflow/config/datacube.json
workflow/src/void_fill/01_datacube/01_define_grid.py
workflow/src/void_fill/01_datacube/02_align_rasters.py
workflow/src/void_fill/01_datacube/03_compute_terrain.py
workflow/src/void_fill/01_datacube/04_rasterize_glaciers.py
workflow/src/void_fill/01_datacube/05_build_datacube.py
workflow/src/void_fill/01_datacube/06_validate_datacube.py
workflow/src/void_fill/01_datacube/cubelib.py
workflow/src/void_fill/02_voids/01a_remap_frozen_chips.py
workflow/src/void_fill/02_voids/02_build_metadata.py
workflow/src/void_fill/02_voids/03_validate_inputs.py
workflow/src/void_fill/03_baselines/01_constant_local_mean.py
workflow/src/void_fill/03_baselines/02_hypsometric_local_mean.py
workflow/src/void_fill/04_model/01_prepare_rows.py
workflow/src/void_fill/04_model/context_features.py
workflow/src/void_fill/05_results/01_build_prediction_cache.py
workflow/src/void_fill/05_results/02_pixel_summary_stats.py
workflow/src/void_fill/05_results/03_glacier_wide_metrics.py
workflow/src/void_fill/05_results/04_glacier_wide_comparison.py
workflow/src/void_fill/05_results/05_spatial_block_bootstrap.py
workflow/src/void_fill/06_figures/01_fig1_study_area_map.py
workflow/src/void_fill/06_figures/04_fig4_spatial_bootstrap.py
workflow/workflow.py
workflow/workflow_support.py
```

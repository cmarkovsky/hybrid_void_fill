# Hybrid hypsometric–machine learning reconstruction of glacier elevation change

Code and supporting results for **A hybrid hypsometric–machine learning approach for reconstructing glacier surface elevation change**, by Cameron Markovsky, Summer Rupper, Simon Brewer, and Richard R. Forster.

The hybrid XGBoost model predicts departures from a realization-specific hypsometric estimate: target `y − h`, reconstruction `h + f(X)`. Constant, hypsometric, and hybrid fills are evaluated on identical artificial-void targets in the Western and Eastern Himalaya.

## Default manuscript evaluation

All performance calculations use **1,754 test glaciers**, excluding **RGI60-14.17287** and its five realizations: 1,422 Western and 332 Eastern glaciers, 8,770 glacier-realizations, and 2,540,722 target instances per method. No exclusion flag is needed.

`config/evaluation.json` defines this selection. `data/metadata/splits/evaluation_split.json` is the default workflow split. The original fitting assignments remain in `training_split.json`: 8,179 training, 1,751 validation, and 1,755 assigned test glaciers. The exclusion affects evaluation, not fitting or the stored predictions. Figure 1 maps the full study population; the reproduction script supplies the fitting split for that map explicitly.

The stored hybrid uses 25 predictors and 172 trees. Combined glacier-wide specific MAE is 0.0515 m a⁻¹ versus 0.0639 for hypsometric filling, a 19.4% reduction. The post hoc exclusion changes the sign of the integrated-RMSE comparison; see `MODEL_CARD.md`.

## Run

Use Python 3.12 and install `requirements.txt` in an isolated environment. The project runs directly from its scripts; no package installation or release tooling is needed.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce.py --data-root /absolute/path/to/data_archive --out /absolute/path/to/new_results
```

The reproduction command rebuilds membership, metrics, spatial-block intervals, Tables 5–10, and Figures 1–5/S1–S8, then verifies the results. It reads the supplied data archive and writes to the chosen output directory. It uses saved predictions and does not fit a model. See `REPRODUCE.md` for optional fitting and individual stages.

## Repository

| Directory | Contents |
|---|---|
| `analysis/manuscript/` | Final evaluation, tables, figures, and data-backed verification |
| `analysis/model/` | Fixed-setting hybrid fitting/prediction and four supplementary model fits |
| `workflow/` | Datacube preparation, voids, baselines, model rows, and numerical result stages |
| `config/` | Evaluation selection, workflow defaults, and data-archive input map |
| `data/metadata/` | Splits, exclusions, attributes, realization inventory, and spatial blocks |
| `models/` | Five manuscript boosters, feature definitions, and experimental designs |
| `results/manuscript/` | Final summary evidence, Tables 5–10, and numerical stage outputs |
| `figures/` | Five main and eight supplementary figures, each as PDF and PNG |
| `manuscript/` | One main LaTeX source and one supplement with final figure references |
| `tests/` | Eight focused scientific and manuscript checks in two files |

`REPOSITORY_CONTENTS.md` gives the complete file inventory. Bulk rasters, chips, row arrays, prediction arrays, and historical run records belong in the separate data archive. No data-archive upload is performed by these scripts. Citation and data-deposition details are in `CITATION.cff` and `DATA_AVAILABILITY.md`.

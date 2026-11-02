# Reproducing the manuscript

Run commands from the repository root with Python 3.12 and the dependencies in `requirements.txt`. The data archive is the supplied `data_archive/` directory. Its historical asset names are mapped in `config/data_root.example.json`; those names identify actual inputs, not alternate model versions offered by this repository.

## Complete result reproduction

```sh
python reproduce.py --data-root /absolute/path/to/data_archive --out /absolute/path/to/new_results
```

Choose an empty output directory. The command writes:

1. `metadata/`: the derived membership and default evaluation split.
2. `evidence/`: point metrics, main/transfer/coordinate intervals, and supplementary summaries. Full-population reference calculations are checked internally; published evidence contains only the selected evaluation population.
3. `tables/` and table-value summaries: the six numerical result tables (Tables 5–10), using three significant figures.
4. `pipeline_stages/`: a filtered keyed cache and pixel, glacier, comparison, and spatial-bootstrap outputs. All stage paths follow `--out`.
5. `figures/`: main Figures 1–5 and supplementary Figures S1–S8 as PNG/PDF.
6. `verification.json`: identities, membership, common targets, masks, denominators, area scaling, table values, manuscript values, and figure checks.

Use `--skip-figures` to omit the final figure rebuild. The stage runner still produces its standard diagnostic plots. Reproduction normally takes several minutes and requires the complete archive. It does not retrain or retune any model.

The default evaluation always excludes RGI60-14.17287. Training and validation retain their manuscript memberships. Do not substitute the training split for performance calculations. The study-area map uses all assigned study glaciers, and the supplementary transfer map marks the excluded glacier for context.

## Fixed-setting model reproduction

```sh
python analysis/model/fit_predict.py fit --data-root /absolute/path/to/data_archive --out /absolute/path/to/new_hybrid_fit
python analysis/model/fit_predict.py predict --data-root /absolute/path/to/data_archive --out /absolute/path/to/new_predictions
python analysis/model/fit_sensitivity.py --data-root /absolute/path/to/data_archive --out /absolute/path/to/new_sensitivity_fits
```

The primary fit uses all 11,792,199 training targets, the stored feature order, equal-glacier/realization weights, seed 42, and the final 172-round settings. It writes a new booster and preserves `models/hybrid/model.json`. Prediction defaults to the stored manuscript booster and the evaluated test targets. Use `--group val` for validation, or `--model <new_fit/model.json>` to predict with a reproduced fit.

The sensitivity script reproduces the two source-region models and the matched coordinate-off/on models from the recorded 1,000-glacier-per-region source lists. It uses the final fixed settings and capped samples, without a new search. Original tuning studies, development assignments, and run records remain in the data archive; the cleaned repository does not provide a second model-selection workflow.

## Individual stages and input rebuilding

```sh
python workflow/workflow.py --root /absolute/path/to/data_archive steps
python workflow/workflow.py --root /absolute/path/to/data_archive status
python workflow/workflow.py --root /absolute/path/to/data_archive cube-grid
python workflow/workflow.py --root /absolute/path/to/data_archive prepare -- --help
```

`workflow/config/datacube.json` points to the archived raw inputs and writes a rebuilt cube to `derived/hma_cube_aligned.zarr`. Datacube commands run in order: `cube-grid`, `cube-align`, `cube-terrain`, `cube-glaciers`, `cube`, `cube-validate` (or the `datacube` group). The workflow also retains remapping, metadata, input validation, both baselines, row preparation, keyed cache building, and result stages. Individual stage flags permit fresh output paths; use them when rebuilding prepared assets. The supplied masks and prepared rows are the manuscript inputs.

`config/workflow.json` is the default profile. Paths prefixed `@repo/` resolve relative to the checkout, all other relative paths resolve under `--root`. The default local-source rule is the deterministic strict-900 m rule, serialized as `v3` in the archived arrays. Earlier rule alternatives have been removed.

## Manuscript sources

From `manuscript/main/`, run `pdflatex main.tex`; from `manuscript/supplement/`, run `pdflatex supplement.tex`. Alternate builds until cross-references resolve. The Copernicus LaTeX class files (`copernicus.cls`, `.cfg`, `.bst`) must be available in the TeX installation; they are not bundled. The final figure paths are already set. See `VALIDATION.md` for the checks actually performed and their limits.

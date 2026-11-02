# Manuscript analysis

`common.py` resolves the data archive and the default evaluation selection. Every test-performance calculation excludes RGI60-14.17287.

| Script | Purpose |
|---|---|
| `make_membership.py` | Write final evaluated membership and split-format selection |
| `recalculate_core.py` | Point metrics, paired intervals, band/realization/geometry summaries |
| `recalculate_transfer_coordinates.py` | Regional-transfer and matched-coordinate metrics and intervals |
| `build_tables.py` | Build the six result tables (Tables 5–10), rounded from unrounded values |
| `run_production_stages.py` | Filter the frozen keyed cache and run numerical stages using the requested output location |
| `make_figures.py` | Assemble all 13 numbered manuscript figures |
| `supplement_figures.py` | Generate Figures S4–S8 from evaluated summaries |
| `verify_results.py` | Check data identities, targets, masks, denominators, tables, manuscript values, and figures |
| `figures/` | Dedicated sources for Figures 2, 3, 5, S1, and S3 |

Figures 1 and 4 use the workflow's map/bootstrap plotting scripts. Figure S2 uses the archived geometry-selected artwork; its source arrays and selection remain in the data archive. Archived full-population calculations are used internally as numerical references. Final exported evidence contains only the manuscript evaluation. Run the full sequence using the repository's `reproduce.py`.

# Scientific workflow stages

Run `python workflow/workflow.py --root <data_archive> steps` for the retained commands. The default `config/workflow.json` uses the manuscript evaluation split (RGI60-14.17287 excluded).

`src/void_fill/01_datacube/` builds/validates the exact 30 m reference lattice, DEM, terrain, and glacier raster. `02_voids/` remaps frozen masks, builds metadata, and validates inputs. `03_baselines/` implements constant and hypsometric local means. `04_model/` prepares rows and constructs realization-specific context. `05_results/` builds common prediction support and calculates pixel/glacier errors and spatial-block intervals. `06_figures/` supplies the study map and bootstrap contrast plot; the other final figure sources are under `analysis/manuscript/figures/`.

`workflow_support.py` resolves paths, settings, split inventories, and calculation provenance. The workflow is executed directly, without installation as a Python package. Fixed-setting model fitting is in `analysis/model/`. Full commands and output conventions are in the repository's `REPRODUCE.md`.

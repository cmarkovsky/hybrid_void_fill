# Manuscript metadata

| File | Content |
|---|---|
| `splits/training_split.json` | Unchanged fitting assignments: 8,179 train, 1,751 validation, 1,755 assigned test glaciers |
| `splits/evaluation_split.json` | Default workflow selection: training/validation assignments plus 1,754 evaluated test glaciers; RGI60-14.17287 removed |
| `evaluation_membership.csv` | Assigned test membership, region, realization/target counts, and explicit evaluated flag |
| `evaluation_exclusions.json` | Post hoc manuscript evaluation exclusion and its scope |
| `excluded_glaciers.json` | The 51 boundary/surge population exclusions preceding fitting |
| `realization_inventory.csv` | All five realizations and valid-reference pixel denominators for the original fitted experiment |
| `glacier_attributes.csv` | RGI-derived static attributes used by the primary model |
| `glacier_geometry.csv` | Evaluated test-glacier centroids for spatial summaries |
| `block_assignments.csv` | Evaluated test-glacier blocks at 50, 100, and 200 km |
| `DATA_SOURCES.csv` | Required archive assets, hashes, citations, packaged paths, and grid/source status |

Regions are `whim` (Western Himalaya) and `ehim` (Eastern Himalaya). Prediction keys are `(rgiid, realization, pixel_flat_index)`, where the pixel index is row-major within a glacier chip. Errors are prediction minus reference, in m a⁻¹. Glacier-specific errors use all valid reference pixels as the denominator. Integrated volume-rate error equals specific error × valid reference area (valid pixel count × 900 m²), in m³ a⁻¹. The full realization inventory and fitting split preserve experiment identity; evaluated summaries use the evaluation split.

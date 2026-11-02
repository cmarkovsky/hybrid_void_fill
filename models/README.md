# Stored manuscript models

| Path | Model |
|---|---|
| `hybrid/model.json` | Primary hybrid: all 8,179 training glaciers, 25 predictors, 172 trees |
| `regional_transfer/source_whim.json` | Western-source hybrid: 1,000 source glaciers |
| `regional_transfer/source_ehim.json` | Eastern-source hybrid: 1,000 source glaciers |
| `coordinate_ablation/coordinates_off/model.json` | Matched capped-sample hybrid with 25 predictors |
| `coordinate_ablation/coordinates_on/model.json` | Same training sample with projected easting/northing added (27 predictors) |

The boosters are unchanged manuscript artifacts. `hybrid/FEATURE_SCHEMA.json`, `training_config.json`, and `feature_importance_gain.csv` describe the primary fit. The two `design.json` files and coordinate `feature_schemas.json` describe the supplementary fits, source lists, weighting, and contrasts. Default performance evaluation excludes RGI60-14.17287; the fitting samples are unchanged. Reproduction commands are in `REPRODUCE.md`.

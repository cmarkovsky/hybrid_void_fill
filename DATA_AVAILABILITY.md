# Data and manuscript inputs

This repository contains code, five saved models, final metadata, numerical summaries, tables, figures, and LaTeX sources. The approximately 12 GB data archive is stored separately as `data_archive/` and is required to reproduce the results. The public data-deposit URL/DOI has not been supplied. No upload or deposition has been performed by this cleanup.

The archive contains the reference dh/dt raster, Copernicus DEM tiles, RGI/study outlines, HIMAP regions, lake/surge inventories, aligned cube, frozen/remapped void chips, model rows, development arrays, saved baseline and model predictions, bootstrap inputs, and original experiment records. Raw-to-cube raster intermediates are rebuilt by the datacube stages; the supplied cube and prepared data are available directly.

`config/data_root.example.json` maps inputs relative to `--data-root`. Historical filenames in that map are exact archive identities and are intentionally preserved so the existing archive works. Repository scripts and manuscript-facing names use the final descriptive terminology. No local username or machine path is required.

| Input | Identifier recorded with the manuscript |
|---|---|
| Shean elevation-change mosaic, 2000–2018 | https://doi.org/10.5281/zenodo.3872696 |
| Copernicus GLO-30 DEM | https://doi.org/10.5069/G9028PQB |
| RGI 6.0 glacier outlines | https://doi.org/10.7265/4M1F-GD79 |



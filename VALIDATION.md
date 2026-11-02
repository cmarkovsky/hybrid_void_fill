# Validation of the manuscript repository

Checks performed on 2 November 2026 in the supplied Python 3.12 scientific environment:

| Check | Result |
|---|---|
| Focused scientific/manuscript tests | 8 passed in two test files |
| Script entry points | All retained command-line entry points accept `--help` from a different working directory |
| Complete one-command reproduction | Passed, including all 13 figures and 19 data-backed checks |
| Default evaluation | 1,754 test glaciers; 8,770 glacier-realizations; 2,540,722 targets per method; RGI60-14.17287 absent |
| Primary hybrid fit | Byte-identical reproduction of the stored 172-tree booster on all 11,792,199 training targets |
| Supplementary fits | All four regional-transfer/coordinate boosters reproduced byte for byte |
| Default hybrid predictions | Keys, reference values, hypsometric offsets, denominators, and predictions exactly equal saved predictions after the exclusion |
| Main interval reference checks | Maximum endpoint difference 3.45×10⁻¹⁶ |
| Coordinate interval reference checks | Maximum endpoint difference 2.28×10⁻¹³ |
| Numerical stage outputs | All 28 CSV tables agree with the original manuscript-stage outputs to numerical precision |
| Result tables | Six LaTeX tables rebuild identically; their unrounded records/Markdown headings use Tables 5–10 |
| Data-backed verifier | All checks pass: frozen identities, membership, support, masks, denominators, area scaling, manuscript values, and figure hashes |
| Figure reproduction | All 13 numbered figures generated successfully; scientific values/populations agree. Twelve PNGs are byte-identical to the supplied artwork; Figure 4 has rendering/layout differences. Original manuscript artwork is retained. PDF byte identity is not required because creation metadata can differ. |
| Manuscript compilation | Main and supplement compile, with zero undefined-reference warnings after alternating builds |
| Input coverage | Passed for all 11,684 glaciers in the default split (58,420 glacier-realizations) |
| Rebuilt model-row sample | All scientific arrays match the corresponding archived training shard exactly |
| Datacube grid | Raw-input grid definition runs and matches the recorded exact 30 m lattice (67,989 columns × 32,127 rows) |

The main and supplementary manuscript sources were compiled using the locally available Copernicus class. The class is not redistributed. Existing minor overfull-line warnings remain from the supplied manuscript layout.

The complete raw-raster alignment, terrain/cube rebuild, every chip remapping, and a new hyperparameter search were not repeated. The complete data archive and the recorded scientific dependencies are required for data-backed runs. Package installation on a clean machine and a different operating system was not tested. The saved masks, fitting/development assignments, and primary model selection are preserved; the default reproduction recalculates results rather than selecting a new model.

Full execution logs and scratch outputs are kept outside the GitHub folder. `reproduce.py` writes its own verification report with each new reproduction.

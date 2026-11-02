#!/usr/bin/env python3
"""Re-run the production results stages 02-05 and figure stages on the manuscript subset.

No model is loaded and nothing is re-predicted. The frozen keyed prediction cache (1,755 test
glaciers, SHA256 a4f6be88...) is filtered to the evaluated glaciers of
data/metadata/evaluation_membership.csv, keeping every remaining row unchanged.
The unchanged production stage scripts then run with config/workflow.json:

  02 pixel-stats, 03 glacier-stats (population = derived evaluation membership), 04 comparison,
  05 bootstrap (50/100/200 km, 2,000 draws, seed 20260917, weighted-histogram pixel NMAD),
  figure4 (all-region and row-scaled x limits), figure5, and figure1 on the full production split
  (study-area map, not a test-performance figure).

    python run_production_stages.py --data-root <data_archive> --out <data_archive>/derived/manuscript_excluding_RGI60_14_17287
"""
from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import Paths, Selection, base_parser, sha256, write_json

FROZEN_CACHE_SHA256 = "a4f6be88c96c5f30fa0f1a04e94b764a271f602b2b9fe5949d64d87a9b66a3d3"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--skip-figure1", action="store_true", help="Figure 1 needs Cartopy and the raw outlines")
    a = ap.parse_args()
    paths, sel = Paths(a.data_root, a.repo), Selection(a.repo)
    src = paths["reference_predictions"]
    if sha256(src) != FROZEN_CACHE_SHA256:
        raise SystemExit(f"{src} is not the frozen keyed cache")
    out = a.out.expanduser().resolve()  # stage scripts resolve relative paths against --root
    (out / "cache").mkdir(parents=True, exist_ok=True)
    cache = out / "cache/common_xy_point_predictions.parquet"
    t = pq.read_table(src)
    keep = pa.compute.invert(pa.compute.is_in(t["rgiid"], value_set=pa.array(sel.excluded)))
    sub = t.filter(keep)
    pq.write_table(sub, cache)
    record = dict(source=str(src), source_sha256=FROZEN_CACHE_SHA256, rows_in=t.num_rows, rows_out=sub.num_rows,
                  excluded=sel.excluded, output_sha256=sha256(cache))
    df = sub.select(["method", "rgiid"]).to_pandas()
    record["glaciers_out"] = int(df.rgiid.nunique())
    record["targets_per_method"] = df.groupby("method").size().to_dict()
    if record["glaciers_out"] != sel.expected["test_glaciers"]["all"] or set(record["targets_per_method"].values()) != {sel.expected["targets_per_method"]["all"]}:
        raise SystemExit(f"filtered cache does not match the expected population: {record}")
    write_json(out / "cache/FILTER_RECORD.json", record)

    wf = a.repo / "workflow/workflow.py"
    profile = out / "workflow_config.json"
    config = json.loads((a.repo / "config/workflow.json").read_text())
    for key, suffix in {"cache": "cache/common_xy_point_predictions.parquet", "pixel_summaries": "pixel_summaries",
                        "glacier_metrics": "glacier_wide_metrics", "comparison": "glacier_wide_comparison",
                        "bootstrap": "spatial_block_bootstrap", "figures": "figures"}.items():
        config["paths"][key] = str(out / suffix)
    config["paths"]["split"] = str(a.repo / "data/metadata/splits/evaluation_split.json")
    write_json(profile, config)
    membership = a.repo / "data/metadata/splits/evaluation_split.json"
    base = [sys.executable, "-B", str(wf), "--root", str(paths.root), "--config", str(profile)]
    runs = [["pixel-stats"], ["glacier-stats", "--", "--split", str(membership)], ["comparison"], ["bootstrap"],
            ["figure4"], ["figure4", "--", "--xlim-scope", "row", "--output-dir", str(out / "figures_row_scaled")],
            ["figure5"]]
    if not a.skip_figure1:
        # Figure 1 maps the whole study population (production split, all groups); the evaluation
        # exclusion does not apply to it.
        runs.append(["figure1", "--", "--split", str(a.repo / "data/metadata/splits/training_split.json")])
    log = []
    for r in runs:
        cmd = base + r
        print(" ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd=paths.root, capture_output=True, text=True)
        log.append(dict(command=cmd, exit_code=p.returncode, stdout_tail=p.stdout[-4000:], stderr_tail=p.stderr[-4000:]))
        write_json(out / "STAGE_LOG.json", log)
        if p.returncode:
            print(p.stdout[-3000:], p.stderr[-3000:])
            raise SystemExit(f"stage failed: {r}")
    print("stages complete")


if __name__ == "__main__":
    main()

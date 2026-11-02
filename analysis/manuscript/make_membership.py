#!/usr/bin/env python3
"""Write the derived evaluation-membership files from the production split and the exclusions.

The production split is not modified. Two derived files are written:
  * evaluation_membership.csv: one row per assigned test glacier with
    region, realization count, target counts and an `evaluated` flag;
  * splits/evaluation_split.json: the production split with the
    excluded entries removed. It exists only because pipeline stage 03 reads its population
    from a split-format file. It is an evaluation-membership input, not a training split.

    python make_membership.py --data-root <data_archive> --out data/metadata
"""
from pathlib import Path

import pandas as pd

from common import Paths, Selection, base_parser, write_json


def main() -> None:
    a = base_parser(__doc__).parse_args()
    paths, sel = Paths(a.data_root, a.repo), Selection(a.repo)
    inv = pd.read_csv(a.repo / "data/metadata/realization_inventory.csv")
    test = inv[inv.group.eq("test")]
    res = pd.read_parquet(paths["reference_residuals"], columns=["method", "rgiid", "region"])
    targets = res[res.method.eq("constant")].groupby("rgiid").size().rename("targets_per_method")
    m = (test.groupby(["rgiid", "region"]).agg(n_realizations=("realization_index", "nunique"),
                                               n_valid_glacier_pixels=("n_valid_glacier_pixels", "first"))
         .reset_index().merge(targets, on="rgiid", how="left", validate="one_to_one"))
    m.insert(1, "assigned_group", "test")
    m["evaluated"] = ~m.rgiid.isin(sel.excluded)
    m["exclusion_record"] = m.rgiid.map(lambda g: "data/metadata/evaluation_exclusions.json" if g in sel.excluded else "")
    if sorted(m.rgiid) != sel.assigned_test:
        raise ValueError("inventory test glaciers differ from the production split")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "evaluation_membership.csv", index=False)
    write_json(out / "splits/evaluation_split.json",
               {g: grp for g, grp in sel.split.items() if g not in sel.excluded})
    e = m[m.evaluated]
    got = {"test_glaciers": {**e.groupby("region").size().to_dict(), "all": len(e)},
           "glacier_realizations": int(e.n_realizations.sum()),
           "targets_per_method": {**e.groupby("region").targets_per_method.sum().astype(int).to_dict(),
                                  "all": int(e.targets_per_method.sum())}}
    exp = sel.expected
    if got != exp:
        raise SystemExit(f"membership counts {got} differ from expected {exp}")
    print("membership matches expected counts:", got)


if __name__ == "__main__":
    main()

"""Shared paths and evaluation selection for the manuscript-subset analyses.

Every script takes --data-root (the extracted data_archive/) and, optionally, --repo (this
repository). Input locations come from config/data_root.example.json's "paths" block, so no
machine-specific path is written into code. The evaluation selection is read from
config/evaluation.json and data/metadata/evaluation_exclusions.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METHODS = ["constant", "hypsometric", "xgb"]


def norm_method(x) -> str:
    x = str(x).lower()
    return "xgb" if x in ("hybrid", "xgboost", "xgb", "accepted_combined") else x


def nmad(x) -> float:
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def interval(x):
    return np.percentile(x, [2.5, 97.5])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Paths:
    """Resolve named inputs under the data root using the repository's example layout."""

    def __init__(self, data_root: Path, repo: Path = REPO):
        self.root = Path(data_root).expanduser().resolve()
        self.repo = Path(repo).resolve()
        self.layout = json.loads((self.repo / "config/data_root.example.json").read_text())["paths"]

    def __getitem__(self, name: str) -> Path:
        return self.root / self.layout[name]


class Selection:
    """The manuscript evaluation selection: production test group minus listed exclusions."""

    def __init__(self, repo: Path = REPO):
        self.repo = Path(repo).resolve()
        self.config = json.loads((self.repo / "config/evaluation.json").read_text())
        record = json.loads((self.repo / self.config["exclusions"]).read_text())
        self.excluded = sorted(e["rgiid"] for e in record["exclusions"])
        self.split = json.loads((self.repo / self.config["production_split"]).read_text())
        self.assigned_test = sorted(g for g, grp in self.split.items() if grp == "test")
        missing = set(self.excluded) - set(self.assigned_test)
        if missing:
            raise ValueError(f"exclusions not in the assigned test group: {sorted(missing)}")
        self.evaluated = sorted(set(self.assigned_test) - set(self.excluded))
        self.expected = self.config["expected"]

    def keep(self, ids) -> np.ndarray:
        """Boolean mask selecting rows whose glacier ID is evaluated."""
        return ~pd.Series(ids).astype(str).isin(self.excluded).to_numpy()


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-root", type=Path, required=True, help="extracted data_archive/ directory")
    ap.add_argument("--repo", type=Path, default=REPO, help="repository root (default: this checkout)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    return ap


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def finalize_evaluation_outputs(out):
    """Write only the selected population after archived-reference checks have passed."""
    excluded = Selection().excluded
    for path in Path(out).glob('*.csv'):
        table = pd.read_csv(path)
        for flag in ['excluded', 'evaluated']:
            if flag in table.columns:
                table = table[table[flag].astype(str).str.lower().eq('true')]
        if 'rgiid' in table.columns:
            table = table[~table.rgiid.isin(excluded)]
        table.to_csv(path, index=False)

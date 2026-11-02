"""Checks of the evaluated population, reported tables, and stored artifact identities."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis/manuscript"))
from common import Selection


class ManuscriptResults(unittest.TestCase):
    def test_default_membership_and_counts(self):
        selection = Selection()
        self.assertEqual(selection.excluded, ["RGI60-14.17287"])
        self.assertEqual(len(selection.evaluated), 1754)
        split = json.loads((REPO / "data/metadata/splits/evaluation_split.json").read_text())
        self.assertNotIn("RGI60-14.17287", split)
        membership = pd.read_csv(REPO / "data/metadata/evaluation_membership.csv")
        evaluated = membership[membership.evaluated]
        self.assertEqual(evaluated.groupby("region").size().to_dict(), {"whim": 1422, "ehim": 332})
        self.assertEqual(int(evaluated.targets_per_method.sum()), 2540722)
        self.assertEqual(int(evaluated.n_realizations.sum()), 8770)

    def test_tables_rebuild_from_committed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run([sys.executable, str(REPO / "analysis/manuscript/build_tables.py"),
                "--evidence", str(REPO / "results/manuscript/evidence"), "--out", temporary], check=True, capture_output=True)
            for table in (REPO / "results/manuscript/tables").glob("tab_*.tex"):
                self.assertEqual((Path(temporary) / "tables" / table.name).read_bytes(), table.read_bytes(), table.name)
                self.assertEqual((REPO / "manuscript/main/tables" / table.name).read_bytes(), table.read_bytes(), table.name)

    def test_figures_match_recorded_hashes(self):
        figures = json.loads((REPO / "figures/FIGURE_SOURCES.json").read_text())
        self.assertEqual(len(figures), 26)
        for path, record in figures.items():
            self.assertEqual(hashlib.sha256((REPO / "figures" / path).read_bytes()).hexdigest(), record["sha256"], path)

    def test_saved_hybrid_identity(self):
        config = json.loads((REPO / "models/hybrid/training_config.json").read_text())
        self.assertEqual(hashlib.sha256((REPO / "models/hybrid/model.json").read_bytes()).hexdigest(), config["training"]["model_sha256"])
        self.assertEqual(config["training"]["rounds"], 172)
        self.assertEqual(len(config["features"]), 25)


if __name__ == "__main__": unittest.main()

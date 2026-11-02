"""Small checks of masking, neighbour selection, and glacier-balanced fitting."""
from pathlib import Path
import importlib.util
import sys
import unittest
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis/model"))
from model_data import context, glacier_weights


class ScientificInvariants(unittest.TestCase):
    def test_equal_glacier_and_realization_weights(self):
        glaciers = np.array([0, 0, 0, 1, 1, 1, 1])
        realizations = np.array([0, 0, 1, 0, 1, 1, 1])
        weights = glacier_weights(glaciers, realizations)
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(weights[glaciers == 0].sum(), weights[glaciers == 1].sum())
        self.assertAlmostEqual(weights[(glaciers == 1) & (realizations == 0)].sum(), weights[(glaciers == 1) & (realizations == 1)].sum())

    def test_strict_radius_and_sparse_sources(self):
        coordinates = np.array([[900., 0.], [540., 720.], [870., 0.], [0., 30.], [30., 0.]])
        residual = np.array([5., 5., 1., 2., 3.])
        _, count, local = context.local_summaries(coordinates, residual, np.arange(5), np.zeros((1, 2)))
        self.assertEqual(count[0], 3)
        self.assertAlmostEqual(local[0], np.average([1, 2, 3], weights=np.exp(-np.array([870, 30, 30]) / 300)))
        _, count, local = context.local_summaries(coordinates[[0, 3, 4]], residual[[0, 3, 4]], np.array([0, 3, 4]), np.zeros((1, 2)))
        self.assertEqual(count[0], 2)
        self.assertTrue(np.isnan(local[0]))

    def test_ties_are_ordered_by_raster_index(self):
        angle = np.arange(20) * 2 * np.pi / 20
        inner = np.column_stack([100 * np.cos(angle), 100 * np.sin(angle)])
        angle = np.arange(40) * 2 * np.pi / 40
        ring = np.column_stack([500 * np.cos(angle), 500 * np.sin(angle)])
        coordinates = np.vstack([inner, ring])
        flat = np.r_[np.arange(20), 1000 - np.arange(40)]
        residual = np.r_[np.zeros(20), np.arange(40, dtype=float)]
        query = np.zeros((1, 2))
        _, count, value = context.local_summaries(coordinates, residual, flat, query)
        chosen = np.argsort(flat[20:])[:12]
        expected = residual[20 + chosen].sum() * np.exp(-500 / 300) / (20 * np.exp(-100 / 300) + 12 * np.exp(-500 / 300))
        self.assertEqual(count[0], 32)
        self.assertAlmostEqual(value[0], expected, places=10)
        order = np.random.default_rng(3).permutation(60)
        np.testing.assert_allclose(context.local_summaries(coordinates[order], residual[order], flat[order], query)[2], value, atol=1e-10)

    def test_hidden_targets_cannot_change_predictors(self):
        sys.path.insert(0, str(REPO / "workflow"))
        spec = importlib.util.spec_from_file_location("hypsometric", REPO / "workflow/src/void_fill/03_baselines/02_hypsometric_local_mean.py")
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        rows, columns = np.indices((6, 6))
        elevation = 4000 + 100 * rows + 5 * columns
        glacier = np.ones((6, 6), bool)
        void = np.zeros((6, 6), bool)
        void[1:3, 1:4] = True
        observed = -1 + rows * .1 + columns * .03
        targets = np.flatnonzero(void.ravel())
        def features(values):
            return context.context_features(elevation, glacier, void, context.mask_observations(values, glacier, void),
                500000 + np.arange(6) * 30., 3300000 - np.arange(6) * 30., targets, 4000., 4525., baseline)[0]
        changed = observed.copy()
        changed[void] += 1000
        np.testing.assert_array_equal(features(observed), features(changed))


if __name__ == "__main__": unittest.main()

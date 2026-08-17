from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HAS_BACKENDS = all(
    importlib.util.find_spec(name) is not None
    for name in ("xgboost", "catboost")
)

if HAS_BACKENDS:
    from src.calibration import select_stable_weights
    from src.models import MODEL_ORDER


@unittest.skipUnless(HAS_BACKENDS, "GBDT training backends are not installed")
class FourModelEnsembleTests(unittest.TestCase):
    def test_simplex_search_supports_four_models_and_recent_fold_weighting(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.float64)
        earlier = {
            "y_true": y,
            "predictions": {
                "xgb": np.asarray([0.2, 0.3, 0.7, 0.8]),
                "cat": np.asarray([0.1, 0.2, 0.8, 0.9]),
                "resnet": np.asarray([0.3, 0.4, 0.6, 0.7]),
                "ft_transformer": np.asarray([0.4, 0.4, 0.6, 0.6]),
            },
        }
        recent = {
            "y_true": y,
            "predictions": {
                "xgb": np.asarray([0.3, 0.4, 0.6, 0.7]),
                "cat": np.asarray([0.4, 0.4, 0.6, 0.6]),
                "resnet": np.asarray([0.1, 0.1, 0.9, 0.9]),
                "ft_transformer": np.asarray([0.2, 0.2, 0.8, 0.8]),
            },
        }
        weights, report = select_stable_weights(
            [earlier, recent], fold_importance=(0.2, 0.8), step=0.05
        )
        self.assertEqual(tuple(MODEL_ORDER), ("xgb", "cat", "resnet", "ft_transformer"))
        self.assertEqual(weights.shape, (4,))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreaterEqual(float(weights[2] + weights[3]), 0.5)
        self.assertEqual(report["fold_importance"], [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()

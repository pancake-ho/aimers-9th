from __future__ import annotations

import unittest

import numpy as np

from src.calibration import (
    fit_prequential_logit_calibrator,
    select_stable_weights,
)
from src.models import MODEL_ORDER


class RegimeEnsembleTests(unittest.TestCase):
    def test_two_model_simplex_uses_three_forward_folds(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.float64)
        folds = [
            {
                "y_true": y,
                "predictions": {
                    "cat": np.asarray([0.30, 0.40, 0.60, 0.70]),
                    "tabm": np.asarray([0.20, 0.30, 0.70, 0.80]),
                },
            },
            {
                "y_true": y,
                "predictions": {
                    "cat": np.asarray([0.25, 0.35, 0.65, 0.75]),
                    "tabm": np.asarray([0.15, 0.25, 0.75, 0.85]),
                },
            },
            {
                "y_true": y,
                "predictions": {
                    "cat": np.asarray([0.35, 0.40, 0.60, 0.65]),
                    "tabm": np.asarray([0.10, 0.20, 0.80, 0.90]),
                },
            },
        ]
        weights, report = select_stable_weights(
            folds,
            fold_importance=(0.15, 0.35, 0.50),
            step=0.025,
            model_order=MODEL_ORDER,
        )
        self.assertEqual(tuple(MODEL_ORDER), ("cat", "tabm"))
        self.assertEqual(weights.shape, (2,))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(float(weights[1]), 0.0)
        self.assertEqual(report["fold_importance"], [0.15, 0.35, 0.5])

    def test_logit_calibration_requires_forward_transfer(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.float64)
        earlier = {
            "y_true": y,
            "predictions": {
                "cat": np.asarray([0.35, 0.45, 0.75, 0.85]),
                "tabm": np.asarray([0.30, 0.40, 0.70, 0.80]),
            },
        }
        recent = {
            "y_true": y,
            "predictions": {
                "cat": np.asarray([0.30, 0.40, 0.70, 0.80]),
                "tabm": np.asarray([0.25, 0.35, 0.65, 0.75]),
            },
        }
        report = fit_prequential_logit_calibrator(
            earlier,
            recent,
            weights=(0.5, 0.5),
            model_order=MODEL_ORDER,
        )
        self.assertEqual(report["method"], "logit_intercept")
        self.assertIn("recent_transferred_brier", report)
        self.assertTrue(np.isfinite(report["intercept"]))


if __name__ == "__main__":
    unittest.main()

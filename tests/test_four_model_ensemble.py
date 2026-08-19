from __future__ import annotations

import unittest

import numpy as np

from src.calibration import (
    fit_prequential_logit_calibrator,
    select_stable_weights,
)
from src.models import MODEL_ORDER


class RegimeEnsembleTests(unittest.TestCase):
    def test_four_model_simplex_protects_recent_folds(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.float64)
        folds = [
            {
                "validation_label": "2023",
                "y_true": y,
                "predictions": {
                    "xgb": np.asarray([0.30, 0.40, 0.60, 0.70]),
                    "cat": np.asarray([0.25, 0.35, 0.65, 0.75]),
                    "resnet": np.asarray([0.20, 0.30, 0.70, 0.80]),
                    "ft_transformer": np.asarray([0.28, 0.38, 0.62, 0.72]),
                },
            },
            {
                "validation_label": "2024",
                "y_true": y,
                "predictions": {
                    "xgb": np.asarray([0.25, 0.35, 0.65, 0.75]),
                    "cat": np.asarray([0.20, 0.30, 0.70, 0.80]),
                    "resnet": np.asarray([0.15, 0.25, 0.75, 0.85]),
                    "ft_transformer": np.asarray([0.22, 0.32, 0.68, 0.78]),
                },
            },
            {
                "validation_label": "2024_late_abs",
                "y_true": y,
                "predictions": {
                    "xgb": np.asarray([0.35, 0.40, 0.60, 0.65]),
                    "cat": np.asarray([0.25, 0.35, 0.65, 0.75]),
                    "resnet": np.asarray([0.10, 0.20, 0.80, 0.90]),
                    "ft_transformer": np.asarray([0.20, 0.30, 0.70, 0.80]),
                },
            },
        ]
        weights, report = select_stable_weights(
            folds,
            fold_importance=(0.15, 0.35, 0.50),
            step=0.025,
            model_order=MODEL_ORDER,
            protected_fold_labels=("2024", "2024_late_abs"),
        )
        self.assertEqual(
            tuple(MODEL_ORDER),
            ("xgb", "cat", "resnet", "ft_transformer"),
        )
        self.assertEqual(weights.shape, (4,))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(report["fold_importance"], [0.15, 0.35, 0.5])
        self.assertGreater(report["feasible_candidates"], 0)
        self.assertTrue(
            all(
                gain >= -1e-12
                for gain in report["protected_fold_gain_vs_reference"].values()
            )
        )

    def test_logit_calibration_requires_forward_transfer(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.float64)
        earlier = {
            "y_true": y,
            "predictions": {
                "xgb": np.asarray([0.35, 0.45, 0.75, 0.85]),
                "cat": np.asarray([0.30, 0.40, 0.70, 0.80]),
                "resnet": np.asarray([0.32, 0.42, 0.72, 0.82]),
                "ft_transformer": np.asarray([0.28, 0.38, 0.68, 0.78]),
            },
        }
        recent = {
            "y_true": y,
            "predictions": {
                "xgb": np.asarray([0.30, 0.40, 0.70, 0.80]),
                "cat": np.asarray([0.25, 0.35, 0.65, 0.75]),
                "resnet": np.asarray([0.27, 0.37, 0.67, 0.77]),
                "ft_transformer": np.asarray([0.23, 0.33, 0.63, 0.73]),
            },
        }
        report = fit_prequential_logit_calibrator(
            earlier,
            recent,
            weights=(0.25, 0.25, 0.25, 0.25),
            model_order=MODEL_ORDER,
        )
        self.assertEqual(report["method"], "logit_intercept")
        self.assertIn("recent_transferred_brier", report)
        self.assertTrue(np.isfinite(report["intercept"]))


if __name__ == "__main__":
    unittest.main()

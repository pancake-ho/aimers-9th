from __future__ import annotations

import unittest

import numpy as np

from src.calibration import (
    fit_abs_regime_logit_calibrator,
    select_stable_weights,
)
from src.models import MODEL_ORDER


class RegimeEnsembleTests(
    unittest.TestCase
):
    @staticmethod
    def _fold_predictions(
        values,
    ):
        return {
            "xgb": np.asarray(
                values[0],
                dtype=np.float64,
            ),
            "lgb": np.asarray(
                values[1],
                dtype=np.float64,
            ),
            "cat": np.asarray(
                values[2],
                dtype=np.float64,
            ),
            "resnet": np.asarray(
                values[3],
                dtype=np.float64,
            ),
            "ft_transformer": np.asarray(
                values[4],
                dtype=np.float64,
            ),
        }

    def test_five_model_simplex_protects_recent_folds(
        self,
    ):
        y = np.asarray(
            [0, 0, 1, 1],
            dtype=np.float64,
        )

        folds = [
            {
                "validation_label": "2023",
                "y_true": y,
                "predictions": (
                    self._fold_predictions(
                        [
                            [0.30, 0.40, 0.60, 0.70],
                            [0.28, 0.38, 0.62, 0.72],
                            [0.25, 0.35, 0.65, 0.75],
                            [0.20, 0.30, 0.70, 0.80],
                            [0.27, 0.37, 0.63, 0.73],
                        ]
                    )
                ),
            },
            {
                "validation_label": "2024",
                "y_true": y,
                "predictions": (
                    self._fold_predictions(
                        [
                            [0.25, 0.35, 0.65, 0.75],
                            [0.22, 0.32, 0.68, 0.78],
                            [0.20, 0.30, 0.70, 0.80],
                            [0.15, 0.25, 0.75, 0.85],
                            [0.21, 0.31, 0.69, 0.79],
                        ]
                    )
                ),
            },
            {
                "validation_label": (
                    "2024_late_abs"
                ),
                "y_true": y,
                "predictions": (
                    self._fold_predictions(
                        [
                            [0.35, 0.40, 0.60, 0.65],
                            [0.28, 0.36, 0.64, 0.72],
                            [0.25, 0.35, 0.65, 0.75],
                            [0.10, 0.20, 0.80, 0.90],
                            [0.20, 0.30, 0.70, 0.80],
                        ]
                    )
                ),
            },
        ]

        weights, report = (
            select_stable_weights(
                folds,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                step=0.025,
                model_order=MODEL_ORDER,
                protected_fold_labels=(
                    "2024",
                    "2024_late_abs",
                ),
            )
        )

        self.assertEqual(
            tuple(MODEL_ORDER),
            (
                "xgb",
                "lgb",
                "cat",
                "resnet",
                "ft_transformer",
            ),
        )

        self.assertEqual(
            weights.shape,
            (5,),
        )

        self.assertAlmostEqual(
            float(weights.sum()),
            1.0,
        )

        self.assertEqual(
            report["fold_importance"],
            [
                0.05,
                0.45,
                0.50,
            ],
        )

        self.assertGreater(
            report[
                "feasible_candidates"
            ],
            0,
        )

        self.assertTrue(
            all(
                gain >= -1e-12
                for gain
                in report[
                    "protected_fold_gain_vs_reference"
                ].values()
            )
        )

    def test_abs_calibration_contract(
        self,
    ):
        n_full = 2000
        n_late = 800

        y_full = np.tile(
            np.asarray(
                [0.0, 1.0]
            ),
            n_full // 2,
        )

        y_late = y_full[
            -n_late:
        ].copy()

        months = np.concatenate(
            [
                np.full(
                    1200,
                    4,
                    dtype=np.int8,
                ),
                np.full(
                    800,
                    8,
                    dtype=np.int8,
                ),
            ]
        )

        def make_predictions(
            y,
            offsets,
        ):
            result = {}

            for name, offset in zip(
                MODEL_ORDER,
                offsets,
            ):
                base = np.where(
                    y > 0.5,
                    0.68,
                    0.32,
                )

                result[name] = np.clip(
                    base + offset,
                    0.01,
                    0.99,
                )

            return result

        full_fold = {
            "y_true": y_full,
            "valid_months": months,
            "predictions": (
                make_predictions(
                    y_full,
                    (
                        0.030,
                        0.020,
                        0.025,
                        0.015,
                        0.010,
                    ),
                )
            ),
        }

        late_fold = {
            "y_true": y_late,
            "predictions": (
                make_predictions(
                    y_late,
                    (
                        0.025,
                        0.015,
                        0.020,
                        0.010,
                        0.005,
                    ),
                )
            ),
        }

        report = (
            fit_abs_regime_logit_calibrator(
                full_2024_fold=full_fold,
                late_2024_fold=late_fold,
                weights=(
                    0.20,
                    0.20,
                    0.20,
                    0.20,
                    0.20,
                ),
                model_order=MODEL_ORDER,
                early_month_max=6,
            )
        )

        self.assertEqual(
            report["method"],
            "abs_regime_logit_intercept",
        )

        self.assertEqual(
            report["early_rows"],
            1200,
        )

        self.assertTrue(
            np.isfinite(
                report[
                    "full_2024_intercept"
                ]
            )
        )

        self.assertTrue(
            np.isfinite(
                report[
                    "late_transferred_brier"
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
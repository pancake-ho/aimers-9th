from __future__ import annotations

import unittest

import numpy as np

from src.calibration import (
    select_calibration_aware_shrunk_weights,
)


class CalibrationShrinkageTests(
    unittest.TestCase
):
    @staticmethod
    def _fold(
        label: str,
        *,
        n_rows: int = 2400,
    ):
        y = np.tile(
            np.asarray(
                [0.0, 1.0],
                dtype=np.float64,
            ),
            n_rows // 2,
        )

        # Deliberately biased upward so that a
        # negative logit intercept has a real,
        # transferable Brier benefit.
        prediction = np.where(
            y > 0.5,
            0.73,
            0.43,
        ).astype(
            np.float64
        )

        months = np.resize(
            np.arange(
                1,
                10,
                dtype=np.int8,
            ),
            n_rows,
        )

        return {
            "validation_label": label,
            "valid_season": (
                2023
                if label == "2023"
                else 2024
            ),
            "y_true": y,
            "valid_months": months,
            "predictions": {
                "xgb": prediction.copy(),
                "cat": prediction.copy(),
            },
        }

    def test_calibration_aware_shrinkage_contract(
        self,
    ) -> None:
        folds = [
            self._fold("2023"),
            self._fold("2024"),
            self._fold(
                "2024_late_abs"
            ),
        ]

        (
            weights,
            report,
            calibrator,
        ) = (
            select_calibration_aware_shrunk_weights(
                folds,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                step=0.50,
                model_order=(
                    "xgb",
                    "cat",
                ),
                protected_fold_labels=(
                    "2024",
                    "2024_late_abs",
                ),
                non_degradation_tolerance=0.0,
                reference_model="xgb",
                alpha_grid=(
                    0.0,
                    0.5,
                    1.0,
                ),
                early_month_max=6,
                maximum_2023_regression_vs_reference=0.0,
                minimum_calibration_transfer_gain=0.0,
            )
        )

        self.assertTrue(
            report[
                "selection_passed"
            ]
        )

        self.assertTrue(
            calibrator[
                "accepted"
            ]
        )

        self.assertIn(
            report[
                "selected_alpha"
            ],
            (
                0.0,
                0.5,
                1.0,
            ),
        )

        self.assertAlmostEqual(
            float(
                weights.sum()
            ),
            1.0,
            places=12,
        )

        self.assertTrue(
            np.all(
                weights >= 0.0
            )
        )

        self.assertEqual(
            report[
                "reference_model"
            ],
            "xgb",
        )

        self.assertGreater(
            calibrator[
                "transfer_gain"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
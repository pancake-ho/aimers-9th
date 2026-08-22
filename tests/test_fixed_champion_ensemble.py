from __future__ import annotations

import unittest

import numpy as np

from src.calibration import (
    evaluate_fixed_calibrated_weights,
)


class FixedChampionEnsembleTests(
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

        xgb_prediction = np.where(
            y > 0.5,
            0.73,
            0.43,
        ).astype(
            np.float64
        )

        cat_prediction = np.where(
            y > 0.5,
            0.70,
            0.40,
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
                "xgb": xgb_prediction,
                "cat": cat_prediction,
            },
        }

    def test_fixed_weights_are_not_reoptimized(
        self,
    ) -> None:
        folds = [
            self._fold("2023"),
            self._fold("2024"),
            self._fold(
                "2024_late_abs"
            ),
        ]

        requested = np.asarray(
            [
                0.70,
                0.30,
            ],
            dtype=np.float64,
        )

        (
            weights,
            report,
            calibrator,
        ) = (
            evaluate_fixed_calibrated_weights(
                folds,
                weights=requested,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                model_order=(
                    "xgb",
                    "cat",
                ),
                early_month_max=6,
            )
        )

        np.testing.assert_allclose(
            weights,
            requested,
            rtol=0.0,
            atol=1.0e-12,
        )

        self.assertEqual(
            report["method"],
            "fixed_v10_champion_weights",
        )

        self.assertTrue(
            report["selection_passed"]
        )

        self.assertTrue(
            calibrator["accepted"]
        )

        self.assertGreater(
            calibrator[
                "transfer_gain"
            ],
            0.0,
        )

    def test_invalid_weight_sum_is_rejected(
        self,
    ) -> None:
        folds = [
            self._fold("2023"),
            self._fold("2024"),
            self._fold(
                "2024_late_abs"
            ),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "sum to one",
        ):
            evaluate_fixed_calibrated_weights(
                folds,
                weights=(
                    0.70,
                    0.20,
                ),
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                model_order=(
                    "xgb",
                    "cat",
                ),
                early_month_max=6,
            )


if __name__ == "__main__":
    unittest.main()
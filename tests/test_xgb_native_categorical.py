from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.runtime import (
    XGB_NATIVE_UNKNOWN,
    preprocess_xgb_native_frame,
)

from src.xgb_native_cat import (
    blend_xgb_native_cat,
    select_xgb_native_cat_weight,
)


class XGBNativeCategoricalTests(
    unittest.TestCase
):
    @staticmethod
    def _preprocessor_state():
        return {
            "state_version": 1,
            "cat_cols": [
                "pitcher_id",
            ],
            "num_cols": [
                "value",
            ],
            "feature_names": [
                "pitcher_id",
                "value",
            ],
            "category_maps": {
                "pitcher_id": {
                    "10": 0,
                    "20": 1,
                },
            },
            "numeric_medians": {
                "value": 3.0,
            },
        }

    def test_native_categories_use_train_fitted_vocabulary(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "pitcher_id": [
                    10,
                    999,
                ],
                "value": [
                    1.0,
                    np.nan,
                ],
            }
        )

        transformed = (
            preprocess_xgb_native_frame(
                frame,
                self._preprocessor_state(),
            )
        )

        self.assertIsInstance(
            transformed[
                "pitcher_id"
            ].dtype,
            pd.CategoricalDtype,
        )

        self.assertEqual(
            list(
                transformed[
                    "pitcher_id"
                ].cat.categories
            ),
            [
                "10",
                "20",
                XGB_NATIVE_UNKNOWN,
            ],
        )

        self.assertEqual(
            str(
                transformed.loc[
                    0,
                    "pitcher_id",
                ]
            ),
            "10",
        )

        self.assertEqual(
            str(
                transformed.loc[
                    1,
                    "pitcher_id",
                ]
            ),
            XGB_NATIVE_UNKNOWN,
        )

        self.assertEqual(
            float(
                transformed.loc[
                    1,
                    "value",
                ]
            ),
            3.0,
        )

    def test_native_preprocessing_is_row_order_independent(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "pitcher_id": [
                    10,
                    999,
                    20,
                ],
                "value": [
                    1.0,
                    2.0,
                    3.0,
                ],
            },
            index=[
                100,
                101,
                102,
            ],
        )

        first = (
            preprocess_xgb_native_frame(
                frame,
                self._preprocessor_state(),
            )
        )

        second = (
            preprocess_xgb_native_frame(
                frame.iloc[
                    ::-1
                ],
                self._preprocessor_state(),
            )
            .reindex(
                frame.index
            )
        )

        np.testing.assert_array_equal(
            first[
                "pitcher_id"
            ].cat.codes.to_numpy(),
            second[
                "pitcher_id"
            ].cat.codes.to_numpy(),
        )

        np.testing.assert_allclose(
            first[
                "value"
            ].to_numpy(),
            second[
                "value"
            ].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )

    @staticmethod
    def _fold(
        label: str,
        *,
        native_good: bool,
    ):
        y = np.tile(
            np.asarray(
                [
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
            1000,
        )

        base = np.where(
            y > 0.5,
            0.56,
            0.44,
        )

        if native_good:
            native = np.where(
                y > 0.5,
                0.62,
                0.38,
            )
        else:
            native = np.where(
                y > 0.5,
                0.48,
                0.52,
            )

        return {
            "validation_label": (
                label
            ),
            "y_true": y,
            "predictions": {
                "xgb": base,
            },
            "xgb_native_cat_prediction": (
                native
            ),
        }

    def test_material_native_gain_can_receive_nonzero_weight(
        self,
    ) -> None:
        folds = [
            self._fold(
                "2023",
                native_good=True,
            ),
            self._fold(
                "2024",
                native_good=True,
            ),
            self._fold(
                "2024_late_abs",
                native_good=True,
            ),
        ]

        weight, report = (
            select_xgb_native_cat_weight(
                folds,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                grid_step=0.025,
                maximum_weight=0.20,
                minimum_material_protected_gain=(
                    2.0e-5
                ),
                minimum_forward_improvement=(
                    1.0e-5
                ),
                maximum_2023_regression=(
                    5.0e-5
                ),
            )
        )

        self.assertGreater(
            weight,
            0.0,
        )

        self.assertTrue(
            report[
                "accepted_nonzero"
            ]
        )

        self.assertGreaterEqual(
            report[
                "gain_vs_base"
            ][
                "2024"
            ],
            0.0,
        )

        self.assertGreaterEqual(
            report[
                "gain_vs_base"
            ][
                "2024_late_abs"
            ],
            0.0,
        )

    def test_protected_regression_forces_zero_weight(
        self,
    ) -> None:
        folds = [
            self._fold(
                "2023",
                native_good=True,
            ),
            self._fold(
                "2024",
                native_good=True,
            ),
            self._fold(
                "2024_late_abs",
                native_good=False,
            ),
        ]

        weight, report = (
            select_xgb_native_cat_weight(
                folds,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                grid_step=0.025,
                maximum_weight=0.20,
                minimum_material_protected_gain=(
                    2.0e-5
                ),
                minimum_forward_improvement=(
                    1.0e-5
                ),
                maximum_2023_regression=(
                    5.0e-5
                ),
            )
        )

        self.assertEqual(
            weight,
            0.0,
        )

        self.assertFalse(
            report[
                "accepted_nonzero"
            ]
        )

    def test_blend_bounds(
        self,
    ) -> None:
        output = blend_xgb_native_cat(
            np.asarray(
                [
                    0.1,
                    0.9,
                ]
            ),
            np.asarray(
                [
                    0.3,
                    0.7,
                ]
            ),
            0.20,
        )

        np.testing.assert_allclose(
            output,
            np.asarray(
                [
                    0.14,
                    0.86,
                ]
            ),
            rtol=0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import unittest

import pandas as pd
import numpy as np

from src.training import (
    _apply_temporal_xgb_blend,
    _split_entity_feature_view,
)

class DualXGBTrainingContractTests(
    unittest.TestCase
):
    def _make_fold(
        self,
        y_true: np.ndarray,
    ) -> dict[str, object]:
        y_true = np.asarray(
            y_true,
            dtype=np.float64,
        )

        n = len(
            y_true
        )

        base = np.full(
            n,
            0.50,
            dtype=np.float64,
        )

        recent1 = (
            0.70 * y_true
            + 0.15
        )

        recent2 = (
            0.60 * y_true
            + 0.20
        )

        return {
            "validation_label": "synthetic",

            "y_true": y_true,

            "xgb_temporal_views": {
                "base": base,
                "recent1": recent1,
                "recent2": recent2,
            },

            "predictions": {
                "xgb": base.copy(),
            },

            "metrics": {
                "xgb": {
                    "brier": 999.0,
                },
            },
        }

    def test_entity_ablation_removes_only_entity_feature_family(
        self,
    ) -> None:
        X = pd.DataFrame(
            {
                "balls_before": [
                    0.0,
                    1.0,
                ],
                "tm_hand_rel_speed_mean": [
                    90.0,
                    91.0,
                ],
                "tm_count_rel_speed_mean": [
                    89.0,
                    90.0,
                ],
                "tm_entity_map_confidence": [
                    0.8,
                    0.6,
                ],
                "tm_entity_rel_speed_mean": [
                    92.0,
                    93.0,
                ],
                "tm_entity_rel_speed_pooled": [
                    91.0,
                    91.5,
                ],
            }
        )

        baseline, entity_columns = (
            _split_entity_feature_view(
                X
            )
        )

        self.assertEqual(
            entity_columns,
            [
                "tm_entity_map_confidence",
                "tm_entity_rel_speed_mean",
                "tm_entity_rel_speed_pooled",
            ],
        )

        self.assertEqual(
            list(
                baseline.columns
            ),
            [
                "balls_before",
                "tm_hand_rel_speed_mean",
                "tm_count_rel_speed_mean",
            ],
        )

        self.assertTrue(
            all(
                not column.startswith(
                    "tm_entity_"
                )
                for column
                in baseline.columns
            )
        )

    def test_entity_ablation_requires_entity_columns(
        self,
    ) -> None:
        X = pd.DataFrame(
            {
                "balls_before": [
                    0.0,
                ],
                "tm_hand_rel_speed_mean": [
                    90.0,
                ],
            }
        )

        with self.assertRaises(
            ValueError
        ):
            _split_entity_feature_view(
                X
            )

    def test_temporal_blend_replaces_xgb_prediction(
        self,
    ) -> None:
        fold = self._make_fold(
            np.asarray(
                [
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                ]
            )
        )

        weights = np.asarray(
            [
                0.50,
                0.30,
                0.20,
            ],
            dtype=np.float64,
        )

        metrics = (
            _apply_temporal_xgb_blend(
                fold,
                weights,
            )
        )

        expected = (
            0.50
            * fold[
                "xgb_temporal_views"
            ][
                "base"
            ]
            + 0.30
            * fold[
                "xgb_temporal_views"
            ][
                "recent1"
            ]
            + 0.20
            * fold[
                "xgb_temporal_views"
            ][
                "recent2"
            ]
        )

        np.testing.assert_allclose(
            fold[
                "predictions"
            ][
                "xgb"
            ],
            expected,
            atol=1.0e-12,
            rtol=0.0,
        )

        expected_brier = float(
            np.mean(
                (
                    expected
                    - fold[
                        "y_true"
                    ]
                )
                ** 2
            )
        )

        self.assertAlmostEqual(
            metrics[
                "brier"
            ],
            expected_brier,
            places=12,
        )

        self.assertAlmostEqual(
            fold[
                "metrics"
            ][
                "xgb"
            ][
                "brier"
            ],
            expected_brier,
            places=12,
        )

    def test_different_fold_sizes_do_not_cross_contaminate(
        self,
    ) -> None:
        fold_a = self._make_fold(
            np.asarray(
                [
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                ]
            )
        )

        fold_b = self._make_fold(
            np.asarray(
                [
                    1.0,
                    0.0,
                    1.0,
                ]
            )
        )

        weights = np.asarray(
            [
                0.60,
                0.20,
                0.20,
            ],
            dtype=np.float64,
        )

        _apply_temporal_xgb_blend(
            fold_a,
            weights,
        )

        prediction_a = (
            fold_a[
                "predictions"
            ][
                "xgb"
            ].copy()
        )

        _apply_temporal_xgb_blend(
            fold_b,
            weights,
        )

        self.assertEqual(
            fold_a[
                "predictions"
            ][
                "xgb"
            ].shape,
            (5,),
        )

        self.assertEqual(
            fold_b[
                "predictions"
            ][
                "xgb"
            ].shape,
            (3,),
        )

        np.testing.assert_array_equal(
            fold_a[
                "predictions"
            ][
                "xgb"
            ],
            prediction_a,
        )


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import unittest

import numpy as np

from src.xgb_temporal import (
    blend_temporal_views,
    recent_window_positions,
    select_temporal_view_weights,
)


class XGBoostTemporalViewTests(
    unittest.TestCase
):
    def test_recent_windows_use_latest_available_seasons(
        self,
    ) -> None:
        seasons = np.asarray(
            [
                2019,
                2019,
                2020,
                2021,
                2021,
                2022,
                2022,
                2022,
            ],
            dtype=np.int16,
        )

        positions1, state1 = (
            recent_window_positions(
                seasons,
                n_seasons=1,
            )
        )

        positions2, state2 = (
            recent_window_positions(
                seasons,
                n_seasons=2,
            )
        )

        self.assertEqual(
            state1[
                "selected_seasons"
            ],
            [2022],
        )

        self.assertEqual(
            state2[
                "selected_seasons"
            ],
            [
                2021,
                2022,
            ],
        )

        np.testing.assert_array_equal(
            seasons[
                positions1
            ],
            [
                2022,
                2022,
                2022,
            ],
        )

    def test_temporal_blend(
        self,
    ) -> None:
        views = {
            "base": np.asarray(
                [0.2, 0.8]
            ),
            "recent1": np.asarray(
                [0.3, 0.7]
            ),
            "recent2": np.asarray(
                [0.4, 0.6]
            ),
        }

        prediction = (
            blend_temporal_views(
                views,
                [
                    0.5,
                    0.25,
                    0.25,
                ],
            )
        )

        np.testing.assert_allclose(
            prediction,
            [
                0.275,
                0.725,
            ],
            atol=1.0e-12,
        )

    def test_selector_falls_back_to_base(
        self,
    ) -> None:
        y = np.asarray(
            [
                0.0,
                1.0,
                0.0,
                1.0,
            ]
        )

        folds = []

        for label in (
            "2023",
            "2024",
            "2024_late_abs",
        ):
            folds.append(
                {
                    "validation_label": (
                        label
                    ),

                    "y_true": y,

                    "xgb_temporal_views": {
                        "base": np.asarray(
                            [
                                0.1,
                                0.9,
                                0.1,
                                0.9,
                            ]
                        ),

                        "recent1": np.asarray(
                            [
                                0.4,
                                0.6,
                                0.4,
                                0.6,
                            ]
                        ),

                        "recent2": np.asarray(
                            [
                                0.5,
                                0.5,
                                0.5,
                                0.5,
                            ]
                        ),
                    },
                }
            )

        weights, _ = (
            select_temporal_view_weights(
                folds,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                grid_step=0.025,
                minimum_base_weight=0.30,
                maximum_aux_weight=0.60,
                protected_labels=(
                    "2024",
                    "2024_late_abs",
                ),
                protected_tolerance=0.0,
            )
        )

        np.testing.assert_allclose(
            weights,
            [
                1.0,
                0.0,
                0.0,
            ],
            atol=1.0e-12,
        )

    def test_selector_can_choose_recent_signal(
        self,
    ) -> None:
        y = np.asarray(
            [
                0.0,
                1.0,
                0.0,
                1.0,
            ]
        )

        folds = []

        for label in (
            "2023",
            "2024",
            "2024_late_abs",
        ):
            folds.append(
                {
                    "validation_label": (
                        label
                    ),

                    "y_true": y,

                    "xgb_temporal_views": {
                        "base": np.asarray(
                            [
                                0.3,
                                0.7,
                                0.3,
                                0.7,
                            ]
                        ),

                        "recent1": np.asarray(
                            [
                                0.1,
                                0.9,
                                0.1,
                                0.9,
                            ]
                        ),

                        "recent2": np.asarray(
                            [
                                0.2,
                                0.8,
                                0.2,
                                0.8,
                            ]
                        ),
                    },
                }
            )

        weights, report = (
            select_temporal_view_weights(
                folds,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                grid_step=0.025,
                minimum_base_weight=0.30,
                maximum_aux_weight=0.60,
                protected_labels=(
                    "2024",
                    "2024_late_abs",
                ),
                protected_tolerance=0.0,
            )
        )

        self.assertLess(
            weights[0],
            1.0,
        )

        self.assertGreater(
            report[
                "gain_vs_base"
            ]["2024"],
            0.0,
        )

        self.assertGreater(
            report[
                "gain_vs_base"
            ][
                "2024_late_abs"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
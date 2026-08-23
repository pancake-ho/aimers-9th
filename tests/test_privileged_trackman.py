from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.privileged_trackman import (
    align_pitcher_season,
    build_distillation_target,
    select_lupi_weight,
)


class PrivilegedTrackmanTests(
    unittest.TestCase
):
    def test_distillation_changes_only_available_rows(
        self,
    ) -> None:
        y = np.asarray(
            [
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        teacher = np.asarray(
            [
                0.2,
                0.8,
                0.9,
                0.1,
            ],
            dtype=np.float32,
        )

        available = np.asarray(
            [
                True,
                False,
                True,
                False,
            ],
            dtype=bool,
        )

        result = (
            build_distillation_target(
                y,
                teacher,
                available,
                strength=0.20,
            )
        )

        self.assertAlmostEqual(
            float(
                result[0]
            ),
            0.04,
            places=6,
        )

        self.assertAlmostEqual(
            float(
                result[1]
            ),
            1.0,
            places=6,
        )

        self.assertAlmostEqual(
            float(
                result[2]
            ),
            0.18,
            places=6,
        )

        self.assertAlmostEqual(
            float(
                result[3]
            ),
            1.0,
            places=6,
        )

    def test_pitch_alignment_is_one_to_one(
        self,
    ) -> None:
        n = 8

        main = pd.DataFrame(
            {
                "asof_pitcher_n": (
                    np.arange(
                        1,
                        n + 1,
                    )
                ),
                "game_month": [
                    4
                ] * n,
                "game_dayofweek": [
                    2
                ] * n,
                "inning": [
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                ],
                "top_bottom": [
                    "T"
                ] * n,
                "balls_before": [
                    0,
                    1,
                    1,
                    2,
                    0,
                    0,
                    1,
                    2,
                ],
                "strikes_before": [
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    2,
                ],
                "outs_before": [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    2,
                ],
                "pitcher_hand": [
                    "Right"
                ] * n,
                "batter_hand": [
                    "Left",
                    "Right",
                    "Left",
                    "Right",
                    "Left",
                    "Right",
                    "Left",
                    "Right",
                ],
            },
            index=np.arange(
                100,
                100 + n,
            ),
        )

        trackman = main.drop(
            columns=[
                "asof_pitcher_n",
            ]
        ).copy()

        trackman[
            "_parsed_game_date"
        ] = pd.Timestamp(
            "2024-04-01"
        )

        trackman[
            "trackman_game_id"
        ] = 1

        trackman[
            "pitch_no"
        ] = np.arange(
            1,
            n + 1,
        )

        trackman[
            "trackman_id"
        ] = np.arange(
            1000,
            1000 + n,
        )

        trackman[
            "pitch_type_group"
        ] = "fastball"

        for column in (
            "rel_speed",
            "spin_rate",
            "induced_vert_break",
            "horz_break",
            "extension",
            "rel_height",
            "rel_side",
            "zone_speed",
        ):
            trackman[
                column
            ] = 1.0

        result = (
            align_pitcher_season(
                main,
                trackman,
                entity_confidence=0.9,
                min_matching_block=6,
                min_pair_matched_rows=6,
                min_pair_match_ratio=0.5,
            )
        )

        self.assertEqual(
            len(result),
            n,
        )

        self.assertFalse(
            result[
                "main_index"
            ]
            .duplicated()
            .any()
        )

        self.assertFalse(
            result[
                "trackman_id"
            ]
            .duplicated()
            .any()
        )

    @staticmethod
    def _fold(
        label: str,
    ) -> dict:
        y = np.asarray(
            [
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        base = np.asarray(
            [
                0.45,
                0.55,
                0.45,
                0.55,
            ],
            dtype=np.float64,
        )

        student = np.asarray(
            [
                0.40,
                0.60,
                0.40,
                0.60,
            ],
            dtype=np.float64,
        )

        return {
            "validation_label": (
                label
            ),

            "y_true": y,

            "predictions": {
                "xgb": base,
            },

            "xgb_lupi_prediction": (
                student
            ),
        }

    def test_lupi_selector_protects_forward_folds(
        self,
    ) -> None:
        folds = [
            self._fold(
                "2023"
            ),
            self._fold(
                "2024"
            ),
            self._fold(
                "2024_late_abs"
            ),
        ]

        (
            weight,
            report,
        ) = select_lupi_weight(
            folds,
            candidate_weights=(
                0.0,
                0.1,
                0.2,
                0.3,
            ),
            fold_importance=(
                0.05,
                0.45,
                0.50,
            ),
            minimum_forward_gain=0.0,
            maximum_2023_regression=0.0,
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


if __name__ == "__main__":
    unittest.main()
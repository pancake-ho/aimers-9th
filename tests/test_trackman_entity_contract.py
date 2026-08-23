from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.config import FeatureConfig
from src.features import StrictPastTrackmanFeatures
from src.runtime import add_trackman_features
from src.trackman_entity import (
    ResolutionThresholds,
    _official_team_token,
    resolve_pitcher_entities,
)


METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
)


def _main_rows(
    include_future: bool = False,
) -> pd.DataFrame:
    rows = []

    specs = {
        11: (
            12,
            0.80,
            0.15,
            0.05,
        ),
        22: (
            13,
            0.25,
            0.65,
            0.10,
        ),
    }

    for season in (
        2019,
        2020,
    ):
        for (
            pitcher_id,
            (
                team,
                fastball,
                breaking,
                offspeed,
            ),
        ) in specs.items():
            for pitch in range(
                20
                + 5
                * (
                    season
                    - 2019
                )
            ):
                rows.append(
                    {
                        "season": season,
                        "pitcher_id": pitcher_id,
                        "pitcher_hand": 2,
                        "pitcher_team_id": team,
                        "asof_pitcher_pitchmix_n": (
                            pitch
                            + 20
                            * (
                                season
                                - 2019
                            )
                        ),
                        "asof_pitcher_fastball_rate": fastball,
                        "asof_pitcher_breaking_rate": breaking,
                        "asof_pitcher_offspeed_rate": offspeed,
                    }
                )

    if include_future:
        rows.append(
            {
                "season": 2021,
                "pitcher_id": 11,
                "pitcher_hand": 2,
                "pitcher_team_id": 12,
                "asof_pitcher_pitchmix_n": 999,
                "asof_pitcher_fastball_rate": 0.01,
                "asof_pitcher_breaking_rate": 0.01,
                "asof_pitcher_offspeed_rate": 0.98,
            }
        )

    return pd.DataFrame(
        rows
    )


def _trackman_rows(
    include_future: bool = False,
) -> pd.DataFrame:
    rows = []

    specs = {
        101: (
            "DOO_BEA",
            92.0,
        ),
        202: (
            "LG_TWI",
            84.0,
        ),
    }

    for season in (
        2019,
        2020,
    ):
        for (
            trackman_id,
            (
                team,
                speed,
            ),
        ) in specs.items():
            n = (
                20
                + 5
                * (
                    season
                    - 2019
                )
            )

            for pitch in range(n):
                if trackman_id == 101:
                    group = (
                        "fastball"
                        if pitch
                        < int(
                            0.80
                            * n
                        )
                        else "breaking"
                    )
                else:
                    group = (
                        "breaking"
                        if pitch
                        < int(
                            0.65
                            * n
                        )
                        else "fastball"
                    )

                row = {
                    "season": season,
                    "pitcher_trackman_id": (
                        trackman_id
                    ),
                    "pitcher_team": team,
                    "pitcher_hand": "Right",
                    "batter_hand": "Right",
                    "balls_before": (
                        pitch
                        % 4
                    ),
                    "strikes_before": (
                        pitch
                        % 3
                    ),
                    "pitch_type_group": group,
                }

                row.update(
                    {
                        metric: (
                            speed
                            + 0.01
                            * pitch
                        )
                        for metric
                        in METRICS
                    }
                )

                rows.append(
                    row
                )

    if include_future:
        row = {
            "season": 2021,
            "pitcher_trackman_id": 101,
            "pitcher_team": "DOO_BEA",
            "pitcher_hand": "Right",
            "batter_hand": "Right",
            "balls_before": 0,
            "strikes_before": 0,
            "pitch_type_group": "offspeed",
        }

        row.update(
            {
                metric: -999.0
                for metric
                in METRICS
            }
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def _staggered_main_rows() -> pd.DataFrame:
    base = _main_rows(
        False
    )

    extra = []

    for pitch in range(30):
        extra.append(
            {
                "season": 2021,
                "pitcher_id": 11,
                "pitcher_hand": 2,
                "pitcher_team_id": 12,
                "asof_pitcher_pitchmix_n": (
                    45
                    + pitch
                ),
                "asof_pitcher_fastball_rate": 0.80,
                "asof_pitcher_breaking_rate": 0.15,
                "asof_pitcher_offspeed_rate": 0.05,
            }
        )

    return pd.concat(
        [
            base,
            pd.DataFrame(
                extra
            ),
        ],
        ignore_index=True,
    )


def _staggered_trackman_rows() -> pd.DataFrame:
    base = _trackman_rows(
        False
    )

    extra = []

    for pitch in range(30):
        row = {
            "season": 2021,
            "pitcher_trackman_id": 101,
            "pitcher_team": "DOO_BEA",
            "pitcher_hand": "Right",
            "batter_hand": "Right",
            "balls_before": (
                pitch
                % 4
            ),
            "strikes_before": (
                pitch
                % 3
            ),
            "pitch_type_group": (
                "fastball"
                if pitch < 24
                else "breaking"
            ),
        }

        row.update(
            {
                metric: (
                    94.0
                    + 0.01
                    * pitch
                )
                for metric
                in METRICS
            }
        )

        extra.append(
            row
        )

    return pd.concat(
        [
            base,
            pd.DataFrame(
                extra
            ),
        ],
        ignore_index=True,
    )


def _test_config() -> FeatureConfig:
    return replace(
        FeatureConfig(),

        trackman_entity_enabled=True,
        trackman_entity_v2_enabled=True,

        trackman_entity_min_pitches=5,
        trackman_entity_max_distance=100.0,
        trackman_entity_min_margin_ratio=1.01,
        trackman_entity_strong_margin_ratio=1.01,
        trackman_entity_min_count_ratio=0.2,
        trackman_entity_max_count_ratio=5.0,

        trackman_entity_team_penalty=0.0,
        trackman_entity_team_min_support=99,

        trackman_entity_min_matched_pitchers=1,
        trackman_entity_min_row_coverage=0.0,
        trackman_entity_min_team_mappings=0,

        trackman_entity_pool_strength=20.0,
        trackman_entity_count_pool_strength=5.0,
        trackman_entity_freshness_decay=0.70,
    )


class TrackmanEntityContractTests(
    unittest.TestCase
):
    def _build(
        self,
        *,
        main: pd.DataFrame | None = None,
        trackman: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "trackman.csv"
            )

            (
                trackman
                if trackman is not None
                else _trackman_rows(
                    False
                )
            ).to_csv(
                path,
                index=False,
            )

            builder = (
                StrictPastTrackmanFeatures(
                    _test_config()
                )
                .fit_from_csv(
                    path,
                    main_train=(
                        main
                        if main is not None
                        else _main_rows(
                            False
                        )
                    ),
                )
            )

            return (
                builder
                .export_state()
            )

    def test_next_main_season_is_materialized_after_trackman_ends(
        self,
    ) -> None:
        state = self._build()

        self.assertEqual(
            state[
                "source_seasons"
            ],
            [
                2019,
                2020,
            ],
        )

        self.assertEqual(
            state[
                "max_target_season"
            ],
            2021,
        )

        mapping = (
            state[
                "entity"
            ][
                "profiles"
            ][
                "mapping"
            ][
                "lookup"
            ]
        )

        self.assertIn(
            (
                2021,
                11,
            ),
            mapping,
        )

        self.assertIn(
            (
                2021,
                22,
            ),
            mapping,
        )

    def test_future_rows_cannot_change_past_mapping_or_profile(
        self,
    ) -> None:
        past = self._build()

        with_future = self._build(
            main=_main_rows(
                True
            ),
            trackman=_trackman_rows(
                True
            ),
        )

        for profile_name in (
            "mapping",
            "pitcher",
            "count",
            "recent",
            "arsenal",
        ):
            past_lookup = (
                past[
                    "entity"
                ][
                    "profiles"
                ][
                    profile_name
                ][
                    "lookup"
                ]
            )

            future_lookup = (
                with_future[
                    "entity"
                ][
                    "profiles"
                ][
                    profile_name
                ][
                    "lookup"
                ]
            )

            past_2021 = {
                key: value
                for key, value
                in past_lookup.items()
                if key[0] == 2021
            }

            future_2021 = {
                key: value
                for key, value
                in future_lookup.items()
                if key[0] == 2021
            }

            self.assertEqual(
                past_2021,
                future_2021,
            )

    def test_recent_profile_uses_each_pitchers_own_latest_season(
        self,
    ) -> None:
        state = self._build(
            main=_staggered_main_rows(),
            trackman=_staggered_trackman_rows(),
        )

        recent = (
            state[
                "entity"
            ][
                "profiles"
            ][
                "recent"
            ][
                "lookup"
            ]
        )

        self.assertIn(
            (
                2022,
                101,
            ),
            recent,
        )

        # Critical regression:
        # pitcher 202's latest season is 2020 even though pitcher 101
        # has Trackman in 2021.
        self.assertIn(
            (
                2022,
                202,
            ),
            recent,
        )

        pitcher = (
            state[
                "entity"
            ][
                "profiles"
            ][
                "pitcher"
            ][
                "lookup"
            ]
        )

        self.assertEqual(
            float(
                pitcher[
                    (
                        2022,
                        101,
                    )
                ][
                    "tm_entity_freshness_gap"
                ]
            ),
            0.0,
        )

        self.assertEqual(
            float(
                pitcher[
                    (
                        2022,
                        202,
                    )
                ][
                    "tm_entity_freshness_gap"
                ]
            ),
            1.0,
        )

    def test_pre_dedup_collision_audit_is_not_erased(
        self,
    ) -> None:
        source = _main_rows(
            False
        )

        first = source.loc[
            source[
                "pitcher_id"
            ]
            == 11
        ].copy()

        duplicate = first.copy()
        duplicate[
            "pitcher_id"
        ] = 33

        main = pd.concat(
            [
                first,
                duplicate,
            ],
            ignore_index=True,
        )

        thresholds = (
            ResolutionThresholds(
                min_pitches=5,
                max_distance=100.0,
                min_margin_ratio=1.01,
                strong_margin_ratio=1.01,
                min_count_ratio=0.2,
                max_count_ratio=5.0,
                team_penalty=0.0,
                team_min_support=99,
                team_min_dominance=1.0,
            )
        )

        result = (
            resolve_pitcher_entities(
                main,
                _trackman_rows(
                    False
                ),
                seasons=(
                    2019,
                    2020,
                ),
                thresholds=thresholds,
            )
        )

        self.assertEqual(
            result.audit[
                "accepted_candidate_count"
            ],
            2,
        )

        self.assertEqual(
            result.audit[
                "unique_trackman_candidate_count"
            ],
            1,
        )

        self.assertEqual(
            result.audit[
                "collided_trackman_ids"
            ],
            [
                101,
            ],
        )

        self.assertEqual(
            result.audit[
                "collision_excess"
            ],
            1,
        )

        self.assertAlmostEqual(
            result.audit[
                "collision_rate"
            ],
            0.5,
        )

        self.assertTrue(
            result.audit[
                "final_one_to_one"
            ]
        )

        self.assertEqual(
            len(
                result.mapping
            ),
            1,
        )

    def test_official_team_tokens_are_not_semantically_aliased(
        self,
    ) -> None:
        self.assertEqual(
            _official_team_token(
                "SK_WYV"
            ),
            "SK_WYV",
        )

        self.assertEqual(
            _official_team_token(
                "SSG_LAN"
            ),
            "SSG_LAN",
        )

        self.assertNotEqual(
            _official_team_token(
                "SK_WYV"
            ),
            _official_team_token(
                "SSG_LAN"
            ),
        )

    def test_population_and_entity_count_profiles_do_not_collide(
        self,
    ) -> None:
        state = self._build()

        raw = pd.DataFrame(
            {
                "season": [
                    2021,
                ],
                "pitcher_id": [
                    11,
                ],
                "pitcher_hand": [
                    2,
                ],
                "batter_hand": [
                    2,
                ],
                "balls_before": [
                    0,
                ],
                "strikes_before": [
                    0,
                ],
            }
        )

        transformed = (
            add_trackman_features(
                raw,
                state,
            )
        )

        self.assertTrue(
            transformed
            .columns
            .is_unique
        )

        self.assertIn(
            "tm_count_rel_speed_mean",
            transformed.columns,
        )

        self.assertIn(
            "tm_entity_count_rel_speed_mean",
            transformed.columns,
        )

        self.assertGreater(
            float(
                transformed.loc[
                    0,
                    "tm_count_available",
                ]
            ),
            0.0,
        )

        self.assertGreater(
            float(
                transformed.loc[
                    0,
                    "tm_entity_count_available",
                ]
            ),
            0.0,
        )

    def test_v2_partial_pooling_matches_fixed_formula(
        self,
    ) -> None:
        state = self._build()

        raw = pd.DataFrame(
            {
                "season": [
                    2021,
                ],
                "pitcher_id": [
                    11,
                ],
                "pitcher_hand": [
                    2,
                ],
                "batter_hand": [
                    2,
                ],
                "balls_before": [
                    0,
                ],
                "strikes_before": [
                    0,
                ],
            }
        )

        transformed = (
            add_trackman_features(
                raw,
                state,
            )
        )

        confidence = float(
            transformed.loc[
                0,
                "tm_entity_map_confidence",
            ]
        )

        n = float(
            np.expm1(
                transformed.loc[
                    0,
                    "tm_entity_log_n",
                ]
            )
        )

        gap = float(
            transformed.loc[
                0,
                "tm_entity_freshness_gap",
            ]
        )

        config = _test_config()

        expected_alpha = (
            confidence
            * n
            / (
                n
                + config
                .trackman_entity_pool_strength
            )
            * np.exp(
                -config
                .trackman_entity_freshness_decay
                * gap
            )
        )

        self.assertAlmostEqual(
            float(
                transformed.loc[
                    0,
                    "tm_entity_pool_alpha",
                ]
            ),
            expected_alpha,
            places=6,
        )

        population = float(
            transformed.loc[
                0,
                "tm_hand_rel_speed_mean",
            ]
        )

        entity = float(
            transformed.loc[
                0,
                "tm_entity_rel_speed_mean",
            ]
        )

        expected_pooled = (
            population
            + expected_alpha
            * (
                entity
                - population
            )
        )

        self.assertAlmostEqual(
            float(
                transformed.loc[
                    0,
                    "tm_entity_rel_speed_pooled",
                ]
            ),
            expected_pooled,
            places=5,
        )

    def test_ambiguous_identity_is_rejected(
        self,
    ) -> None:
        main = (
            _main_rows(
                False
            )
            .loc[
                lambda frame:
                frame[
                    "pitcher_id"
                ]
                == 11
            ]
        )

        trackman = _trackman_rows(
            False
        )

        duplicate = (
            trackman.loc[
                trackman[
                    "pitcher_trackman_id"
                ]
                == 101
            ]
            .copy()
        )

        duplicate[
            "pitcher_trackman_id"
        ] = 303

        trackman = pd.concat(
            [
                trackman,
                duplicate,
            ],
            ignore_index=True,
        )

        thresholds = (
            ResolutionThresholds(
                min_pitches=5,
                max_distance=100.0,
                min_margin_ratio=1.35,
                strong_margin_ratio=2.0,
                min_count_ratio=0.2,
                max_count_ratio=5.0,
                team_penalty=0.0,
                team_min_support=99,
                team_min_dominance=1.0,
            )
        )

        result = (
            resolve_pitcher_entities(
                main,
                trackman,
                seasons=(
                    2019,
                    2020,
                ),
                thresholds=thresholds,
            )
        )

        self.assertEqual(
            result.mapping,
            {},
        )


if __name__ == "__main__":
    unittest.main()
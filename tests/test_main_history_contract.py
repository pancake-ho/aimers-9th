from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.config import FeatureConfig
from src.features import StrictPastMainHistoryFeatures
from src.runtime import add_main_history_features


def _history_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020, 2021, 2021, 2022, 2022],
            "pitcher_id": [1, 1, 2, 2, 1, 2, 1, 2],
            "pitcher_hand": [2, 2, 1, 1, 2, 1, 2, 1],
            "batter_hand": [2, 1, 1, 2, 2, 1, 1, 2],
            "balls_before": [0, 2, 0, 2, 0, 0, 1, 1],
            "strikes_before": [0, 1, 0, 1, 0, 0, 1, 1],
            "control_success": [1, 1, 0, 0, 1, 0, 1, 0],
        }
    )


class MainHistoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FeatureConfig()
        self.rows = _history_rows()

    def _state(self, rows: pd.DataFrame):
        return StrictPastMainHistoryFeatures(self.config).fit(
            rows, "control_success"
        ).export_state()

    def test_same_season_target_never_changes_that_seasons_features(self) -> None:
        original = self._state(self.rows)
        changed_rows = self.rows.copy()
        changed_rows.loc[changed_rows["season"] == 2021, "control_success"] = 1 - changed_rows.loc[
            changed_rows["season"] == 2021, "control_success"
        ]
        changed = self._state(changed_rows)

        valid_2021 = self.rows.loc[self.rows["season"] == 2021].drop(
            columns="control_success"
        )
        before = add_main_history_features(valid_2021, original)
        after = add_main_history_features(valid_2021, changed)
        assert_frame_equal(before, after, check_dtype=True)

    def test_previous_season_target_can_change_next_seasons_profile(self) -> None:
        original = self._state(self.rows)
        changed_rows = self.rows.copy()
        changed_rows.loc[
            (changed_rows["season"] == 2021) & (changed_rows["pitcher_id"] == 1),
            "control_success",
        ] = 0
        changed = self._state(changed_rows)

        row_2022 = self.rows.loc[
            (self.rows["season"] == 2022) & (self.rows["pitcher_id"] == 1)
        ].drop(columns="control_success")
        before = add_main_history_features(row_2022, original)
        after = add_main_history_features(row_2022, changed)
        self.assertNotEqual(
            float(before["hist_pitcher_effect"].iloc[0]),
            float(after["hist_pitcher_effect"].iloc[0]),
        )

    def test_profile_lookup_is_row_independent_and_order_independent(self) -> None:
        state = self._state(self.rows)
        raw = self.rows.drop(columns="control_success")
        batch = add_main_history_features(raw, state).sort_index()
        singles = pd.concat(
            [add_main_history_features(raw.iloc[[i]], state) for i in range(len(raw))]
        ).sort_index()
        assert_frame_equal(batch, singles, check_dtype=True)

        shuffled = raw.sample(frac=1.0, random_state=2026)
        shuffled_features = add_main_history_features(shuffled, state).sort_index()
        assert_frame_equal(batch, shuffled_features, check_dtype=True)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FeatureConfig
from src.features import LeakageSafeFeatureEngineer
from src.preprocessing import TabularPreprocessor
from src.runtime import apply_probability_bias, blend_predictions


def _raw_rows() -> pd.DataFrame:
    base = {
        "row_id": "test_000001",
        "season": 2025,
        "game_month": 4,
        "game_dayofweek": 6,
        "inning": 7,
        "top_bottom": "T",
        "game_type": "R",
        "balls_before": 3,
        "strikes_before": 2,
        "outs_before": 1,
        "run_top_before": 2,
        "run_bot_before": 1,
        "run_total_before": 3,
        "score_diff_home": -1,
        "score_diff_pitcher_team": 1,
        "runner_on_1b": 1,
        "runner_on_2b": 1,
        "runner_on_3b": 0,
        "num_runners_on": 2,
        "base_state": "110",
        "home_win_expectancy": 41.0,
        "away_win_expectancy": 59.0,
        "li": 2.15,
        "pitcher_id": "pitcher_001",
        "batter_id": "batter_001",
        "pitcher_hand": "Right",
        "batter_hand": "Left",
        "pitcher_team_id": "team_01",
        "batter_team_id": "team_02",
        "asof_pitcher_n": 800,
        "asof_pitcher_success_rate": 0.51,
        "asof_pitcher_reverse_rate": 0.11,
        "asof_pitcher_middle_rate": 0.17,
        "asof_pitcher_ball_rate": 0.32,
        "asof_pitcher_strike_rate": 0.57,
        "asof_pitcher_prev1_game_success_rate": 0.47,
        "asof_pitcher_prev3_game_success_rate": 0.49,
        "asof_pitcher_prev5_game_success_rate": 0.50,
        "asof_pitcher_prev1_game_middle_rate": 0.18,
        "asof_pitcher_prev3_game_middle_rate": 0.17,
        "asof_pitcher_prev5_game_middle_rate": 0.16,
        "asof_batter_n": 650,
        "asof_batter_success_rate": 0.53,
        "asof_batter_middle_rate": 0.18,
        "asof_pitcher_pitchmix_n": 790,
        "asof_pitcher_fastball_rate": 0.55,
        "asof_pitcher_breaking_rate": 0.29,
        "asof_pitcher_offspeed_rate": 0.16,
    }
    rows = []
    for index in range(4):
        row = dict(base)
        row["row_id"] = f"test_{index + 1:06d}"
        row["balls_before"] = index
        row["strikes_before"] = min(index, 2)
        row["pitcher_id"] = f"pitcher_{index % 2 + 1:03d}"
        row["asof_pitcher_success_rate"] += 0.01 * index
        rows.append(row)
    return pd.DataFrame(rows)


class RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = _raw_rows()
        cls.config = FeatureConfig()
        cls.engineer = LeakageSafeFeatureEngineer(cls.config)

    def test_feature_values_do_not_depend_on_other_test_rows(self):
        batch = self.engineer.transform(self.raw)
        singles = pd.concat(
            [self.engineer.transform(self.raw.iloc[[index]]) for index in range(len(self.raw))]
        ).sort_index()
        assert_frame_equal(batch.sort_index(), singles, check_dtype=True)

    def test_feature_values_do_not_depend_on_test_order(self):
        natural = self.engineer.transform(self.raw).set_index("row_id").sort_index()
        shuffled_raw = self.raw.sample(frac=1.0, random_state=2026)
        shuffled = self.engineer.transform(shuffled_raw).set_index("row_id").sort_index()
        assert_frame_equal(natural, shuffled, check_dtype=True)

    def test_preprocessor_unknown_categories_are_stable(self):
        train_features = self.engineer.transform(self.raw.iloc[:3])
        valid_raw = self.raw.iloc[[3]].copy()
        valid_raw["pitcher_id"] = "never_seen_pitcher"
        valid_features = self.engineer.transform(valid_raw)
        preprocessor = TabularPreprocessor(
            self.config.categorical_cols,
            self.config.excluded_cols,
        ).fit(train_features)
        X = preprocessor.transform(valid_features)
        self.assertEqual(int(X["pitcher_id"].iloc[0]), -1)
        self.assertTrue(np.isfinite(X.to_numpy()).all())

    def test_blend_and_bias_remain_probabilities(self):
        pred = blend_predictions(
            [np.array([0.1, 0.9]), np.array([0.3, 0.7])],
            [0.43, 0.57],
        )
        corrected = apply_probability_bias(pred, 0.01)
        self.assertTrue(((corrected > 0.0) & (corrected < 1.0)).all())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.baseline.comparison import clustered_brier_delta_bootstrap
from src.baseline.pitcher_residual import (
    PitcherBaselineConfig,
    margin_to_probability,
    pitcher_baseline_probability,
    probability_to_margin,
    select_strict_residual_features,
)


class PitcherResidualBaselineTests(unittest.TestCase):
    def test_empirical_bayes_baseline_handles_cold_start_and_history(self) -> None:
        frame = pd.DataFrame(
            {
                "asof_pitcher_n": [0, 100, 2000, np.nan],
                "asof_pitcher_success_rate": [np.nan, 0.60, 0.57, 0.90],
            }
        )
        config = PitcherBaselineConfig(
            prior_probability=0.5,
            prior_strength=100.0,
            probability_clip=0.02,
        )
        actual = pitcher_baseline_probability(frame, config)
        expected = np.asarray(
            [
                0.5,
                (100.0 * 0.60 + 100.0 * 0.5) / 200.0,
                (2000.0 * 0.57 + 100.0 * 0.5) / 2100.0,
                0.5,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)

    def test_logit_round_trip(self) -> None:
        probability = np.asarray([0.02, 0.49, 0.57, 0.98], dtype=np.float64)
        actual = margin_to_probability(probability_to_margin(probability))
        np.testing.assert_allclose(actual, probability, rtol=0.0, atol=1e-7)

    def test_strict_residual_removes_pitcher_intercept_but_keeps_context(self) -> None:
        frame = pd.DataFrame(
            {
                "pitcher_id": [1],
                "pitcher_count_combo": [2],
                "asof_pitcher_success_rate_smoothed": [0.55],
                "asof_pitcher_prev3_game_success_rate": [0.52],
                "recent_success_blend": [0.53],
                "asof_pitcher_n": [500],
                "pitcher_history_log_n": [6.2],
                "pitcher_form_gap": [-0.02],
                "diff_prev3_success": [-0.03],
                "count_pressure": [0.6],
                "li_log": [0.7],
            }
        )
        selected = select_strict_residual_features(frame)
        self.assertNotIn("pitcher_id", selected.columns)
        self.assertNotIn("pitcher_count_combo", selected.columns)
        self.assertNotIn("asof_pitcher_success_rate_smoothed", selected.columns)
        self.assertNotIn("asof_pitcher_prev3_game_success_rate", selected.columns)
        self.assertNotIn("recent_success_blend", selected.columns)
        self.assertIn("asof_pitcher_n", selected.columns)
        self.assertIn("pitcher_history_log_n", selected.columns)
        self.assertIn("pitcher_form_gap", selected.columns)
        self.assertIn("diff_prev3_success", selected.columns)
        self.assertIn("count_pressure", selected.columns)

    def test_cluster_bootstrap_is_paired_and_deterministic(self) -> None:
        y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float64)
        reference = np.asarray([0.45, 0.55, 0.40, 0.60, 0.35, 0.65])
        candidate = np.asarray([0.25, 0.75, 0.20, 0.80, 0.30, 0.70])
        clusters = np.asarray([1, 1, 2, 2, 3, 3])
        first = clustered_brier_delta_bootstrap(
            y,
            candidate,
            reference,
            clusters,
            n_bootstrap=200,
            random_seed=2026,
        )
        second = clustered_brier_delta_bootstrap(
            y,
            candidate,
            reference,
            clusters,
            n_bootstrap=200,
            random_seed=2026,
        )
        self.assertEqual(first, second)
        self.assertLess(first["brier_delta"], 0.0)
        self.assertEqual(first["n_clusters"], 3)


if __name__ == "__main__":
    unittest.main()

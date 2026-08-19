from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.tabfm.config import TabDPTExperimentConfig
from src.tabfm.context import (
    select_recent_context_indices,
    tabdpt_feature_names,
    to_tabdpt_array,
)


class TabDPTContextTests(unittest.TestCase):
    def test_recent_context_is_deterministic_and_stays_in_latest_season(self):
        rows = []
        for season in (2022, 2023, 2024):
            for index in range(120):
                rows.append(
                    {
                        "season": season,
                        "game_month": 1 + index % 6,
                        "control_success": (index + season) % 2,
                    }
                )
        frame = pd.DataFrame(rows)
        allowed = np.arange(len(frame), dtype=np.int64)
        first = select_recent_context_indices(
            frame,
            allowed,
            target_col="control_success",
            context_size=60,
            seed=42,
        )
        second = select_recent_context_indices(
            frame,
            allowed,
            target_col="control_success",
            context_size=60,
            seed=42,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(frame.iloc[first]["season"]), {2024})
        self.assertAlmostEqual(
            frame.iloc[first]["control_success"].mean(),
            frame[frame["season"] == 2024]["control_success"].mean(),
            places=6,
        )

    def test_context_respects_allowed_fold_boundary(self):
        frame = pd.DataFrame(
            {
                "season": [2023] * 20 + [2024] * 20,
                "game_month": [1, 2] * 20,
                "control_success": [0, 1] * 20,
            }
        )
        allowed = np.arange(20, dtype=np.int64)
        selected = select_recent_context_indices(
            frame,
            allowed,
            target_col="control_success",
            context_size=10,
            seed=7,
        )
        self.assertTrue(np.isin(selected, allowed).all())
        self.assertEqual(set(frame.iloc[selected]["season"]), {2023})

    def test_numerical_view_refuses_silent_pca(self):
        with self.assertRaises(ValueError):
            tabdpt_feature_names([f"f{i}" for i in range(129)], max_features=128)
        names = tabdpt_feature_names(["a", "b"], max_features=128)
        array = to_tabdpt_array(pd.DataFrame({"a": [1], "b": [2]}), names)
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(array.shape, (1, 2))

    def test_config_matches_l4_safety_contract(self):
        config = TabDPTExperimentConfig()
        config.validate()
        self.assertEqual(config.context_size, 32_768)
        self.assertLessEqual(config.max_features, 128)
        self.assertFalse(config.compile_model)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.baseline.pitcher_residual import (
    probability_to_margin,
    train_residual_xgboost_fold,
    validate_residual_xgboost_backend,
)
from src.config import ModelConfig
from src.models import xgb


@unittest.skipIf(xgb is None, "XGBoost is not installed")
class PitcherResidualXGBoostTests(unittest.TestCase):
    def test_base_margin_fold_training_returns_probabilities(self) -> None:
        rng = np.random.default_rng(2026)
        n_rows = 160
        context = rng.normal(size=n_rows).astype(np.float32)
        leverage = rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32)
        baseline_probability = np.clip(
            0.50 + 0.08 * rng.normal(size=n_rows), 0.15, 0.85
        )
        true_margin = probability_to_margin(baseline_probability) + 0.7 * context
        true_probability = 1.0 / (1.0 + np.exp(-true_margin))
        target = rng.binomial(1, true_probability).astype(np.float32)
        frame = pd.DataFrame(
            {"context": context, "leverage": leverage}, dtype=np.float32
        )
        split = 120
        config = ModelConfig(
            xgb_device="cpu",
            num_threads=2,
            xgb_num_boost_round=30,
            xgb_early_stopping_rounds=5,
            xgb_max_depth=2,
            xgb_min_child_weight=1.0,
            xgb_max_bin=64,
        )
        model, prediction, best_iteration = train_residual_xgboost_fold(
            frame.iloc[:split],
            target[:split],
            probability_to_margin(baseline_probability[:split]),
            frame.iloc[split:],
            target[split:],
            probability_to_margin(baseline_probability[split:]),
            np.ones(split, dtype=np.float32),
            config,
        )
        self.assertEqual(prediction.shape, (n_rows - split,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.logical_and(prediction > 0.0, prediction < 1.0).all())
        self.assertGreaterEqual(best_iteration, 1)
        del model

    def test_backend_probe(self) -> None:
        validate_residual_xgboost_backend(
            ModelConfig(xgb_device="cpu", num_threads=2, xgb_max_bin=64)
        )


if __name__ == "__main__":
    unittest.main()

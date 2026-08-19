from __future__ import annotations

import unittest

from src.config import ExperimentConfig, NeuralConfig


class ConfigContractTests(unittest.TestCase):
    def test_neural_models_have_separate_optimization_regimes(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.models.num_threads, 16)
        self.assertEqual(config.neural.models, ("resnet", "ft_transformer"))
        self.assertEqual(
            config.neural.ft_d_token % config.neural.ft_n_heads,
            0,
        )
        self.assertEqual(config.ensemble_grid_step, 0.025)
        self.assertEqual(config.temporal_fold_importance, (0.15, 0.35, 0.50))
        self.assertEqual(
            config.ensemble_protected_folds,
            ("2024", "2024_late_abs"),
        )
        self.assertFalse(config.use_main_history)
        self.assertIn("season", config.features.excluded_cols)
        self.assertEqual(config.features.history_season_decay, 0.70)
        self.assertLess(
            config.features.history_count_strength,
            config.features.history_pitcher_strength,
        )

    def test_neural_training_can_be_disabled_without_changing_model_defaults(self) -> None:
        enabled = NeuralConfig()
        disabled = NeuralConfig(enabled=False)

        self.assertTrue(enabled.enabled)
        self.assertFalse(disabled.enabled)
        self.assertEqual(enabled.models, ("resnet", "ft_transformer"))
        self.assertEqual(enabled.resnet_max_epochs, disabled.resnet_max_epochs)
        self.assertEqual(enabled.ft_max_epochs, disabled.ft_max_epochs)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from src.config import (
    ExperimentConfig,
    NeuralConfig,
)
from src.models import (
    GBDT_MODEL_ORDER,
    MODEL_ORDER,
)


class ConfigContractTests(
    unittest.TestCase
):
    def test_current_ensemble_contract(
        self,
    ) -> None:
        config = ExperimentConfig()

        self.assertEqual(
            config.models.num_threads,
            16,
        )

        self.assertEqual(
            GBDT_MODEL_ORDER,
            (
                "xgb",
                "lgb",
                "cat",
            ),
        )

        self.assertEqual(
            MODEL_ORDER,
            (
                "xgb",
                "lgb",
                "cat",
                "resnet",
                "ft_transformer",
            ),
        )

        self.assertEqual(
            config.neural.models,
            (
                "resnet",
                "ft_transformer",
            ),
        )

        self.assertEqual(
            config.neural.ft_d_token
            % config.neural.ft_n_heads,
            0,
        )

        self.assertEqual(
            config.ensemble_grid_step,
            0.025,
        )

        self.assertEqual(
            config.temporal_fold_importance,
            (
                0.05,
                0.45,
                0.50,
            ),
        )

        self.assertEqual(
            config.ensemble_protected_folds,
            (
                "2024",
                "2024_late_abs",
            ),
        )

        self.assertFalse(
            config.use_main_history
        )

        self.assertTrue(
            config.features.trackman_entity_enabled
        )
        self.assertTrue(
            config.features.trackman_entity_v2_enabled
        )
        self.assertEqual(
            config.features.trackman_entity_team_penalty,
            2.0,
        )
        self.assertEqual(
            config.features.trackman_entity_min_team_mappings,
            0,
        )
        self.assertGreater(
            config.features.trackman_entity_pool_strength,
            0.0,
        )
        self.assertGreater(
            config.features.trackman_entity_count_pool_strength,
            0.0,
        )

        self.assertIn(
            "season",
            config.features.excluded_cols,
        )

        self.assertEqual(
            config.features
            .history_season_decay,
            0.70,
        )

        self.assertLess(
            config.features
            .history_count_strength,
            config.features
            .history_pitcher_strength,
        )

        self.assertGreater(
            config.models
            .lgb_num_boost_round,
            0,
        )
        self.assertEqual(
            config
            .ensemble_shrinkage_reference_model,
            "xgb",
        )

        self.assertEqual(
            config
            .ensemble_shrinkage_alphas,
            (
                0.00,
                0.25,
                0.50,
                0.75,
                1.00,
            ),
        )

        self.assertEqual(
            config
            .submission_gate_calibration_min_transfer_gain,
            3.0e-5,
        )

        self.assertEqual(
            config
            .submission_gate_late_calibrated_min_gain_vs_reference,
            5.0e-5,
        )

        self.assertEqual(
            config.models
            .xgb_bagging_seeds,
            (
                2026,
                2027,
                2028,
            ),
        )
        self.assertEqual(
            config
            .submission_gate_previous_2024_raw_brier,
            0.2480388865242645,
        )

        self.assertEqual(
            config
            .submission_gate_previous_late_calibrated_brier,
            0.2477184595089623,
        )

        self.assertEqual(
            config
            .submission_gate_previous_forward_brier,
            0.24798923858583477,
        )

        self.assertEqual(
            config
            .submission_gate_min_forward_improvement,
            2.0e-6,
        )     

        self.assertEqual(
            config
            .ensemble_fixed_champion_weights,
            (
                0.700,
                0.000,
                0.150,
                0.125,
                0.025,
            ),
        )

        self.assertAlmostEqual(
            sum(
                config
                .ensemble_fixed_champion_weights
            ),
            1.0,
            places=12,
        )
        self.assertTrue(
            config.models
            .xgb_multiview_enabled
        )

        self.assertEqual(
            config.models
            .xgb_multiview_top_raw_features,
            92,
        )

        self.assertEqual(
            config.models
            .xgb_multiview_pca_components,
            8,
        )
        
        self.assertTrue(
            config.models
            .xgb_temporal_enabled
        )

        self.assertEqual(
            config.models
            .xgb_temporal_recent1_seasons,
            1,
        )

        self.assertEqual(
            config.models
            .xgb_temporal_recent2_seasons,
            2,
        )

        self.assertEqual(
            config.models
            .xgb_temporal_grid_step,
            0.025,
        )

        self.assertEqual(
            config.models
            .xgb_temporal_minimum_base_weight,
            0.30,
        )

        self.assertEqual(
            config.models
            .xgb_temporal_maximum_aux_weight,
            0.60,
        )

        self.assertEqual(
            (
                config.models
                .xgb_multiview_top_raw_features
                + config.models
                .xgb_multiview_pca_components
            ),
            100,
        )
        self.assertEqual(
            config.models
            .xgb_multiview_group_columns,
            (
                "pitcher_count_combo",
                "pitcher_base_combo",
            ),
        )

        self.assertEqual(
            config.models
            .xgb_multiview_leverage_column,
            "li_log",
        )

        self.assertEqual(
            config.models
            .xgb_multiview_grid_step,
            0.05,
        )

        self.assertEqual(
            config.models
            .xgb_multiview_minimum_base_weight,
            0.45,
        )

        self.assertEqual(
            config.models
            .xgb_multiview_maximum_aux_weight,
            0.40,
        )

        self.assertEqual(
            config.models
            .xgb_multiview_protected_tolerance,
            1.0e-5,
        )        

    def test_neural_training_can_be_disabled_without_changing_model_defaults(
        self,
    ) -> None:
        enabled = NeuralConfig()

        disabled = NeuralConfig(
            enabled=False
        )

        self.assertTrue(
            enabled.enabled
        )
        self.assertFalse(
            disabled.enabled
        )

        self.assertEqual(
            enabled.models,
            (
                "resnet",
                "ft_transformer",
            ),
        )

        self.assertEqual(
            enabled.resnet_max_epochs,
            disabled.resnet_max_epochs,
        )

        self.assertEqual(
            enabled.ft_max_epochs,
            disabled.ft_max_epochs,
        )


if __name__ == "__main__":
    unittest.main()
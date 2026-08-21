from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    # Seraph jobs stage the official archive under /local_datasets and set
    # AIMERS_DATA_DIR. Local work keeps the original repository default.
    configured = os.environ.get("AIMERS_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "baseline" / "data"


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=_default_data_dir)
    output_dir: Path = PROJECT_ROOT / "outputs"
    submission_build_dir: Path = PROJECT_ROOT / "submission_build"
    dist_dir: Path = PROJECT_ROOT / "dist"

    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def trackman_path(self) -> Path:
        return self.data_dir / "trackman_history.csv"

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / "sample_submission.csv"


@dataclass(frozen=True)
class FeatureConfig:
    target_col: str = "control_success"
    id_col: str = "row_id"
    smoothing_prior: float = 0.5
    pitcher_prior_strength: float = 100.0
    batter_prior_strength: float = 50.0
    cold_start_threshold: int = 50
    # Strictly-past cross-season target profiles.  These expose stable,
    # pitcher-specific context effects to every model while retaining the
    # official row-independence contract.  Older seasons are discounted and
    # sparse child contexts are partially pooled toward the pitcher profile.
    history_season_decay: float = 0.70
    history_pitcher_strength: float = 150.0
    history_count_strength: float = 60.0
    history_matchup_strength: float = 90.0
    history_count_matchup_strength: float = 100.0

    # Official-file-only entity resolution between main pitcher_id and the
    # disjoint pitcher_trackman_id namespace.  Ambiguous pairs are rejected;
    # rejected/unseen pitchers retain the existing hand/count Trackman
    # fallback.  Thresholds are label-free and fixed before temporal scoring.
    trackman_entity_enabled: bool = False
    trackman_entity_min_pitches: int = 300
    trackman_entity_max_distance: float = 10.0
    trackman_entity_min_margin_ratio: float = 1.35
    trackman_entity_strong_margin_ratio: float = 2.0
    trackman_entity_min_count_ratio: float = 0.55
    trackman_entity_max_count_ratio: float = 1.80
    trackman_entity_team_penalty: float = 16.0
    trackman_entity_team_min_support: int = 3
    trackman_entity_team_min_dominance: float = 0.50
    # Fail closed if the final 2019--2024 -> 2025 resolver is too sparse or
    # violates its one-to-one contract.  This prevents a silent weak submit.
    trackman_entity_min_matched_pitchers: int = 150
    trackman_entity_min_row_coverage: float = 0.45
    trackman_entity_min_team_mappings: int = 9

    # All are fitted on training rows only. High-cardinality player IDs are
    # intentionally categorical: CatBoost's ordered statistics can use them
    # without hand-written target encoding.
    categorical_cols: Tuple[str, ...] = (
        "game_month",
        "game_dayofweek",
        "top_bottom",
        "game_type",
        "base_state",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_id",
        "batter_id",
        "count_state",
        "stadium_owner_team",
        "pitcher_count_combo",
        "batter_count_combo",
        "pitcher_base_combo",
        "hand_matchup",
        "pitcher_experience_bin",
        "batter_experience_bin",
    )

    # Absolute season cannot extrapolate reliably to an unseen year.  The
    # binary ABS-regime feature remains available, while season/index features
    # are excluded from every submitted model.
    excluded_cols: Tuple[str, ...] = (
        "row_id",
        "control_success",
        "season",
        "season_index",
        "years_since_abs",
    )


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 2026
    num_threads: int = 16

    # ------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 7
    xgb_min_child_weight: float = 30.0
    xgb_subsample: float = 0.90
    xgb_colsample_bytree: float = 0.90
    xgb_reg_lambda: float = 8.0
    xgb_reg_alpha: float = 0.05
    xgb_gamma: float = 0.0
    xgb_max_bin: int = 256
    xgb_device: str = "cpu"
    xgb_num_boost_round: int = 1200
    xgb_early_stopping_rounds: int = 80

    # ------------------------------------------------------------
    # LightGBM
    #
    # Deliberately smoother than XGBoost/CatBoost.
    # We want complementary probability errors, not another copy of XGB.
    # ------------------------------------------------------------
    lgb_learning_rate: float = 0.025
    lgb_num_leaves: int = 31
    lgb_max_depth: int = -1
    lgb_min_data_in_leaf: int = 300
    lgb_feature_fraction: float = 0.90
    lgb_bagging_fraction: float = 0.90
    lgb_bagging_freq: int = 1
    lgb_lambda_l1: float = 0.05
    lgb_lambda_l2: float = 8.0
    lgb_min_gain_to_split: float = 0.0
    lgb_max_bin: int = 255
    lgb_num_boost_round: int = 1600
    lgb_early_stopping_rounds: int = 100
    lgb_device_type: str = "cpu"

    # ------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------
    cat_learning_rate: float = 0.03
    cat_depth: int = 8
    cat_l2_leaf_reg: float = 8.0
    cat_random_strength: float = 0.35
    cat_bootstrap_type: str = "Bayesian"
    cat_bagging_temperature: float = 0.5
    cat_iterations: int = 1200
    cat_early_stopping_rounds: int = 80
    cat_task_type: str = "CPU"


@dataclass(frozen=True)
class NeuralConfig:
    """Training settings for optional out-of-family tabular learners."""

    enabled: bool = True
    # Keep the two neural families that produced the team's strongest real
    # leaderboard result. They have different inductive biases from boosted
    # trees and remain small enough for the DACON L4 inference budget.
    models: Tuple[str, ...] = ("resnet", "ft_transformer")
    device: str = "cuda"
    num_workers: int = 0
    max_grad_norm: float = 1.0

    # ResNet-like MLP: keep the proven 1e-3 AdamW scale, but avoid the very
    # low-update 4096 batch and add regularization for temporal transfer.
    resnet_learning_rate: float = 1e-3
    resnet_weight_decay: float = 1e-3
    resnet_max_epochs: int = 20
    resnet_early_stopping_patience: int = 5
    resnet_min_epochs: int = 4
    resnet_batch_size: int = 2048
    resnet_eval_batch_size: int = 8192
    resnet_d_main: int = 256
    resnet_d_hidden: int = 512
    resnet_n_blocks: int = 3
    resnet_dropout_first: float = 0.25
    resnet_dropout_second: float = 0.10

    # Resource-constrained FT-Transformer. The established default is
    # d_token=192/lr=1e-4; 64 tokens preserve the optimizer/dropout regime
    # while fitting three temporal/full runs into the RTX 3090 budget.
    ft_learning_rate: float = 1e-4
    ft_weight_decay: float = 1e-5
    ft_max_epochs: int = 24
    ft_early_stopping_patience: int = 6
    ft_min_epochs: int = 5
    ft_batch_size: int = 512
    ft_eval_batch_size: int = 2048
    ft_d_token: int = 64
    ft_n_heads: int = 8
    ft_n_layers: int = 3
    ft_d_ffn: int = 96
    ft_attention_dropout: float = 0.20
    ft_ffn_dropout: float = 0.10
    ft_residual_dropout: float = 0.00


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    use_trackman: bool = True
    # The strictly-past target profile is leakage-safe, but the completed
    # 2024 ablation worsened CatBoost by 0.00040154 Brier. Keep the code for
    # controlled experiments and disable it for the next submission.
    use_main_history: bool = False
    temporal_fold_importance: Tuple[float, ...] = (0.05, 0.45, 0.50)
    ensemble_grid_step: float = 0.025
    # An ensemble may optimize the weighted average while regressing on the
    # closest observable regimes. Require it to dominate the best stable
    # single model on both full-2024 and late-2024 before packaging.
    ensemble_protected_folds: Tuple[str, ...] = ("2024", "2024_late_abs")
    ensemble_non_degradation_tolerance: float = 0.0
    temporal_folds: Tuple[Tuple[Tuple[int, ...], int], ...] = (
        ((2019, 2020, 2021, 2022), 2023),
        ((2019, 2020, 2021, 2022, 2023), 2024),
    )
    abs_late_train_month_max: int = 6
    abs_late_valid_months: Tuple[int, ...] = (7, 8, 9)
    # The 895-point branch recorded about 0.24796 on the 2024 holdout.  A
    # reduction near 0.00026 is the score-equivalent improvement needed to
    # reach 1000 from that anchor, so final training is gated at 0.24770.
    submission_gate_2023_max_brier: float = 0.25018
    submission_gate_2024_max_brier: float = 0.24800
    submission_gate_late_min_blend_gain: float = 0.0
    # Paired XGBoost ablation isolates the new physical/entity feature family
    # from model-family and ensemble changes.  Both closest temporal proxies
    # must clear a non-trivial Brier improvement before a zip is produced.
    submission_gate_entity_2024_min_gain: float = 0.00003
    submission_gate_entity_late_min_gain: float = 0.00003

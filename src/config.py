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
    # The residual learner starts from a conservative empirical-Bayes
    # probability instead of relearning the global base rate.  A strong
    # shrinkage level was deliberately chosen because the raw within-season
    # pitcher rate is noisy and the league target mean drifts across years.
    residual_prior_strength: float = 1000.0
    residual_probability_clip: float = 0.02

    # Strictly-past cross-season target profiles.  These expose stable,
    # pitcher-specific context effects to every model while retaining the
    # official row-independence contract.  Older seasons are discounted and
    # sparse child contexts are partially pooled toward the pitcher profile.
    history_season_decay: float = 0.70
    history_pitcher_strength: float = 150.0
    history_count_strength: float = 60.0
    history_matchup_strength: float = 90.0
    history_count_matchup_strength: float = 100.0

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
    # are excluded from both CatBoost and TabM.
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
    # Seraph allocates 16 CPU cores. Submission inference separately caps the
    # PyTorch runtime at the DACON limit of 6.
    num_threads: int = 16

    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 7
    xgb_min_child_weight: float = 30.0
    xgb_subsample: float = 0.90
    xgb_colsample_bytree: float = 0.90
    xgb_reg_lambda: float = 8.0
    xgb_reg_alpha: float = 0.05
    xgb_gamma: float = 0.0
    # QuantileDMatrix and the hist Booster must use exactly the same value.
    # Keep this explicit instead of relying on XGBoost's constructor default.
    xgb_max_bin: int = 256
    xgb_device: str = "cpu"
    xgb_num_boost_round: int = 1200
    xgb_early_stopping_rounds: int = 80

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
    # The next submission uses a single parameter-efficient neural family.
    # Legacy ResNet/FT-Transformer code remains available for controlled
    # ablations, but is not part of the submitted ensemble.
    models: Tuple[str, ...] = ("tabm",)
    device: str = "cuda"
    num_workers: int = 0
    max_grad_norm: float = 1.0

    # TabM-style BatchEnsemble residual MLP.  The model returns k predictions
    # that share the expensive weights but keep per-member adapters and heads.
    # It learns a correction around hierarchical_success_logit.
    tabm_learning_rate: float = 2e-3
    tabm_weight_decay: float = 3e-4
    tabm_max_epochs: int = 12
    tabm_early_stopping_patience: int = 3
    tabm_min_epochs: int = 3
    tabm_batch_size: int = 1024
    tabm_eval_batch_size: int = 4096
    tabm_k: int = 32
    tabm_d_block: int = 256
    tabm_n_blocks: int = 3
    tabm_dropout: float = 0.10
    tabm_offset_col: str = "hierarchical_success_logit"

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
    # 2024 is the first season from the same ABS regime as the hidden 2025
    # target. Keep 2023 as a robustness guard, but optimize the blend mainly
    # for the one-step-ahead 2024 fold.
    # 2024 late-season is the closest observable same-regime proxy for 2025.
    # The full 2024 fold remains a larger-sample stability anchor.
    temporal_fold_importance: Tuple[float, ...] = (0.15, 0.35, 0.50)
    ensemble_grid_step: float = 0.025
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
    submission_gate_2024_max_brier: float = 0.24770
    submission_gate_late_min_blend_gain: float = 0.00005

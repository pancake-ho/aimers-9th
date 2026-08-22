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

    # --------------------------------------------------------
    # XGBoost seed bagging
    #
    # The current dominant model uses subsample and column
    # subsampling, so different seeds produce genuinely
    # different tree ensembles.  Keep the validated V10 seed
    # as the first member and average two additional seeds.
    # --------------------------------------------------------

    xgb_bagging_seeds: Tuple[
        int,
        ...
    ] = (
        2026,
        2027,
        2028,
    )

    # --------------------------------------------------------
    # EXAONE-inspired XGBoost multi-view expert ensemble.
    #
    # All representations are fitted on official training
    # rows only.
    # --------------------------------------------------------

    xgb_multiview_enabled: bool = True

    # 92 raw + 8 PCA = exactly 100 representation features.
    xgb_multiview_top_raw_features: int = 92

    xgb_multiview_pca_components: int = 8

    xgb_multiview_representation_seed: int = 3026

    xgb_multiview_representative_seed: int = 4026

    xgb_multiview_group_columns: Tuple[
        str,
        ...
    ] = (
        "pitcher_count_combo",
        "pitcher_base_combo",
    )

    xgb_multiview_leverage_column: str = (
        "li_log"
    )

    xgb_multiview_leverage_bins: int = 5

    xgb_multiview_repr_weight_clip_low: float = (
        0.50
    )

    xgb_multiview_repr_weight_clip_high: float = (
        2.00
    )

    xgb_multiview_grid_step: float = 0.05

    xgb_multiview_minimum_base_weight: float = (
        0.45
    )

    xgb_multiview_maximum_aux_weight: float = (
        0.40
    )

    # Permit at most 1e-5 Brier loss on either protected
    # future fold while searching for larger complementary gain.
    xgb_multiview_protected_tolerance: float = (
        1.0e-5
    )

    # --------------------------------------------------------
    # V13 temporal-distribution XGBoost experts.
    #
    # Unlike the failed PCA/frequency views, these models
    # intentionally learn the feature -> target mapping from
    # different strictly-past time regimes.
    # --------------------------------------------------------

    xgb_temporal_enabled: bool = True

    xgb_temporal_recent1_seasons: int = 1

    xgb_temporal_recent2_seasons: int = 2

    xgb_temporal_recent1_seed: int = 5026

    xgb_temporal_recent2_seed: int = 6026

    xgb_temporal_grid_step: float = 0.025

    # Aggressive enough to allow a strong recent expert
    # to materially change predictions.
    xgb_temporal_minimum_base_weight: float = (
        0.30
    )

    xgb_temporal_maximum_aux_weight: float = (
        0.60
    )

    # Neither 2024 protected regime may become worse
    # at the logical-XGB layer.
    xgb_temporal_protected_tolerance: float = (
        0.0
    )
    
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
    paths: PathConfig = field(
        default_factory=PathConfig
    )

    features: FeatureConfig = field(
        default_factory=FeatureConfig
    )

    models: ModelConfig = field(
        default_factory=ModelConfig
    )

    neural: NeuralConfig = field(
        default_factory=NeuralConfig
    )

    use_trackman: bool = True

    # Completed ablation worsened temporal Brier.
    # Keep the implementation but exclude it from V10.
    use_main_history: bool = False

    # --------------------------------------------------------
    # Temporal validation
    # --------------------------------------------------------

    temporal_fold_importance: Tuple[
        float,
        ...
    ] = (
        0.05,
        0.45,
        0.50,
    )

    temporal_folds: Tuple[
        Tuple[
            Tuple[int, ...],
            int,
        ],
        ...
    ] = (
        (
            (
                2019,
                2020,
                2021,
                2022,
            ),
            2023,
        ),
        (
            (
                2019,
                2020,
                2021,
                2022,
                2023,
            ),
            2024,
        ),
    )

    abs_late_train_month_max: int = 6

    abs_late_valid_months: Tuple[
        int,
        ...
    ] = (
        7,
        8,
        9,
    )

    # --------------------------------------------------------
    # Raw ensemble search
    # --------------------------------------------------------

    ensemble_grid_step: float = 0.025

    ensemble_protected_folds: Tuple[
        str,
        ...
    ] = (
        "2024",
        "2024_late_abs",
    )

    ensemble_non_degradation_tolerance: float = 0.0

    # --------------------------------------------------------
    # Calibration-aware shrinkage
    #
    # First find the raw temporal-optimal ensemble.
    # Then shrink it toward the most stable model, XGBoost.
    #
    # alpha=0 -> XGB only
    # alpha=1 -> original raw-optimal ensemble
    #
    # Candidate alpha is selected using the actually deployed
    # procedure: raw 2024 + forward-calibrated late-2024 Brier.
    # --------------------------------------------------------

    ensemble_shrinkage_reference_model: str = "xgb"

    ensemble_shrinkage_alphas: Tuple[
        float,
        ...
    ] = (
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
    )

    # --------------------------------------------------------
    # Submission quality gate
    #
    # V10 deliberately replaces the old arbitrary absolute
    # 0.24800 raw-Brier target with forward relative checks.
    #
    # The candidate must not regress against the stable XGB
    # reference and its ABS calibration must transfer.
    # --------------------------------------------------------

    submission_gate_2023_max_brier: float = 0.25018

    submission_gate_2024_min_gain_vs_reference: float = 0.0

    submission_gate_late_raw_min_gain_vs_reference: float = 0.0

    submission_gate_calibration_min_transfer_gain: float = 3.0e-5

    submission_gate_late_calibrated_min_gain_vs_reference: float = 5.0e-5

    # --------------------------------------------------------
    # V11 XGBoost seed-bagging gate.
    #
    # Seed bagging itself must not regress on either of the
    # two closest temporal regimes.
    # --------------------------------------------------------

    submission_gate_xgb_bagging_2024_min_gain: float = 0.0

    submission_gate_xgb_bagging_late_min_gain: float = 0.0

    # V10 champion forward-validation anchor.
    #
    # Frozen before V11 evaluation.
    # Public leaderboard statistics are not used
    # for model fitting, calibration, or gating.
    submission_gate_previous_forward_brier: float = (
        0.24803627124515615
    )

    submission_gate_previous_2024_raw_brier: float = (
        0.24802482024929126
    )

    submission_gate_previous_late_calibrated_brier: float = (
        0.24784041429332387
    )

    submission_gate_previous_2024_min_gain: float = 0.0

    submission_gate_previous_late_calibrated_min_gain: float = (
        0.0
    )

    submission_gate_min_forward_improvement: float = (
        2.0e-6
    )

    # --------------------------------------------------------
    # Trackman entity gates.
    #
    # Entity resolution remains disabled in production V10.
    # Keep these values for the separate entity experiment.
    # --------------------------------------------------------

    submission_gate_entity_2024_min_gain: float = 3.0e-5

    submission_gate_entity_late_min_gain: float = 3.0e-5

    # --------------------------------------------------------
    # V11b frozen champion ensemble.
    #
    # These outer weights are frozen from the previously
    # validated V10 champion before XGBoost seed bagging was
    # evaluated.
    #
    # V11b changes only the internal XGBoost estimator:
    #
    #   single-seed XGB -> 3-seed averaged XGB
    #
    # The outer ensemble allocation is deliberately not
    # re-optimized, preventing the bagging experiment from
    # being confounded by a second weight-search change.
    # --------------------------------------------------------

    ensemble_fixed_champion_weights: Tuple[
        float,
        ...
    ] = (
        0.700,
        0.000,
        0.150,
        0.125,
        0.025,
    )
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

    # season is deliberately retained as a numeric feature. A tree trained
    # through 2024 routes 2025 to the newest side of its temporal splits.
    excluded_cols: Tuple[str, ...] = (
        "row_id",
        "control_success",
    )


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 2026
    num_threads: int = 6

    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 7
    # QuantileDMatrix and the hist Booster must use exactly the same value.
    # Keep this explicit instead of relying on XGBoost's constructor default.
    xgb_max_bin: int = 256
    xgb_device: str = "cpu"
    xgb_num_boost_round: int = 1200
    xgb_early_stopping_rounds: int = 80

    cat_learning_rate: float = 0.03
    cat_depth: int = 8
    cat_iterations: int = 1200
    cat_early_stopping_rounds: int = 80
    cat_task_type: str = "CPU"


@dataclass(frozen=True)
class NeuralConfig:
    """GPU training settings for the two out-of-family tabular learners."""

    device: str = "cuda"
    num_workers: int = 0
    max_epochs: int = 16
    early_stopping_patience: int = 4
    min_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    resnet_batch_size: int = 4096
    resnet_eval_batch_size: int = 8192
    resnet_d_main: int = 256
    resnet_d_hidden: int = 512
    resnet_n_blocks: int = 4
    resnet_dropout_first: float = 0.20
    resnet_dropout_second: float = 0.10

    ft_batch_size: int = 512
    ft_eval_batch_size: int = 2048
    ft_d_token: int = 32
    ft_n_heads: int = 4
    ft_n_layers: int = 3
    ft_d_ffn: int = 64
    ft_attention_dropout: float = 0.15
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
    temporal_fold_importance: Tuple[float, ...] = (0.20, 0.80)
    ensemble_grid_step: float = 0.05
    temporal_folds: Tuple[Tuple[Tuple[int, ...], int], ...] = (
        ((2019, 2020, 2021, 2022), 2023),
        ((2019, 2020, 2021, 2022, 2023), 2024),
    )

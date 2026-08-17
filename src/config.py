from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "baseline" / "data"
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
    xgb_num_boost_round: int = 1200
    xgb_early_stopping_rounds: int = 80

    cat_learning_rate: float = 0.03
    cat_depth: int = 8
    cat_iterations: int = 1200
    cat_early_stopping_rounds: int = 80
    cat_task_type: str = "CPU"


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    use_trackman: bool = True
    temporal_folds: Tuple[Tuple[Tuple[int, ...], int], ...] = (
        ((2019, 2020, 2021, 2022), 2023),
        ((2019, 2020, 2021, 2022, 2023), 2024),
    )

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
    random_seed: int = 2026
    pitcher_prior_strength: float = 100.0
    batter_prior_strength: float = 50.0

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
        "count_state",
        "stadium_owner_team",
    )

    excluded_cols: Tuple[str, ...] = (
        "row_id",
        "control_success",
        "season",
        "pitcher_id",
        "batter_id",
    )


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 2026
    learning_rate: float = 0.03
    lgb_num_leaves: int = 63
    lgb_num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    num_threads: int = 6


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    recency_lambda: float = 0.0
    use_trackman: bool = True
    temporal_folds: Tuple[Tuple[Tuple[int, ...], int], ...] = (
        ((2019, 2020, 2021, 2022), 2023),
        ((2019, 2020, 2021, 2022, 2023), 2024),
    )

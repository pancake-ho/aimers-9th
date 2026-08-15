# src/config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "baseline" / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"

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

    # Empirical-Bayes shrinkage strength.
    pitcher_prior_strength: float = 100.0
    batter_prior_strength: float = 50.0

    # KMeans is fitted only on the training side of each temporal fold.
    pitcher_cluster_k: int = 5
    random_seed: int = 2026

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
        "pitcher_style_cluster",
    )

    # ID를 첫 실험에서는 의도적으로 제외.
    #
    # pitcher_id / batter_id를 단순 categorical로 넣는 실험은
    # temporal benchmark를 만든 뒤 별도 ablation에서 수행한다.
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

    xgb_num_boost_round: int = 1000
    lgb_num_boost_round: int = 1000
    cat_iterations: int = 1000

    early_stopping_rounds: int = 50

    learning_rate: float = 0.03

    xgb_max_depth: int = 6
    lgb_num_leaves: int = 63
    cat_depth: int = 6


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    # 첫 benchmark는 반드시 0.0으로 실행.
    #
    # 이후 0.1 / 0.25 / 0.5 등을 동일 split에서 ablation.
    recency_lambda: float = 0.0

    use_trackman: bool = True

    # 공식 문제의 핵심 temporal benchmark
    temporal_folds: Tuple[Tuple[Tuple[int, ...], int], ...] = (
        ((2019, 2020, 2021, 2022), 2023),
        ((2019, 2020, 2021, 2022, 2023), 2024),
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_seasons: Tuple[int, ...]
    valid_season: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    validation_label: str = ""


def make_temporal_folds(
    df: pd.DataFrame,
    fold_specs: Sequence[Tuple[Tuple[int, ...], int]],
) -> Iterator[TemporalFold]:
    seasons = pd.to_numeric(df["season"], errors="raise").to_numpy()
    for train_seasons, valid_season in fold_specs:
        if max(train_seasons) >= valid_season:
            raise ValueError("Temporal leakage: train season must precede validation.")
        train_idx = np.flatnonzero(np.isin(seasons, np.asarray(train_seasons)))
        valid_idx = np.flatnonzero(seasons == valid_season)
        if not len(train_idx) or not len(valid_idx):
            raise ValueError(
                f"Empty temporal fold: train={train_seasons}, valid={valid_season}"
            )
        yield TemporalFold(
            name=f"train_{min(train_seasons)}_{max(train_seasons)}_valid_{valid_season}",
            train_seasons=tuple(train_seasons),
            valid_season=int(valid_season),
            train_idx=train_idx,
            valid_idx=valid_idx,
            validation_label=str(valid_season),
        )


def make_abs_late_fold(
    df: pd.DataFrame,
    train_month_max: int = 6,
    valid_months: Sequence[int] = (7, 8, 9),
) -> TemporalFold:
    """Create a same-regime forward holdout inside the 2024 ABS season.

    Earlier seasons and January--June 2024 are training data. Validation uses
    July--September 2024. October is intentionally omitted because its small,
    postseason-heavy sample is not representative of the hidden full season.
    """
    season = pd.to_numeric(df["season"], errors="raise").to_numpy()
    month = pd.to_numeric(df["game_month"], errors="raise").to_numpy()
    train_mask = (season < 2024) | ((season == 2024) & (month <= int(train_month_max)))
    valid_mask = (season == 2024) & np.isin(month, np.asarray(tuple(valid_months)))
    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)
    if not len(train_idx) or not len(valid_idx):
        raise ValueError("Empty ABS late-season temporal fold.")
    if np.intersect1d(train_idx, valid_idx).size:
        raise ValueError("ABS late-season fold has overlapping train/validation rows.")
    return TemporalFold(
        name=(
            f"train_2019_2024m{int(train_month_max):02d}_"
            + "valid_2024m"
            + "_".join(str(int(value)) for value in valid_months)
        ),
        train_seasons=(2019, 2020, 2021, 2022, 2023, 2024),
        valid_season=2024,
        train_idx=train_idx,
        valid_idx=valid_idx,
        validation_label="2024_late_abs",
    )


def uniform_weights(length: int) -> np.ndarray:
    return np.ones(int(length), dtype=np.float32)

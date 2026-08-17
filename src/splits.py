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
        )


def uniform_weights(length: int) -> np.ndarray:
    return np.ones(int(length), dtype=np.float32)

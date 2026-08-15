# src/splits.py

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
    fold_specs: Sequence[
        Tuple[Tuple[int, ...], int]
    ],
) -> Iterator[TemporalFold]:

    seasons = df["season"].to_numpy()

    for train_seasons, valid_season in fold_specs:

        train_mask = np.isin(
            seasons,
            np.asarray(train_seasons),
        )

        valid_mask = seasons == valid_season

        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)

        if len(train_idx) == 0:
            raise ValueError(
                f"No training rows for seasons={train_seasons}"
            )

        if len(valid_idx) == 0:
            raise ValueError(
                f"No validation rows for season={valid_season}"
            )

        if max(train_seasons) >= valid_season:
            raise ValueError(
                "Temporal leakage: train season must be "
                "strictly earlier than valid season."
            )

        yield TemporalFold(
            name=f"train_{min(train_seasons)}_{max(train_seasons)}"
                 f"_valid_{valid_season}",
            train_seasons=train_seasons,
            valid_season=valid_season,
            train_idx=train_idx,
            valid_idx=valid_idx,
        )


def make_recency_weights(
    seasons: np.ndarray,
    max_train_season: int,
    decay_lambda: float,
) -> np.ndarray:
    """
    w(t) = exp(-lambda * (max_train_season - season))

    lambda=0 -> every observation has weight 1.
    """

    seasons = np.asarray(
        seasons,
        dtype=np.float32,
    )

    age = (
        float(max_train_season)
        - seasons
    )

    weights = np.exp(
        -decay_lambda * age
    ).astype(np.float32)

    return weights
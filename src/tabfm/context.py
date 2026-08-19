from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def tabdpt_feature_names(
    numerical_columns: Sequence[str],
    *,
    max_features: int = 128,
) -> list[str]:
    """Return the stable numerical view consumed by TabDPT-Turbo.

    High-cardinality player and combination columns are ordinal-coded for
    XGBoost/CatBoost.  Euclidean or linear row tokenizers would interpret
    those arbitrary codes as distances, so the foundation-model path uses
    only the numerical features.  Official ``asof_*`` rates and Trackman
    profiles still carry player ability and recent-form information.
    """
    names = list(numerical_columns)
    if not names:
        raise ValueError("TabDPT feature view cannot be empty.")
    if len(names) > int(max_features):
        raise ValueError(
            f"TabDPT numerical view has {len(names)} features; "
            f"the declared maximum is {max_features}. Refusing silent PCA."
        )
    if len(names) != len(set(names)):
        raise ValueError("TabDPT feature names contain duplicates.")
    return names


def to_tabdpt_array(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"TabDPT frame is missing features: {missing}")
    array = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float32, copy=True)
    if array.ndim != 2 or array.shape[1] != len(feature_names):
        raise RuntimeError(f"Unexpected TabDPT array shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("TabDPT array contains NaN or infinity after preprocessing.")
    return np.ascontiguousarray(array)


def _proportional_stratified_sample(
    indices: np.ndarray,
    strata: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Sample without changing the observed month/target proportions.

    Target-aware *sampling* is legal training-time subsampling.  Allocation is
    proportional, not class balancing, so the recent-season base rate is not
    overwritten.  The returned order is shuffled deterministically.
    """
    if sample_size >= len(indices):
        return np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(int(seed))
    unique, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    expected = counts.astype(np.float64) * float(sample_size) / float(len(indices))
    allocation = np.floor(expected).astype(np.int64)
    allocation = np.minimum(allocation, counts)

    remainder = int(sample_size - allocation.sum())
    fractional_order = np.argsort(-(expected - allocation), kind="stable")
    for position in fractional_order:
        if remainder <= 0:
            break
        if allocation[position] < counts[position]:
            allocation[position] += 1
            remainder -= 1

    selected_parts: list[np.ndarray] = []
    for stratum_position, _ in enumerate(unique):
        candidates = indices[inverse == stratum_position]
        take = int(allocation[stratum_position])
        if take:
            selected_parts.append(rng.choice(candidates, size=take, replace=False))

    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    if len(selected) < sample_size:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        fill = rng.choice(remaining, size=sample_size - len(selected), replace=False)
        selected = np.concatenate([selected, fill.astype(np.int64, copy=False)])
    if len(selected) != sample_size or len(np.unique(selected)) != sample_size:
        raise RuntimeError("Recent-context sampler produced an invalid sample.")
    rng.shuffle(selected)
    return selected


def select_recent_context_indices(
    train: pd.DataFrame,
    allowed_indices: Sequence[int] | np.ndarray,
    *,
    target_col: str,
    context_size: int,
    seed: int,
) -> np.ndarray:
    """Select a recent, proportional and leakage-safe ICL context.

    Only the latest season present on the training side of a temporal fold is
    eligible.  Consequently the validation contexts are 2022 -> 2023,
    2023 -> 2024 and Jan-Jun 2024 -> late-2024.  The final 2025 context comes
    exclusively from 2024.
    """
    allowed = np.asarray(allowed_indices, dtype=np.int64)
    if allowed.ndim != 1 or not len(allowed):
        raise ValueError("allowed_indices must be a non-empty vector.")
    if len(np.unique(allowed)) != len(allowed):
        raise ValueError("allowed_indices contains duplicates.")

    subset = train.iloc[allowed]
    season = pd.to_numeric(subset["season"], errors="raise").to_numpy(dtype=np.int64)
    latest_season = int(season.max())
    recent = allowed[season == latest_season]
    if not len(recent):
        raise RuntimeError("No latest-season rows available for TabDPT context.")

    recent_frame = train.iloc[recent]
    month = pd.to_numeric(recent_frame["game_month"], errors="raise").to_numpy(dtype=np.int64)
    target = pd.to_numeric(recent_frame[target_col], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("TabDPT context target must be binary.")
    strata = month * 2 + target
    selected = _proportional_stratified_sample(
        recent,
        strata,
        min(int(context_size), len(recent)),
        int(seed),
    )
    if not np.isin(selected, allowed).all():
        raise RuntimeError("TabDPT sampler crossed the training-fold boundary.")
    return selected

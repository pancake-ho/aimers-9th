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


def _sqrt_quota_sample(
    indices: np.ndarray,
    groups: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Sample groups with sqrt-frequency allocation.

    Large groups still receive more rows, but their dominance is compressed.
    Selection is strictly label-free; `groups` must be derived from
    pre-pitch/training-side metadata only.
    """
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    groups = np.asarray(groups)

    if indices.ndim != 1:
        raise ValueError(
            "indices must be one-dimensional."
        )

    if len(indices) != len(groups):
        raise ValueError(
            "indices/groups length mismatch."
        )

    if sample_size <= 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    if sample_size >= len(indices):
        return indices.copy()

    rng = np.random.default_rng(
        int(seed)
    )

    unique, inverse, counts = np.unique(
        groups,
        return_inverse=True,
        return_counts=True,
    )

    weights = np.sqrt(
        counts.astype(np.float64)
    )

    expected = (
        weights
        / weights.sum()
        * float(sample_size)
    )

    allocation = np.floor(
        expected
    ).astype(np.int64)

    allocation = np.minimum(
        allocation,
        counts,
    )

    remaining = int(
        sample_size - allocation.sum()
    )

    # Largest fractional quota first.
    fractional = (
        expected - allocation
    )

    while remaining > 0:
        available = (
            allocation < counts
        )

        if not available.any():
            break

        priority = np.where(
            available,
            fractional,
            -np.inf,
        )

        group_pos = int(
            np.argmax(priority)
        )

        allocation[group_pos] += 1
        fractional[group_pos] = 0.0
        remaining -= 1

    selected_parts: list[np.ndarray] = []

    for group_pos in range(
        len(unique)
    ):
        take = int(
            allocation[group_pos]
        )

        if take <= 0:
            continue

        candidates = indices[
            inverse == group_pos
        ]

        chosen = rng.choice(
            candidates,
            size=take,
            replace=False,
        )

        selected_parts.append(
            np.asarray(
                chosen,
                dtype=np.int64,
            )
        )

    if selected_parts:
        selected = np.concatenate(
            selected_parts
        )
    else:
        selected = np.empty(
            0,
            dtype=np.int64,
        )

    if len(selected) < sample_size:
        unused = indices[
            ~np.isin(
                indices,
                selected,
            )
        ]

        fill = rng.choice(
            unused,
            size=(
                sample_size
                - len(selected)
            ),
            replace=False,
        )

        selected = np.concatenate(
            [
                selected,
                np.asarray(
                    fill,
                    dtype=np.int64,
                ),
            ]
        )

    if (
        len(selected) != sample_size
        or len(np.unique(selected))
        != sample_size
    ):
        raise RuntimeError(
            "sqrt-quota sampler "
            "produced invalid selection."
        )

    rng.shuffle(selected)

    return selected


def _latest_season_pool(
    train: pd.DataFrame,
    allowed_indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    allowed = np.asarray(
        allowed_indices,
        dtype=np.int64,
    )

    subset = train.iloc[allowed]

    seasons = pd.to_numeric(
        subset["season"],
        errors="raise",
    ).to_numpy(
        dtype=np.int64,
    )

    latest_season = int(
        seasons.max()
    )

    latest = allowed[
        seasons == latest_season
    ]

    if not len(latest):
        raise RuntimeError(
            "No latest-season rows available."
        )

    return latest, latest_season


def _situation_groups(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Build fixed pre-pitch situation strata.

    No target or validation/test distribution statistic is used.
    """
    balls = pd.to_numeric(
        frame["balls_before"],
        errors="raise",
    ).to_numpy(
        dtype=np.int16,
    )

    strikes = pd.to_numeric(
        frame["strikes_before"],
        errors="raise",
    ).to_numpy(
        dtype=np.int16,
    )

    base_state = (
        frame["base_state"]
        .fillna("__MISSING__")
        .astype(str)
        .to_numpy()
    )

    li = pd.to_numeric(
        frame["li"],
        errors="coerce",
    ).fillna(
        1.0
    ).to_numpy(
        dtype=np.float64,
    )

    # Fixed baseball-reasonable bins.
    # These thresholds are not fitted from validation/test data.
    leverage_bin = np.where(
        li < 0.75,
        0,
        np.where(
            li < 1.50,
            1,
            2,
        ),
    )

    return np.asarray(
        [
            (
                f"{b}:{s}:"
                f"{base}:"
                f"{lev}"
            )
            for b, s, base, lev
            in zip(
                balls,
                strikes,
                base_state,
                leverage_bin,
            )
        ],
        dtype=object,
    )


def select_representative_context_indices(
    train: pd.DataFrame,
    allowed_indices: Sequence[int] | np.ndarray,
    *,
    target_col: str,
    context_size: int,
    seed: int,
    recent_fraction: float = 0.70,
    pitcher_fraction: float = 0.20,
    situation_fraction: float = 0.10,
) -> np.ndarray:
    """Select a latest-season representative TabDPT context.

    Components:
      1. proportional month/target recent sample,
      2. sqrt-frequency pitcher coverage,
      3. sqrt-frequency game-situation coverage.

    Validation/test rows never participate.
    """
    allowed = np.asarray(
        allowed_indices,
        dtype=np.int64,
    )

    if (
        allowed.ndim != 1
        or not len(allowed)
    ):
        raise ValueError(
            "allowed_indices must be "
            "a non-empty vector."
        )

    if (
        len(np.unique(allowed))
        != len(allowed)
    ):
        raise ValueError(
            "allowed_indices contains duplicates."
        )

    fractions = np.asarray(
        [
            recent_fraction,
            pitcher_fraction,
            situation_fraction,
        ],
        dtype=np.float64,
    )

    if (
        (fractions < 0.0).any()
        or not np.isfinite(
            fractions
        ).all()
        or not np.isclose(
            fractions.sum(),
            1.0,
        )
    ):
        raise ValueError(
            "Representative context fractions "
            "must be finite, non-negative, "
            "and sum to one."
        )

    latest, latest_season = (
        _latest_season_pool(
            train,
            allowed,
        )
    )

    sample_size = min(
        int(context_size),
        len(latest),
    )

    n_recent = int(
        round(
            sample_size
            * float(recent_fraction)
        )
    )

    n_pitcher = int(
        round(
            sample_size
            * float(pitcher_fraction)
        )
    )

    n_recent = min(
        n_recent,
        sample_size,
    )

    n_pitcher = min(
        n_pitcher,
        sample_size - n_recent,
    )

    n_situation = (
        sample_size
        - n_recent
        - n_pitcher
    )


    # --------------------------------------------------------
    # 1. Existing proportional recent sampler.
    #
    # Target is used only to preserve the observed latest-season
    # class proportion, never to rebalance it.
    # --------------------------------------------------------

    latest_frame = train.iloc[
        latest
    ]

    month = pd.to_numeric(
        latest_frame["game_month"],
        errors="raise",
    ).to_numpy(
        dtype=np.int64,
    )

    target = pd.to_numeric(
        latest_frame[target_col],
        errors="raise",
    ).to_numpy(
        dtype=np.int64,
    )

    if not set(
        np.unique(target)
    ).issubset(
        {0, 1}
    ):
        raise ValueError(
            "TabDPT context target "
            "must be binary."
        )

    recent_strata = (
        month * 2 + target
    )

    recent_selected = (
        _proportional_stratified_sample(
            latest,
            recent_strata,
            n_recent,
            int(seed),
        )
    )


    # --------------------------------------------------------
    # 2. Pitcher coverage.
    # --------------------------------------------------------

    remaining = latest[
        ~np.isin(
            latest,
            recent_selected,
        )
    ]

    pitcher_frame = train.iloc[
        remaining
    ]

    pitcher_groups = (
        pitcher_frame["pitcher_id"]
        .fillna(-1)
        .astype(str)
        .to_numpy()
    )

    pitcher_take = min(
        n_pitcher,
        len(remaining),
    )

    pitcher_selected = (
        _sqrt_quota_sample(
            remaining,
            pitcher_groups,
            pitcher_take,
            int(seed) + 1,
        )
    )


    # --------------------------------------------------------
    # 3. Count/base/leverage situation coverage.
    # --------------------------------------------------------

    remaining = remaining[
        ~np.isin(
            remaining,
            pitcher_selected,
        )
    ]

    situation_take = min(
        n_situation,
        len(remaining),
    )

    situation_groups = (
        _situation_groups(
            train.iloc[remaining]
        )
    )

    situation_selected = (
        _sqrt_quota_sample(
            remaining,
            situation_groups,
            situation_take,
            int(seed) + 2,
        )
    )


    selected = np.concatenate(
        [
            recent_selected,
            pitcher_selected,
            situation_selected,
        ]
    ).astype(
        np.int64,
        copy=False,
    )


    # --------------------------------------------------------
    # Fill only when one component lacked enough rows.
    # --------------------------------------------------------

    if len(selected) < sample_size:
        leftover = latest[
            ~np.isin(
                latest,
                selected,
            )
        ]

        rng = np.random.default_rng(
            int(seed) + 3
        )

        fill = rng.choice(
            leftover,
            size=(
                sample_size
                - len(selected)
            ),
            replace=False,
        )

        selected = np.concatenate(
            [
                selected,
                np.asarray(
                    fill,
                    dtype=np.int64,
                ),
            ]
        )


    if (
        len(selected) != sample_size
        or len(np.unique(selected))
        != sample_size
    ):
        raise RuntimeError(
            "Representative context "
            "has invalid size/duplicates."
        )

    if not np.isin(
        selected,
        allowed,
    ).all():
        raise RuntimeError(
            "Representative context crossed "
            "the fold training boundary."
        )

    selected_seasons = pd.to_numeric(
        train.iloc[selected]["season"],
        errors="raise",
    ).to_numpy(
        dtype=np.int64,
    )

    if not np.all(
        selected_seasons
        == latest_season
    ):
        raise RuntimeError(
            "Representative context escaped "
            "the latest training season."
        )

    rng = np.random.default_rng(
        int(seed) + 4
    )

    rng.shuffle(selected)

    return selected
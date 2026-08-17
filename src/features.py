from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import FeatureConfig
from src.runtime import FEATURE_VERSION, HAND_MAP, build_features


TRACKMAN_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
)

TRACKMAN_STD_METRICS = (
    "rel_speed",
    "spin_rate",
)

PITCH_GROUPS = (
    "fastball",
    "breaking",
    "offspeed",
    "other",
)


def _finite_or_zero(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _summarize_profile(
    past: pd.DataFrame,
    group_cols: Sequence[str],
    prefix: str,
) -> tuple[Dict[tuple, Dict[str, float]], list[str]]:
    grouped = past.groupby(list(group_cols), observed=True, sort=False)
    means = grouped[list(TRACKMAN_METRICS)].mean()
    stds = grouped[list(TRACKMAN_STD_METRICS)].std()
    counts = grouped.size()
    pitch_shares = grouped[list(PITCH_GROUPS)].mean()

    feature_names = [
        *(f"{prefix}_{metric}_mean" for metric in TRACKMAN_METRICS),
        *(f"{prefix}_{metric}_std" for metric in TRACKMAN_STD_METRICS),
        *(f"{prefix}_{group}_share" for group in PITCH_GROUPS),
        f"{prefix}_log_n",
        f"{prefix}_available",
    ]
    lookup: Dict[tuple, Dict[str, float]] = {}

    for raw_key in means.index:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        values: Dict[str, float] = {}
        for metric in TRACKMAN_METRICS:
            values[f"{prefix}_{metric}_mean"] = _finite_or_zero(means.loc[raw_key, metric])
        for metric in TRACKMAN_STD_METRICS:
            values[f"{prefix}_{metric}_std"] = _finite_or_zero(stds.loc[raw_key, metric])
        for group in PITCH_GROUPS:
            values[f"{prefix}_{group}_share"] = _finite_or_zero(pitch_shares.loc[raw_key, group])
        values[f"{prefix}_log_n"] = float(np.log1p(counts.loc[raw_key]))
        values[f"{prefix}_available"] = 1.0
        lookup[tuple(int(v) for v in key)] = values

    return lookup, feature_names


class StrictPastTrackmanFeatures:
    """Strictly-past, non-target Trackman profiles.

    For a main-table row from season S, every Trackman value is aggregated
    only from seasons < S. The feature uses handedness and pre-pitch count;
    it never assumes that main pitcher_id equals pitcher_trackman_id and never
    reads another evaluation row.
    """

    def __init__(self) -> None:
        self.state_: Dict[str, object] | None = None

    def fit_from_csv(self, trackman_path: Path | str) -> "StrictPastTrackmanFeatures":
        print("[FEATURE] Building strictly-past Trackman profiles...")
        usecols = [
            "season",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
            "pitch_type_group",
            *TRACKMAN_METRICS,
        ]
        tm = pd.read_csv(trackman_path, usecols=usecols, low_memory=False)
        tm["pitcher_hand"] = tm["pitcher_hand"].map(HAND_MAP)
        tm["batter_hand"] = tm["batter_hand"].map(HAND_MAP)
        tm = tm.dropna(subset=["season", "pitcher_hand", "batter_hand"])
        tm["season"] = pd.to_numeric(tm["season"], errors="coerce").astype(np.int16)
        tm["pitcher_hand"] = tm["pitcher_hand"].astype(np.int8)
        tm["batter_hand"] = tm["batter_hand"].astype(np.int8)
        tm["balls_before"] = pd.to_numeric(tm["balls_before"], errors="coerce").fillna(-1).astype(np.int8)
        tm["strikes_before"] = pd.to_numeric(tm["strikes_before"], errors="coerce").fillna(-1).astype(np.int8)

        for metric in TRACKMAN_METRICS:
            tm[metric] = pd.to_numeric(tm[metric], errors="coerce").astype(np.float32)

        pitch_group = tm["pitch_type_group"].fillna("other").astype(str).str.lower()
        for group in PITCH_GROUPS:
            tm[group] = (pitch_group == group).astype(np.float32)
        tm.drop(columns=["pitch_type_group"], inplace=True)

        min_season = int(tm["season"].min())
        max_season = int(tm["season"].max())
        hand_lookup: Dict[tuple, Dict[str, float]] = {}
        count_lookup: Dict[tuple, Dict[str, float]] = {}
        hand_names: list[str] = []
        count_names: list[str] = []

        for target_season in range(min_season, max_season + 2):
            past = tm.loc[tm["season"] < target_season]
            if past.empty:
                continue

            per_hand, hand_names = _summarize_profile(
                past,
                group_cols=("pitcher_hand", "batter_hand"),
                prefix="tm_hand",
            )
            for key, values in per_hand.items():
                hand_lookup[(target_season, *key)] = values

            per_count, count_names = _summarize_profile(
                past,
                group_cols=(
                    "pitcher_hand",
                    "batter_hand",
                    "balls_before",
                    "strikes_before",
                ),
                prefix="tm_count",
            )
            for key, values in per_count.items():
                count_lookup[(target_season, *key)] = values

        self.state_ = {
            "state_version": 1,
            "source_seasons": [min_season, max_season],
            "profiles": {
                "hand": {
                    "lookup": hand_lookup,
                    "feature_names": hand_names,
                    "defaults": {name: 0.0 for name in hand_names},
                },
                "hand_count": {
                    "lookup": count_lookup,
                    "feature_names": count_names,
                    "defaults": {name: 0.0 for name in count_names},
                },
            },
        }
        del tm
        print(
            f"[FEATURE] Trackman keys: hand={len(hand_lookup):,}, "
            f"hand_count={len(count_lookup):,}"
        )
        return self

    def export_state(self):
        if self.state_ is None:
            raise RuntimeError("Trackman feature builder has not been fitted.")
        return self.state_


class LeakageSafeFeatureEngineer:
    def __init__(
        self,
        config: FeatureConfig,
        trackman_features: StrictPastTrackmanFeatures | None = None,
    ) -> None:
        self.config = config
        self.trackman_features = trackman_features

    def fit(self, df: pd.DataFrame, y=None) -> "LeakageSafeFeatureEngineer":
        # All transformations are fixed or strictly-past; no label-dependent
        # state is fitted here.
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.export_runtime_state())

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def export_runtime_state(self) -> Dict[str, object]:
        return {
            "feature_version": FEATURE_VERSION,
            "smoothing_prior": float(self.config.smoothing_prior),
            "pitcher_prior_strength": float(self.config.pitcher_prior_strength),
            "batter_prior_strength": float(self.config.batter_prior_strength),
            "cold_start_threshold": int(self.config.cold_start_threshold),
            "trackman": (
                None
                if self.trackman_features is None
                else self.trackman_features.export_state()
            ),
        }

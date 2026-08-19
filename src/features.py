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


class StrictPastMainHistoryFeatures:
    """Season-prequential pitcher context profiles from the labeled table.

    A row from season S can only receive statistics computed from seasons
    strictly earlier than S.  The hidden 2025 rows therefore use 2019--2024
    labels, while the 2023 and 2024 validation folds remain uncontaminated.
    All lookups are keyed by the current row only; no evaluation-row statistic
    is ever computed.

    The design follows a partial-pooling hierarchy:
      league -> pitcher -> pitcher x count / handedness context.
    Only centered effects are exported.  This removes the league-wide target
    drift while retaining an individual pitcher's persistent context signal.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.state_: Dict[str, object] | None = None

    @staticmethod
    def _normalize_id(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int64)

    @staticmethod
    def _raw_key(raw_key) -> tuple[int, ...]:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        return tuple(int(value) for value in key)

    def _summarize_target_season(
        self,
        past: pd.DataFrame,
        target_season: int,
        group_cols: Sequence[str],
        prefix: str,
        strength: float,
        league_prior: float,
        pitcher_parent: Mapping[tuple, float] | None,
    ) -> tuple[Dict[tuple, Dict[str, float]], Dict[tuple, float], list[str]]:
        grouped = past.groupby(list(group_cols), observed=True, sort=False)
        stats = grouped.agg(
            weighted_y=("weighted_y", "sum"),
            weighted_n=("season_weight", "sum"),
            raw_n=("target", "size"),
        )

        is_child = pitcher_parent is not None
        feature_names = [f"{prefix}_effect"]
        if is_child:
            feature_names.append(f"{prefix}_delta")
        feature_names.extend(
            [f"{prefix}_log_n", f"{prefix}_available"]
        )
        lookup: Dict[tuple, Dict[str, float]] = {}
        rate_lookup: Dict[tuple, float] = {}

        for raw_key, row in stats.iterrows():
            key = self._raw_key(raw_key)
            parent = (
                float(pitcher_parent.get((key[0],), league_prior))
                if is_child
                else float(league_prior)
            )
            rate = (
                float(row["weighted_y"]) + float(strength) * parent
            ) / (float(row["weighted_n"]) + float(strength))
            values = {
                f"{prefix}_effect": float(rate - league_prior),
                f"{prefix}_log_n": float(np.log1p(row["raw_n"])),
                f"{prefix}_available": 1.0,
            }
            if is_child:
                values[f"{prefix}_delta"] = float(rate - parent)
            lookup[(target_season, *key)] = values
            rate_lookup[key] = float(rate)

        return lookup, rate_lookup, feature_names

    def fit(self, train: pd.DataFrame, target_col: str) -> "StrictPastMainHistoryFeatures":
        required = {
            "season",
            "pitcher_id",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
            target_col,
        }
        missing = required - set(train.columns)
        if missing:
            raise ValueError(f"Missing main-history columns: {sorted(missing)}")

        print("[FEATURE] Building strictly-past pitcher context profiles...")
        pitcher_hand = train["pitcher_hand"].map(HAND_MAP)
        pitcher_hand = pitcher_hand.fillna(
            pd.to_numeric(train["pitcher_hand"], errors="coerce")
        )
        batter_hand = train["batter_hand"].map(HAND_MAP)
        batter_hand = batter_hand.fillna(
            pd.to_numeric(train["batter_hand"], errors="coerce")
        )

        history = pd.DataFrame(
            {
                "season": pd.to_numeric(train["season"], errors="raise").astype(np.int16),
                "pitcher_id": self._normalize_id(train["pitcher_id"]),
                "balls": pd.to_numeric(train["balls_before"], errors="coerce")
                .fillna(-1)
                .astype(np.int8),
                "strikes": pd.to_numeric(train["strikes_before"], errors="coerce")
                .fillna(-1)
                .astype(np.int8),
                "same_hand": (pitcher_hand == batter_hand).fillna(False).astype(np.int8),
                "target": pd.to_numeric(train[target_col], errors="raise").astype(np.float32),
            }
        )

        profile_specs = (
            (
                "pitcher",
                ("pitcher_id",),
                float(self.config.history_pitcher_strength),
            ),
            (
                "pitcher_count",
                ("pitcher_id", "balls", "strikes"),
                float(self.config.history_count_strength),
            ),
            (
                "pitcher_matchup",
                ("pitcher_id", "same_hand"),
                float(self.config.history_matchup_strength),
            ),
            (
                "pitcher_count_matchup",
                ("pitcher_id", "balls", "strikes", "same_hand"),
                float(self.config.history_count_matchup_strength),
            ),
        )

        profiles: Dict[str, Dict[str, object]] = {
            name: {"lookup": {}, "feature_names": [], "defaults": {}}
            for name, _, _ in profile_specs
        }
        league_priors: Dict[int, float] = {}
        min_season = int(history["season"].min())
        max_season = int(history["season"].max())

        for target_season in range(min_season, max_season + 2):
            past = history.loc[history["season"] < target_season].copy()
            if past.empty:
                continue
            age = (target_season - 1 - past["season"]).clip(lower=0)
            past["season_weight"] = np.power(
                float(self.config.history_season_decay),
                age.to_numpy(dtype=np.float64, copy=False),
            )
            past["weighted_y"] = (
                past["season_weight"] * past["target"].astype(np.float64)
            )
            league_prior = float(past["weighted_y"].sum() / past["season_weight"].sum())
            league_priors[target_season] = league_prior

            pitcher_parent: Dict[tuple, float] | None = None
            for name, group_cols, strength in profile_specs:
                lookup, rates, feature_names = self._summarize_target_season(
                    past=past,
                    target_season=target_season,
                    group_cols=group_cols,
                    prefix=f"hist_{name}",
                    strength=strength,
                    league_prior=league_prior,
                    pitcher_parent=pitcher_parent if name != "pitcher" else None,
                )
                profiles[name]["lookup"].update(lookup)
                profiles[name]["feature_names"] = feature_names
                profiles[name]["defaults"] = {feature: 0.0 for feature in feature_names}
                if name == "pitcher":
                    pitcher_parent = rates

        self.state_ = {
            "state_version": 1,
            "source_seasons": [min_season, max_season],
            "season_decay": float(self.config.history_season_decay),
            "league_priors": league_priors,
            "profiles": profiles,
        }
        print(
            "[FEATURE] Main-history keys: "
            + ", ".join(
                f"{name}={len(profile['lookup']):,}"
                for name, profile in profiles.items()
            )
        )
        return self

    def export_state(self) -> Dict[str, object]:
        if self.state_ is None:
            raise RuntimeError("Main-history feature builder has not been fitted.")
        return self.state_


class LeakageSafeFeatureEngineer:
    def __init__(
        self,
        config: FeatureConfig,
        trackman_features: StrictPastTrackmanFeatures | None = None,
        main_history_features: StrictPastMainHistoryFeatures | None = None,
    ) -> None:
        self.config = config
        self.trackman_features = trackman_features
        self.main_history_features = main_history_features

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
            "main_history": (
                None
                if self.main_history_features is None
                else self.main_history_features.export_state()
            ),
        }


def validate_feature_config_contract(config: FeatureConfig) -> None:
    """Fail fast when feature code and ``FeatureConfig`` are out of sync.

    This intentionally builds only the small, empty runtime state.  It runs
    before the multi-hundred-megabyte training/Trackman files are loaded, so a
    partially applied strategy change cannot waste a scheduled GPU job.
    """
    state = LeakageSafeFeatureEngineer(config).export_runtime_state()
    required = {
        "feature_version",
        "smoothing_prior",
        "pitcher_prior_strength",
        "batter_prior_strength",
        "cold_start_threshold",
        "trackman",
        "main_history",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"Feature runtime state is missing keys: {missing}")

    rejected = sorted(
        {"residual_prior_strength", "residual_probability_clip"} & set(state)
    )
    if rejected:
        raise RuntimeError(
            "Rejected TabM residual feature state is still present: "
            f"{rejected}"
        )
    print(
        f"[BACKEND] Feature config/state contract PASS "
        f"(feature_version={state['feature_version']})"
    )

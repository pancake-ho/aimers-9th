from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import FeatureConfig
from src.runtime import FEATURE_VERSION, HAND_MAP, build_features
from src.trackman_entity import ResolutionThresholds, resolve_pitcher_entities


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

ENTITY_STD_METRICS = TRACKMAN_METRICS

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


def _empty_profile(feature_names: Sequence[str]) -> Dict[str, object]:
    return {
        "lookup": {},
        "feature_names": list(feature_names),
        "defaults": {name: 0.0 for name in feature_names},
    }


def _update_entity_pitcher_profiles(
    profiles: Dict[str, Dict[str, object]],
    past: pd.DataFrame,
    target_season: int,
    trackman_ids: set[int],
) -> None:
    if not trackman_ids:
        return
    selected = past.loc[past["pitcher_trackman_id"].isin(trackman_ids)]
    if selected.empty:
        return

    pitcher = selected.groupby("pitcher_trackman_id", observed=True, sort=False)
    means = pitcher[list(TRACKMAN_METRICS)].mean()
    stds = pitcher[list(ENTITY_STD_METRICS)].std()
    shares = pitcher[list(PITCH_GROUPS)].mean()
    counts = pitcher.size()
    pitcher_lookup = profiles["pitcher"]["lookup"]
    for trackman_id in means.index:
        values: Dict[str, float] = {}
        for metric in TRACKMAN_METRICS:
            values[f"tm_entity_{metric}_mean"] = _finite_or_zero(
                means.loc[trackman_id, metric]
            )
        for metric in ENTITY_STD_METRICS:
            values[f"tm_entity_{metric}_std"] = _finite_or_zero(
                stds.loc[trackman_id, metric]
            )
        for group in PITCH_GROUPS:
            values[f"tm_entity_{group}_share"] = _finite_or_zero(
                shares.loc[trackman_id, group]
            )
        values["tm_entity_log_n"] = float(np.log1p(counts.loc[trackman_id]))
        values["tm_entity_available"] = 1.0
        pitcher_lookup[(target_season, int(trackman_id))] = values

    count_group = selected.groupby(
        ["pitcher_trackman_id", "balls_before", "strikes_before"],
        observed=True,
        sort=False,
    )
    count_means = count_group[list(TRACKMAN_METRICS)].mean()
    count_shares = count_group[list(PITCH_GROUPS)].mean()
    count_n = count_group.size()
    count_lookup = profiles["count"]["lookup"]
    for raw_key in count_means.index:
        trackman_id, balls, strikes = (int(value) for value in raw_key)
        values = {
            f"tm_entity_count_{metric}_mean": _finite_or_zero(
                count_means.loc[raw_key, metric]
            )
            for metric in TRACKMAN_METRICS
        }
        for group in PITCH_GROUPS:
            values[f"tm_entity_count_{group}_share"] = _finite_or_zero(
                count_shares.loc[raw_key, group]
            )
        values["tm_entity_count_log_n"] = float(np.log1p(count_n.loc[raw_key]))
        values["tm_entity_count_available"] = 1.0
        count_lookup[(target_season, trackman_id, balls, strikes)] = values

    latest_season = int(selected["season"].max())
    recent = selected.loc[selected["season"] == latest_season]
    older = selected.loc[selected["season"] < latest_season]
    if not recent.empty and not older.empty:
        recent_group = recent.groupby("pitcher_trackman_id", observed=True, sort=False)
        older_group = older.groupby("pitcher_trackman_id", observed=True, sort=False)
        recent_means = recent_group[list(TRACKMAN_METRICS)].mean()
        older_means = older_group[list(TRACKMAN_METRICS)].mean()
        recent_shares = recent_group[list(PITCH_GROUPS)].mean()
        older_shares = older_group[list(PITCH_GROUPS)].mean()
        recent_n = recent_group.size()
        common = recent_means.index.intersection(older_means.index)
        recent_lookup = profiles["recent"]["lookup"]
        for trackman_id in common:
            values = {
                f"tm_entity_recent_{metric}_delta": _finite_or_zero(
                    recent_means.loc[trackman_id, metric]
                    - older_means.loc[trackman_id, metric]
                )
                for metric in TRACKMAN_METRICS
            }
            for group in PITCH_GROUPS:
                values[f"tm_entity_recent_{group}_share_delta"] = _finite_or_zero(
                    recent_shares.loc[trackman_id, group]
                    - older_shares.loc[trackman_id, group]
                )
            values["tm_entity_recent_log_n"] = float(
                np.log1p(recent_n.loc[trackman_id])
            )
            values["tm_entity_recent_available"] = 1.0
            recent_lookup[(target_season, int(trackman_id))] = values

    group_profile = selected.groupby(
        ["pitcher_trackman_id", "pitch_type_group"],
        observed=True,
        sort=False,
    )[list(TRACKMAN_METRICS)].agg(["mean", "size"])
    arsenal_lookup = profiles["arsenal"]["lookup"]
    for trackman_id in sorted(trackman_ids):
        values: Dict[str, float] = {}
        available_pairs = 0
        for comparison, other in (("fb_break", "breaking"), ("fb_off", "offspeed")):
            fast_key = (trackman_id, "fastball")
            other_key = (trackman_id, other)
            if fast_key not in group_profile.index or other_key not in group_profile.index:
                for metric in TRACKMAN_METRICS:
                    values[f"tm_entity_{comparison}_{metric}_gap"] = 0.0
                continue
            fast_n = float(group_profile.loc[fast_key, (TRACKMAN_METRICS[0], "size")])
            other_n = float(group_profile.loc[other_key, (TRACKMAN_METRICS[0], "size")])
            pair_available = fast_n >= 20.0 and other_n >= 20.0
            available_pairs += int(pair_available)
            for metric in TRACKMAN_METRICS:
                gap = (
                    group_profile.loc[fast_key, (metric, "mean")]
                    - group_profile.loc[other_key, (metric, "mean")]
                    if pair_available
                    else 0.0
                )
                values[f"tm_entity_{comparison}_{metric}_gap"] = _finite_or_zero(gap)
        values["tm_entity_arsenal_available_pairs"] = float(available_pairs)
        arsenal_lookup[(target_season, trackman_id)] = values


class StrictPastTrackmanFeatures:
    """Strictly-past, non-target Trackman profiles.

    For a main-table row from season S, every Trackman value is aggregated
    only from seasons < S. The feature uses handedness and pre-pitch count;
    it never assumes that main pitcher_id equals pitcher_trackman_id and never
    reads another evaluation row.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.state_: Dict[str, object] | None = None

    def fit_from_csv(
        self,
        trackman_path: Path | str,
        main_train: pd.DataFrame | None = None,
    ) -> "StrictPastTrackmanFeatures":
        print("[FEATURE] Building strictly-past Trackman profiles...")
        usecols = [
            "season",
            "pitcher_trackman_id",
            "pitcher_team",
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
        tm["pitcher_trackman_id"] = pd.to_numeric(
            tm["pitcher_trackman_id"], errors="coerce"
        ).fillna(-1).astype(np.int64)
        tm["pitcher_hand"] = tm["pitcher_hand"].astype(np.int8)
        tm["batter_hand"] = tm["batter_hand"].astype(np.int8)
        tm["balls_before"] = pd.to_numeric(tm["balls_before"], errors="coerce").fillna(-1).astype(np.int8)
        tm["strikes_before"] = pd.to_numeric(tm["strikes_before"], errors="coerce").fillna(-1).astype(np.int8)

        for metric in TRACKMAN_METRICS:
            tm[metric] = pd.to_numeric(tm[metric], errors="coerce").astype(np.float32)

        pitch_group = tm["pitch_type_group"].fillna("other").astype(str).str.lower()
        tm["pitch_type_group"] = pitch_group
        for group in PITCH_GROUPS:
            tm[group] = (pitch_group == group).astype(np.float32)

        min_season = int(tm["season"].min())
        max_season = int(tm["season"].max())
        # Trackman ends before the main table (currently 2023 vs 2024).  The
        # old max_trackman+1 loop therefore produced no 2025 lookup and made
        # every Trackman feature zero at evaluation time.  Always materialize
        # states through the next main-table season.
        max_target_season = (
            max_season + 1
            if main_train is None
            else max(
                max_season + 1,
                int(pd.to_numeric(main_train["season"], errors="raise").max()) + 1,
            )
        )
        hand_lookup: Dict[
            tuple,
            Dict[str, float],
        ] = {}

        count_lookup: Dict[
            tuple,
            Dict[str, float],
        ] = {}

        hand_names: list[str] = []

        hand_count_names: list[str] = []

        for target_season in range(min_season, max_target_season + 1):
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

            per_count, hand_count_names = (
                _summarize_profile(
                    past,
                    group_cols=(
                        "pitcher_hand",
                        "batter_hand",
                        "balls_before",
                        "strikes_before",
                    ),
                    prefix="tm_count",
                )
            )
            for key, values in per_count.items():
                count_lookup[(target_season, *key)] = values

        entity_state = None
        if self.config.trackman_entity_enabled:
            if main_train is None:
                raise ValueError(
                    "main_train is required when Trackman entity resolution is enabled."
                )
            thresholds = ResolutionThresholds(
                min_pitches=int(self.config.trackman_entity_min_pitches),
                max_distance=float(self.config.trackman_entity_max_distance),
                min_margin_ratio=float(self.config.trackman_entity_min_margin_ratio),
                strong_margin_ratio=float(
                    self.config.trackman_entity_strong_margin_ratio
                ),
                min_count_ratio=float(self.config.trackman_entity_min_count_ratio),
                max_count_ratio=float(self.config.trackman_entity_max_count_ratio),
                team_penalty=float(self.config.trackman_entity_team_penalty),
                team_min_support=int(self.config.trackman_entity_team_min_support),
                team_min_dominance=float(
                    self.config.trackman_entity_team_min_dominance
                ),
            )
            mapping_names = [
                "tm_entity_map_confidence",
                "tm_entity_map_distance",
                "tm_entity_map_log_margin",
                "tm_entity_map_main_log_n",
                "tm_entity_map_trackman_log_n",
                "tm_entity_map_count_ratio",
                "tm_entity_map_team_agreement",
                "tm_entity_map_stable",
                "tm_entity_map_available",
            ]
            pitcher_names = [
                *(f"tm_entity_{metric}_mean" for metric in TRACKMAN_METRICS),
                *(f"tm_entity_{metric}_std" for metric in ENTITY_STD_METRICS),
                *(f"tm_entity_{group}_share" for group in PITCH_GROUPS),
                "tm_entity_log_n",
                "tm_entity_available",
            ]
            entity_count_names = [
                *(
                    f"tm_entity_count_{metric}_mean"
                    for metric in TRACKMAN_METRICS
                ),
                *(
                    f"tm_entity_count_{group}_share"
                    for group in PITCH_GROUPS
                ),
                "tm_entity_count_log_n",
                "tm_entity_count_available",
            ]
            recent_names = [
                *(f"tm_entity_recent_{metric}_delta" for metric in TRACKMAN_METRICS),
                *(
                    f"tm_entity_recent_{group}_share_delta"
                    for group in PITCH_GROUPS
                ),
                "tm_entity_recent_log_n",
                "tm_entity_recent_available",
            ]
            arsenal_names = [
                *(
                    f"tm_entity_{comparison}_{metric}_gap"
                    for comparison in ("fb_break", "fb_off")
                    for metric in TRACKMAN_METRICS
                ),
                "tm_entity_arsenal_available_pairs",
            ]
            entity_profiles = {
                "mapping": _empty_profile(
                    mapping_names
                ),
                "pitcher": _empty_profile(
                    pitcher_names
                ),
                "count": _empty_profile(
                    entity_count_names
                ),
                "recent": _empty_profile(
                    recent_names
                ),
                "arsenal": _empty_profile(
                    arsenal_names
                ),
            }
            audits: Dict[int, Dict[str, object]] = {}
            previous_top1: Dict[int, int] = {}
            main_season = pd.to_numeric(main_train["season"], errors="raise")

            for target_season in range(min_season, max_target_season + 1):
                main_past = main_train.loc[main_season < target_season]
                trackman_past = tm.loc[tm["season"] < target_season]
                if main_past.empty or trackman_past.empty:
                    continue
                result = resolve_pitcher_entities(
                    main_past=main_past,
                    trackman_past=trackman_past,
                    seasons=tuple(range(min_season, target_season)),
                    thresholds=thresholds,
                    previous_top1=previous_top1,
                )
                previous_top1 = result.top1
                audits[target_season] = result.audit
                mapping_lookup = entity_profiles["mapping"]["lookup"]
                for main_id, raw_values in result.mapping.items():
                    mapping_lookup[(target_season, int(main_id))] = {
                        "trackman_id": int(raw_values["trackman_id"]),
                        "tm_entity_map_confidence": float(raw_values["confidence"]),
                        "tm_entity_map_distance": float(raw_values["distance"]),
                        "tm_entity_map_log_margin": float(
                            np.log1p(raw_values["margin_ratio"])
                        ),
                        "tm_entity_map_main_log_n": float(raw_values["main_log_n"]),
                        "tm_entity_map_trackman_log_n": float(
                            raw_values["trackman_log_n"]
                        ),
                        "tm_entity_map_count_ratio": float(raw_values["count_ratio"]),
                        "tm_entity_map_team_agreement": float(
                            raw_values["team_agreement"]
                        ),
                        "tm_entity_map_stable": float(raw_values["stable"]),
                        "tm_entity_map_available": 1.0,
                    }
                accepted_trackman_ids = {
                    int(values["trackman_id"])
                    for values in result.mapping.values()
                }
                _update_entity_pitcher_profiles(
                    profiles=entity_profiles,
                    past=trackman_past,
                    target_season=target_season,
                    trackman_ids=accepted_trackman_ids,
                )
                print(
                    f"[FEATURE] Entity target={target_season} "
                    f"matched={result.audit['matched_pitchers']:,} "
                    f"coverage={result.audit['row_coverage']:.3f} "
                    f"teams={len(result.team_map)}"
                )

            final_target = max_target_season
            final_audit = audits.get(final_target, {})
            audit_checks = {
                "matched_pitchers": int(final_audit.get("matched_pitchers", 0))
                >= int(self.config.trackman_entity_min_matched_pitchers),
                "row_coverage": float(final_audit.get("row_coverage", 0.0))
                >= float(self.config.trackman_entity_min_row_coverage),
                "team_mappings": len(final_audit.get("team_map", {}))
                >= int(self.config.trackman_entity_min_team_mappings),
                "one_to_one": int(final_audit.get("collision_count", 1)) == 0,
            }
            if not all(audit_checks.values()):
                raise RuntimeError(
                    "Trackman entity-resolution quality gate failed: "
                    f"checks={audit_checks}, observed={final_audit}"
                )
            entity_state = {
                "state_version": 1,
                "profiles": entity_profiles,
                "audits": audits,
                "final_target_season": final_target,
                "thresholds": thresholds.__dict__.copy(),
            }

        self.state_ = {
            "state_version": 2,
            "source_seasons": [min_season, max_season],
            "max_target_season": max_target_season,
            "profiles": {
                "hand": {
                    "lookup": hand_lookup,
                    "count_feature_names": hand_count_names,
                    "defaults": {name: 0.0 for name in hand_names},
                },
                "hand_count": {
                    "lookup": count_lookup,
                    "feature_names": entity_count_names,
                    "defaults": {name: 0.0 for name in count_names},
                },
            },
            "entity": entity_state,
        }
        del tm
        print(
            f"[FEATURE] Trackman keys: hand={len(hand_lookup):,}, "
            f"hand_count={len(count_lookup):,}, "
            f"entity_mapping={0 if entity_state is None else len(entity_state['profiles']['mapping']['lookup']):,}"
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

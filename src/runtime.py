from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
FEATURE_VERSION = 5

PITCHER_RATE_COLS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)

BATTER_RATE_COLS = (
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
)

PITCHMIX_COLS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)

HAND_MAP = {
    "Left": 1,
    "Right": 2,
    "L": 1,
    "R": 2,
    1: 1,
    2: 2,
    "1": 1,
    "2": 2,
}

TRACKMAN_ENTITY_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
)

TRACKMAN_ENTITY_PITCH_GROUPS = (
    "fastball",
    "breaking",
    "offspeed",
    "other",
)


def _numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-5) -> pd.Series:
    return numerator / (denominator + eps)


def _experience_bin(values: pd.Series) -> pd.Series:
    n = _numeric(values, 0.0).to_numpy(dtype=np.float64, copy=False)
    labels = np.select(
        [n < 25, n < 100, n < 500, n < 2000],
        ["lt25", "25_99", "100_499", "500_1999"],
        default="ge2000",
    )
    return pd.Series(labels, index=values.index, dtype="object")


def _smooth_rate(
    rate: pd.Series,
    count: pd.Series,
    prior: float,
    strength: float,
) -> pd.Series:
    raw = _numeric(rate, prior).clip(0.0, 1.0).astype(np.float32)
    n = _numeric(count, 0.0).clip(lower=0.0).astype(np.float32)
    return ((n * raw + float(strength) * float(prior)) / (n + float(strength))).astype(np.float32)


def _lookup_trackman_profile(
    index: pd.Index,
    profile_state: Mapping[str, object],
    keys: Sequence[tuple],
) -> pd.DataFrame:
    lookup = profile_state.get("lookup", {})
    feature_names = list(profile_state.get("feature_names", ()))
    defaults = dict(profile_state.get("defaults", {}))
    n_rows = len(index)

    arrays: Dict[str, np.ndarray] = {}
    for name in feature_names:
        default = float(defaults.get(name, 0.0))
        arrays[name] = np.full(n_rows, default, dtype=np.float32)

    # This is a pure row lookup. No statistic is computed from test rows.
    for row_pos, key in enumerate(keys):
        values = lookup.get(tuple(key))
        if values is None:
            continue
        for name, value in values.items():
            if name in arrays and np.isfinite(value):
                arrays[name][row_pos] = np.float32(value)

    return pd.DataFrame(arrays, index=index)


def add_trackman_features(
    out: pd.DataFrame,
    trackman_state,
) -> pd.DataFrame:
    if not trackman_state:
        return out

    season = (
        _numeric(
            out["season"],
            -1,
        )
        .astype(int)
        .to_numpy()
    )

    pitcher_hand = (
        out["pitcher_hand"]
        .map(HAND_MAP)
        .fillna(
            _numeric(
                out["pitcher_hand"],
                -1,
            )
        )
        .astype(int)
        .to_numpy()
    )

    batter_hand = (
        out["batter_hand"]
        .map(HAND_MAP)
        .fillna(
            _numeric(
                out["batter_hand"],
                -1,
            )
        )
        .astype(int)
        .to_numpy()
    )

    profiles = trackman_state.get(
        "profiles",
        {},
    )

    profile_frames = []

    global_profile = profiles.get(
        "hand"
    )

    if global_profile:
        keys = list(
            zip(
                season,
                pitcher_hand,
                batter_hand,
            )
        )

        profile_frames.append(
            _lookup_trackman_profile(
                out.index,
                global_profile,
                keys,
            )
        )

    count_profile = profiles.get(
        "hand_count"
    )

    if count_profile:
        balls = (
            _numeric(
                out["balls_before"],
                -1,
            )
            .astype(int)
            .to_numpy()
        )

        strikes = (
            _numeric(
                out["strikes_before"],
                -1,
            )
            .astype(int)
            .to_numpy()
        )

        keys = list(
            zip(
                season,
                pitcher_hand,
                batter_hand,
                balls,
                strikes,
            )
        )

        profile_frames.append(
            _lookup_trackman_profile(
                out.index,
                count_profile,
                keys,
            )
        )

    # --------------------------------------------------------
    # Precomputed official-data pitcher entity state.
    # Runtime never resolves identities using evaluation rows.
    # --------------------------------------------------------
    entity_state = trackman_state.get(
        "entity"
    )

    if entity_state:
        entity_profiles = (
            entity_state.get(
                "profiles",
                {},
            )
        )

        mapping_profile = (
            entity_profiles.get(
                "mapping"
            )
        )

        if mapping_profile:
            pitcher_id = (
                _numeric(
                    out["pitcher_id"],
                    -1,
                )
                .astype(np.int64)
                .to_numpy()
            )

            mapping_keys = list(
                zip(
                    season,
                    pitcher_id,
                )
            )

            profile_frames.append(
                _lookup_trackman_profile(
                    out.index,
                    mapping_profile,
                    mapping_keys,
                )
            )

            mapping_lookup = (
                mapping_profile.get(
                    "lookup",
                    {},
                )
            )

            trackman_id = np.full(
                len(out),
                -1,
                dtype=np.int64,
            )

            for row_pos, key in enumerate(
                mapping_keys
            ):
                values = (
                    mapping_lookup.get(
                        tuple(key)
                    )
                )

                if values is not None:
                    trackman_id[
                        row_pos
                    ] = int(
                        values.get(
                            "trackman_id",
                            -1,
                        )
                    )

            entity_key_specs = {
                "pitcher": list(
                    zip(
                        season,
                        trackman_id,
                    )
                ),
                "recent": list(
                    zip(
                        season,
                        trackman_id,
                    )
                ),
                "arsenal": list(
                    zip(
                        season,
                        trackman_id,
                    )
                ),
            }

            balls = (
                _numeric(
                    out["balls_before"],
                    -1,
                )
                .astype(int)
                .to_numpy()
            )

            strikes = (
                _numeric(
                    out["strikes_before"],
                    -1,
                )
                .astype(int)
                .to_numpy()
            )

            entity_key_specs[
                "count"
            ] = list(
                zip(
                    season,
                    trackman_id,
                    balls,
                    strikes,
                )
            )

            for (
                name,
                keys,
            ) in entity_key_specs.items():
                profile = (
                    entity_profiles.get(
                        name
                    )
                )

                if profile:
                    profile_frames.append(
                        _lookup_trackman_profile(
                            out.index,
                            profile,
                            keys,
                        )
                    )

    if not profile_frames:
        return out

    result = pd.concat(
        [
            out,
            *profile_frames,
        ],
        axis=1,
        copy=False,
    )

    # Fail closed on the population/entity feature-name collision that existed
    # in the dormant implementation.
    if not result.columns.is_unique:
        duplicate_columns = (
            result.columns[
                result.columns.duplicated()
            ]
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "Duplicate Trackman feature columns: "
            f"{duplicate_columns[:20]}"
        )

    if entity_state:
        result = (
            _add_trackman_entity_v2_features(
                result,
                entity_state,
            )
        )

    return result


def _add_trackman_entity_v2_features(
    out: pd.DataFrame,
    entity_state: Mapping[str, object],
) -> pd.DataFrame:
    v2 = entity_state.get(
        "v2",
        {},
    )

    if not bool(
        v2.get(
            "enabled",
            False,
        )
    ):
        return out

    required = {
        "tm_entity_map_confidence",
        "tm_entity_available",
        "tm_entity_log_n",
        "tm_entity_freshness_gap",
        "tm_hand_available",
        "tm_entity_count_available",
        "tm_entity_count_log_n",
        "tm_count_available",
    }

    for metric in TRACKMAN_ENTITY_METRICS:
        required.update(
            {
                f"tm_entity_{metric}_mean",
                f"tm_hand_{metric}_mean",
                (
                    "tm_entity_count_"
                    f"{metric}_mean"
                ),
                f"tm_count_{metric}_mean",
            }
        )

    for group in TRACKMAN_ENTITY_PITCH_GROUPS:
        required.update(
            {
                f"tm_entity_{group}_share",
                f"tm_hand_{group}_share",
                (
                    "tm_entity_count_"
                    f"{group}_share"
                ),
                f"tm_count_{group}_share",
            }
        )

    missing = sorted(
        required
        - set(out.columns)
    )

    if missing:
        raise ValueError(
            "Trackman entity v2 requires "
            "missing columns: "
            f"{missing[:20]}"
        )

    pool_strength = float(
        v2.get(
            "pool_strength",
            500.0,
        )
    )
    count_pool_strength = float(
        v2.get(
            "count_pool_strength",
            100.0,
        )
    )
    freshness_decay = float(
        v2.get(
            "freshness_decay",
            0.70,
        )
    )

    if (
        pool_strength <= 0.0
        or count_pool_strength <= 0.0
    ):
        raise ValueError(
            "Trackman entity pooling "
            "strengths must be positive."
        )

    if (
        freshness_decay < 0.0
        or not np.isfinite(
            freshness_decay
        )
    ):
        raise ValueError(
            "Trackman entity freshness decay "
            "must be finite and non-negative."
        )

    confidence = (
        _numeric(
            out[
                "tm_entity_map_confidence"
            ],
            0.0,
        )
        .clip(
            0.0,
            1.0,
        )
    )

    freshness_gap = (
        _numeric(
            out[
                "tm_entity_freshness_gap"
            ],
            0.0,
        )
        .clip(lower=0.0)
    )

    freshness_factor = (
        np.exp(
            -freshness_decay
            * freshness_gap
        )
        .astype(np.float32)
    )

    entity_available = (
        _numeric(
            out[
                "tm_entity_available"
            ],
            0.0,
        )
        > 0.0
    ).astype(np.float32)

    population_available = (
        _numeric(
            out[
                "tm_hand_available"
            ],
            0.0,
        )
        > 0.0
    ).astype(np.float32)

    pair_available = (
        entity_available
        * population_available
    ).astype(np.float32)

    entity_log_n = (
        _numeric(
            out[
                "tm_entity_log_n"
            ],
            0.0,
        )
        .clip(
            lower=0.0,
            upper=20.0,
        )
    )

    entity_n = (
        np.expm1(
            entity_log_n
        )
        .clip(lower=0.0)
    )

    sample_factor = (
        entity_n
        / (
            entity_n
            + pool_strength
        )
    )

    alpha = (
        confidence
        * sample_factor
        * freshness_factor
        * pair_available
    ).clip(
        0.0,
        1.0,
    ).astype(np.float32)

    # Count-specific reliability.
    count_entity_available = (
        _numeric(
            out[
                "tm_entity_count_available"
            ],
            0.0,
        )
        > 0.0
    ).astype(np.float32)

    count_population_available = (
        _numeric(
            out[
                "tm_count_available"
            ],
            0.0,
        )
        > 0.0
    ).astype(np.float32)

    count_pair_available = (
        count_entity_available
        * count_population_available
    ).astype(np.float32)

    count_log_n = (
        _numeric(
            out[
                "tm_entity_count_log_n"
            ],
            0.0,
        )
        .clip(
            lower=0.0,
            upper=20.0,
        )
    )

    count_n = (
        np.expm1(
            count_log_n
        )
        .clip(lower=0.0)
    )

    count_sample_factor = (
        count_n
        / (
            count_n
            + count_pool_strength
        )
    )

    count_alpha = (
        confidence
        * count_sample_factor
        * freshness_factor
        * count_pair_available
    ).clip(
        0.0,
        1.0,
    ).astype(np.float32)

    out[
        "tm_entity_freshness_factor"
    ] = freshness_factor

    out[
        "tm_entity_pool_alpha"
    ] = alpha

    out[
        "tm_entity_count_pool_alpha"
    ] = count_alpha

    out[
        "tm_entity_v2_available"
    ] = (
        pair_available
        .astype(np.int8)
    )

    out[
        "tm_entity_count_v2_available"
    ] = (
        count_pair_available
        .astype(np.int8)
    )

    # --------------------------------------------------------
    # Physical Trackman metrics.
    # --------------------------------------------------------
    for metric in TRACKMAN_ENTITY_METRICS:
        entity_value = _numeric(
            out[
                f"tm_entity_{metric}_mean"
            ],
            0.0,
        )

        population_value = _numeric(
            out[
                f"tm_hand_{metric}_mean"
            ],
            0.0,
        )

        deviation = (
            (
                entity_value
                - population_value
            )
            * pair_available
        ).astype(np.float32)

        out[
            f"tm_entity_{metric}_pop_dev"
        ] = deviation

        pooled = (
            population_value
            + alpha
            * deviation
        )

        out[
            f"tm_entity_{metric}_pooled"
        ] = (
            pooled.where(
                population_available
                > 0.0,
                0.0,
            )
            .astype(np.float32)
        )

        count_entity_value = _numeric(
            out[
                (
                    "tm_entity_count_"
                    f"{metric}_mean"
                )
            ],
            0.0,
        )

        count_population_value = _numeric(
            out[
                f"tm_count_{metric}_mean"
            ],
            0.0,
        )

        count_deviation = (
            (
                count_entity_value
                - count_population_value
            )
            * count_pair_available
        ).astype(np.float32)

        out[
            (
                "tm_entity_count_"
                f"{metric}_pop_dev"
            )
        ] = count_deviation

        count_pooled = (
            count_population_value
            + count_alpha
            * count_deviation
        )

        out[
            (
                "tm_entity_count_"
                f"{metric}_pooled"
            )
        ] = (
            count_pooled.where(
                count_population_available
                > 0.0,
                0.0,
            )
            .astype(np.float32)
        )

    # --------------------------------------------------------
    # Pitch-group usage shares.
    # --------------------------------------------------------
    for group in TRACKMAN_ENTITY_PITCH_GROUPS:
        entity_value = _numeric(
            out[
                f"tm_entity_{group}_share"
            ],
            0.0,
        )

        population_value = _numeric(
            out[
                f"tm_hand_{group}_share"
            ],
            0.0,
        )

        deviation = (
            (
                entity_value
                - population_value
            )
            * pair_available
        ).astype(np.float32)

        out[
            (
                f"tm_entity_{group}_"
                "share_pop_dev"
            )
        ] = deviation

        pooled = (
            population_value
            + alpha
            * deviation
        )

        out[
            (
                f"tm_entity_{group}_"
                "share_pooled"
            )
        ] = (
            pooled.where(
                population_available
                > 0.0,
                0.0,
            )
            .astype(np.float32)
        )

        count_entity_value = _numeric(
            out[
                (
                    "tm_entity_count_"
                    f"{group}_share"
                )
            ],
            0.0,
        )

        count_population_value = _numeric(
            out[
                f"tm_count_{group}_share"
            ],
            0.0,
        )

        count_deviation = (
            (
                count_entity_value
                - count_population_value
            )
            * count_pair_available
        ).astype(np.float32)

        out[
            (
                "tm_entity_count_"
                f"{group}_share_pop_dev"
            )
        ] = count_deviation

        count_pooled = (
            count_population_value
            + count_alpha
            * count_deviation
        )

        out[
            (
                "tm_entity_count_"
                f"{group}_share_pooled"
            )
        ] = (
            count_pooled.where(
                count_population_available
                > 0.0,
                0.0,
            )
            .astype(np.float32)
        )

    return out


def add_main_history_features(out: pd.DataFrame, history_state) -> pd.DataFrame:
    """Attach precomputed season-past profiles using only the current row key."""
    if not history_state:
        return out

    season = _numeric(out["season"], -1).astype(int).to_numpy()
    pitcher_id = _numeric(out["pitcher_id"], -1).astype(np.int64).to_numpy()
    balls = _numeric(out["balls_before"], -1).astype(int).to_numpy()
    strikes = _numeric(out["strikes_before"], -1).astype(int).to_numpy()
    same_hand = (
        out["pitcher_hand"].map(HAND_MAP).fillna(_numeric(out["pitcher_hand"], -1))
        == out["batter_hand"].map(HAND_MAP).fillna(_numeric(out["batter_hand"], -1))
    ).astype(int).to_numpy()

    profiles = history_state.get("profiles", {})
    profile_keys = {
        "pitcher": list(zip(season, pitcher_id)),
        "pitcher_count": list(zip(season, pitcher_id, balls, strikes)),
        "pitcher_matchup": list(zip(season, pitcher_id, same_hand)),
        "pitcher_count_matchup": list(
            zip(season, pitcher_id, balls, strikes, same_hand)
        ),
    }
    frames = []
    for name, keys in profile_keys.items():
        profile = profiles.get(name)
        if profile:
            frames.append(_lookup_trackman_profile(out.index, profile, keys))
    if not frames:
        return out
    return pd.concat([out, *frames], axis=1, copy=False)


def build_features(df: pd.DataFrame, feature_state: Mapping[str, object]) -> pd.DataFrame:
    if int(feature_state.get("feature_version", -1)) != FEATURE_VERSION:
        raise ValueError(
            f"Unsupported feature version: {feature_state.get('feature_version')}"
        )

    out = df.copy()
    prior = float(feature_state["smoothing_prior"])
    pitcher_strength = float(feature_state["pitcher_prior_strength"])
    batter_strength = float(feature_state["batter_prior_strength"])
    cold_start = int(feature_state.get("cold_start_threshold", 50))

    balls = _numeric(out["balls_before"], -1)
    strikes = _numeric(out["strikes_before"], -1)
    inning = _numeric(out["inning"], 0.0)
    leverage = _numeric(out["li"], 0.0).clip(lower=0.0)
    score_diff = _numeric(out["score_diff_pitcher_team"], 0.0)

    # Temporal regime. Only the official season column is used; no external
    # measurement or test-distribution statistic enters this feature.
    season = _numeric(out["season"], 2019.0)
    out["season_index"] = (season - 2019.0).astype(np.float32)
    out["is_abs_era"] = (season >= 2024).astype(np.int8)
    out["years_since_abs"] = (season - 2023.0).clip(lower=0.0).astype(np.float32)
    month_angle = 2.0 * np.pi * (_numeric(out["game_month"], 1.0) - 1.0) / 12.0
    out["game_month_sin"] = np.sin(month_angle).astype(np.float32)
    out["game_month_cos"] = np.cos(month_angle).astype(np.float32)

    # Count state and plate-appearance pressure.
    out["count_state"] = balls.astype(int).astype(str) + "-" + strikes.astype(int).astype(str)
    out["is_hitter_count"] = ((balls >= 2) & (strikes <= 1)).astype(np.int8)
    out["is_pitcher_count"] = ((strikes == 2) & (balls <= 1)).astype(np.int8)
    out["is_full_count"] = ((balls == 3) & (strikes == 2)).astype(np.int8)
    out["is_two_strike"] = (strikes == 2).astype(np.int8)
    out["is_three_ball"] = (balls == 3).astype(np.int8)
    out["count_leverage"] = (balls - strikes).astype(np.float32)
    out["count_depth"] = (balls + strikes).astype(np.float32)
    out["count_pressure"] = ((balls + strikes) / 5.0).astype(np.float32)

    # Runner, score, leverage and handedness state.
    out["is_scoring_position"] = (
        (_numeric(out["runner_on_2b"]) == 1) | (_numeric(out["runner_on_3b"]) == 1)
    ).astype(np.int8)
    out["is_bases_loaded"] = (_numeric(out["num_runners_on"]) == 3).astype(np.int8)
    out["abs_score_diff"] = score_diff.abs().astype(np.float32)
    out["is_close_game"] = (score_diff.abs() <= 2).astype(np.int8)
    out["is_late_game"] = (inning >= 7).astype(np.int8)
    out["is_extra_inning"] = (inning >= 10).astype(np.int8)
    out["li_log"] = np.log1p(leverage).astype(np.float32)
    out["is_high_leverage"] = (leverage >= 2.0).astype(np.int8)
    out["is_clutch"] = (
        (inning >= 7) & (score_diff.abs() <= 2) & (leverage >= 1.5)
    ).astype(np.int8)
    out["late_leverage"] = ((inning >= 7).astype(np.float32) * np.log1p(leverage)).astype(np.float32)
    out["runner_leverage"] = (
        _numeric(out["num_runners_on"], 0.0) * np.log1p(leverage)
    ).astype(np.float32)
    out["win_expectancy_diff"] = (
        _numeric(out["home_win_expectancy"], 50.0)
        - _numeric(out["away_win_expectancy"], 50.0)
    ).astype(np.float32)
    out["win_expectancy_certainty"] = (
        (_numeric(out["home_win_expectancy"], 50.0) - 50.0).abs() / 50.0
    ).astype(np.float32)
    out["stadium_owner_team"] = np.where(
        out["top_bottom"].astype(str).eq("T"),
        out["pitcher_team_id"],
        out["batter_team_id"],
    )
    out["same_hand"] = (
        out["pitcher_hand"].astype(str) == out["batter_hand"].astype(str)
    ).astype(np.int8)
    out["hand_matchup"] = (
        out["pitcher_hand"].astype(str) + "_" + out["batter_hand"].astype(str)
    )

    # IDs are not target encoded here. Combinations are fixed, row-wise keys.
    pitcher_key = out["pitcher_id"].fillna("__MISSING__").astype(str)
    batter_key = out["batter_id"].fillna("__MISSING__").astype(str)
    out["pitcher_count_combo"] = pitcher_key + "_" + out["count_state"]
    out["batter_count_combo"] = batter_key + "_" + out["count_state"]
    out["pitcher_base_combo"] = pitcher_key + "_" + out["base_state"].astype(str)

    pitcher_n = _numeric(out["asof_pitcher_n"], 0.0).clip(lower=0.0)
    batter_n = _numeric(out["asof_batter_n"], 0.0).clip(lower=0.0)
    pitchmix_n = _numeric(out["asof_pitcher_pitchmix_n"], 0.0).clip(lower=0.0)
    out["pitcher_history_log_n"] = np.log1p(pitcher_n).astype(np.float32)
    out["batter_history_log_n"] = np.log1p(batter_n).astype(np.float32)
    out["pitchmix_history_log_n"] = np.log1p(pitchmix_n).astype(np.float32)
    out["pitcher_cold_start"] = (pitcher_n < cold_start).astype(np.int8)
    out["batter_cold_start"] = (batter_n < cold_start).astype(np.int8)
    out["pitcher_experience_bin"] = _experience_bin(pitcher_n)
    out["batter_experience_bin"] = _experience_bin(batter_n)

    # Fixed-prior empirical-Bayes-style shrinkage. The prior does not use any
    # label or distribution from validation/test rows.
    for col in PITCHER_RATE_COLS:
        out[f"{col}_smoothed"] = _smooth_rate(
            out[col], pitcher_n, prior, pitcher_strength
        )
    for col in BATTER_RATE_COLS:
        out[f"{col}_smoothed"] = _smooth_rate(
            out[col], batter_n, prior, batter_strength
        )

    # Let trees choose between lighter and heavier shrinkage for the dominant
    # historical success signals.
    out["asof_pitcher_success_rate_s25"] = _smooth_rate(
        out["asof_pitcher_success_rate"], pitcher_n, prior, 25.0
    )
    out["asof_pitcher_success_rate_s500"] = _smooth_rate(
        out["asof_pitcher_success_rate"], pitcher_n, prior, 500.0
    )
    out["asof_batter_success_rate_s200"] = _smooth_rate(
        out["asof_batter_success_rate"], batter_n, prior, 200.0
    )

    baseline_success = out["asof_pitcher_success_rate_smoothed"]
    baseline_middle = out["asof_pitcher_middle_rate_smoothed"]
    recent_success = []
    recent_middle = []
    for window in (1, 3, 5):
        success = _numeric(out[f"asof_pitcher_prev{window}_game_success_rate"], prior)
        middle = _numeric(out[f"asof_pitcher_prev{window}_game_middle_rate"], prior)
        out[f"diff_prev{window}_success"] = (success - baseline_success).astype(np.float32)
        out[f"diff_prev{window}_middle"] = (middle - baseline_middle).astype(np.float32)
        recent_success.append(success)
        recent_middle.append(middle)

    out["recent_success_blend"] = (
        0.50 * recent_success[0] + 0.30 * recent_success[1] + 0.20 * recent_success[2]
    ).astype(np.float32)
    out["recent_middle_blend"] = (
        0.50 * recent_middle[0] + 0.30 * recent_middle[1] + 0.20 * recent_middle[2]
    ).astype(np.float32)
    out["pitcher_form_gap"] = (out["recent_success_blend"] - baseline_success).astype(np.float32)
    out["pitcher_volatility"] = (
        out["diff_prev1_success"].abs() + out["diff_prev3_success"].abs()
    ).astype(np.float32)
    out["pressure_interaction"] = (out["pitcher_volatility"] * out["li_log"]).astype(np.float32)
    out["recent_form_leverage"] = (out["pitcher_form_gap"] * out["li_log"]).astype(np.float32)

    eps = 1e-4
    out["strike_to_ball_ratio"] = _safe_divide(
        out["asof_pitcher_strike_rate_smoothed"],
        out["asof_pitcher_ball_rate_smoothed"],
        eps,
    ).clip(upper=20.0).astype(np.float32)
    out["reverse_to_success_ratio"] = _safe_divide(
        out["asof_pitcher_reverse_rate_smoothed"], baseline_success, eps
    ).clip(upper=20.0).astype(np.float32)
    out["middle_to_success_ratio"] = _safe_divide(
        out["asof_pitcher_middle_rate_smoothed"], baseline_success, eps
    ).clip(upper=20.0).astype(np.float32)
    out["two_strike_ball_tendency"] = (
        out["is_two_strike"] * out["asof_pitcher_ball_rate_smoothed"]
    ).astype(np.float32)
    out["three_ball_strike_tendency"] = (
        out["is_three_ball"] * out["asof_pitcher_strike_rate_smoothed"]
    ).astype(np.float32)
    out["count_success_interaction"] = (
        out["count_leverage"] * baseline_success
    ).astype(np.float32)
    out["abs_history_interaction"] = (
        out["is_abs_era"] * baseline_success
    ).astype(np.float32)

    pitchmix = out[list(PITCHMIX_COLS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pitchmix = pitchmix.clip(lower=0.0, upper=1.0)
    entropy_input = pitchmix.clip(lower=1e-5)
    out["pitchmix_entropy"] = (-(entropy_input * np.log(entropy_input)).sum(axis=1)).astype(np.float32)
    out["pitchmix_max_share"] = pitchmix.max(axis=1).astype(np.float32)
    out["pitchmix_missing"] = (pitchmix_n <= 0).astype(np.int8)

    out = add_trackman_features(out, feature_state.get("trackman"))
    out = add_main_history_features(out, feature_state.get("main_history"))

    # A cross-season empirical-Bayes prior is most valuable early in a season,
    # when the official within-season asof count is small.  Context deltas are
    # added to the already-smoothed current-season baseline, so a league-wide
    # target shift does not get copied from older seasons.
    if "hist_pitcher_effect" in out.columns:
        history_available = _numeric(out["hist_pitcher_available"], 0.0)
        history_prior = (
            prior + _numeric(out["hist_pitcher_effect"], 0.0)
        ).clip(0.02, 0.98)
        history_prior = history_prior.where(history_available > 0.0, prior)
        current_rate = _numeric(out["asof_pitcher_success_rate"], prior).clip(0.0, 1.0)
        for strength in (100.0, 300.0):
            out[f"cross_season_success_s{int(strength)}"] = (
                (pitcher_n * current_rate + strength * history_prior)
                / (pitcher_n + strength)
            ).astype(np.float32)

        for context in (
            "pitcher_count",
            "pitcher_matchup",
            "pitcher_count_matchup",
        ):
            delta_col = f"hist_{context}_delta"
            if delta_col in out.columns:
                out[f"{context}_adjusted_success"] = (
                    baseline_success + _numeric(out[delta_col], 0.0)
                ).clip(0.02, 0.98).astype(np.float32)

        zero = pd.Series(0.0, index=out.index, dtype=np.float32)
        out["history_context_spread"] = (
            _numeric(out["hist_pitcher_count_delta"], 0.0).abs()
            if "hist_pitcher_count_delta" in out.columns
            else zero
        )
        out["history_context_spread"] = (
            out["history_context_spread"]
            + (
                _numeric(out["hist_pitcher_matchup_delta"], 0.0).abs()
                if "hist_pitcher_matchup_delta" in out.columns
                else zero
            )
            + (
                _numeric(out["hist_pitcher_count_matchup_delta"], 0.0).abs()
                if "hist_pitcher_count_matchup_delta" in out.columns
                else zero
            )
        ).astype(np.float32)

    return out


def preprocess_frame(df: pd.DataFrame, state: Mapping[str, object]) -> pd.DataFrame:
    result: Dict[str, pd.Series] = {}
    for col in state["cat_cols"]:
        values = df[col].fillna("__MISSING__").astype(str)
        mapping = state["category_maps"][col]
        result[col] = values.map(mapping).fillna(-1).astype(np.int32)

    for col in state["num_cols"]:
        values = pd.to_numeric(df[col], errors="coerce")
        result[col] = values.fillna(state["numeric_medians"][col]).astype(np.float32)

    matrix = pd.DataFrame(result, index=df.index)
    return matrix[list(state["feature_names"])]


def apply_numeric_pca_state(
    X: pd.DataFrame,
    state: Mapping[str, object],
    *,
    chunk_size: int = 65536,
) -> pd.DataFrame:
    """Apply an immutable train-fitted PCA representation.

    This function is intentionally row-independent at inference:
    - PCA mean/std/components come only from training.
    - No statistic is fitted from evaluation rows.
    - Evaluation rows do not interact with one another.
    """

    columns = list(
        state["input_columns"]
    )

    if not columns:
        raise ValueError(
            "PCA input column list is empty."
        )

    missing = [
        column
        for column in columns
        if column not in X.columns
    ]

    if missing:
        raise ValueError(
            "PCA input columns are missing: "
            f"{missing[:10]}"
        )

    mean = np.asarray(
        state["mean"],
        dtype=np.float64,
    )

    std = np.asarray(
        state["std"],
        dtype=np.float64,
    )

    components = np.asarray(
        state["components"],
        dtype=np.float64,
    )

    if mean.shape != (
        len(columns),
    ):
        raise ValueError(
            "PCA mean/input shape mismatch."
        )

    if std.shape != (
        len(columns),
    ):
        raise ValueError(
            "PCA std/input shape mismatch."
        )

    if components.ndim != 2:
        raise ValueError(
            "PCA components must be "
            "two-dimensional."
        )

    if (
        components.shape[1]
        != len(columns)
    ):
        raise ValueError(
            "PCA component/input mismatch."
        )

    if (
        not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or not np.isfinite(
            components
        ).all()
    ):
        raise ValueError(
            "PCA state contains "
            "NaN or infinity."
        )

    if (
        std <= 0.0
    ).any():
        raise ValueError(
            "PCA standard deviation "
            "must be positive."
        )

    clip = float(
        state.get(
            "clip",
            8.0,
        )
    )

    if (
        not np.isfinite(clip)
        or clip <= 0.0
    ):
        raise ValueError(
            "Invalid PCA clipping bound."
        )

    n_rows = int(
        len(X)
    )

    n_components = int(
        components.shape[0]
    )

    output = np.empty(
        (
            n_rows,
            n_components,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        n_rows,
        int(chunk_size),
    ):
        stop = min(
            n_rows,
            start + int(chunk_size),
        )

        chunk = (
            X.iloc[
                start:stop
            ][columns]
            .to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

        if not np.isfinite(
            chunk
        ).all():
            raise ValueError(
                "PCA input matrix contains "
                "NaN or infinity."
            )

        chunk -= mean
        chunk /= std

        np.clip(
            chunk,
            -clip,
            clip,
            out=chunk,
        )

        projection = (
            chunk
            @ components.T
        )

        output[
            start:stop
        ] = projection.astype(
            np.float32,
            copy=False,
        )

    if not np.isfinite(
        output
    ).all():
        raise ValueError(
            "PCA representation contains "
            "NaN or infinity."
        )

    prefix = str(
        state.get(
            "output_prefix",
            "mv_pca",
        )
    )

    names = [
        f"{prefix}_{index:02d}"
        for index in range(
            n_components
        )
    ]

    return pd.DataFrame(
        output,
        index=X.index,
        columns=names,
    )


def apply_probability_bias(prediction, bias: float) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    return np.clip(pred - float(bias), 1e-6, 1.0 - 1e-6)


def apply_logit_intercept(prediction, intercept: float) -> np.ndarray:
    """Apply a train-fitted intercept without distorting probability bounds."""
    pred = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(pred) - np.log1p(-pred)
    shifted = logit + float(intercept)
    # Stable sigmoid for the small calibration range used by training.
    output = np.empty_like(shifted)
    positive = shifted >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-shifted[positive]))
    exp_value = np.exp(shifted[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return np.clip(output, 1e-6, 1.0 - 1e-6)


def blend_predictions(predictions: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(predictions) != len(weights):
        raise ValueError("Prediction/weight length mismatch.")
    weight = np.asarray(weights, dtype=np.float64)
    if (weight < 0).any() or not np.isfinite(weight).all() or weight.sum() <= 0:
        raise ValueError("Invalid ensemble weights.")
    weight = weight / weight.sum()
    stacked = np.column_stack([np.asarray(p, dtype=np.float64) for p in predictions])
    return np.clip(stacked @ weight, 1e-6, 1.0 - 1e-6)

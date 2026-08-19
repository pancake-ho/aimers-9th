from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
FEATURE_VERSION = 3

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


def add_trackman_features(out: pd.DataFrame, trackman_state) -> pd.DataFrame:
    if not trackman_state:
        return out

    season = _numeric(out["season"], -1).astype(int).to_numpy()
    pitcher_hand = (
        out["pitcher_hand"]
        .map(HAND_MAP)
        .fillna(_numeric(out["pitcher_hand"], -1))
        .astype(int)
        .to_numpy()
    )
    batter_hand = (
        out["batter_hand"]
        .map(HAND_MAP)
        .fillna(_numeric(out["batter_hand"], -1))
        .astype(int)
        .to_numpy()
    )

    profiles = trackman_state.get("profiles", {})
    profile_frames = []
    global_profile = profiles.get("hand")
    if global_profile:
        keys = list(zip(season, pitcher_hand, batter_hand))
        profile_frames.append(_lookup_trackman_profile(out.index, global_profile, keys))

    count_profile = profiles.get("hand_count")
    if count_profile:
        balls = _numeric(out["balls_before"], -1).astype(int).to_numpy()
        strikes = _numeric(out["strikes_before"], -1).astype(int).to_numpy()
        keys = list(zip(season, pitcher_hand, batter_hand, balls, strikes))
        profile_frames.append(_lookup_trackman_profile(out.index, count_profile, keys))

    if not profile_frames:
        return out
    return pd.concat([out, *profile_frames], axis=1, copy=False)


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

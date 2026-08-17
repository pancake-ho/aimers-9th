from __future__ import annotations

import gc
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

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


def load_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def add_trackman_features(out: pd.DataFrame, trackman_state):
    if trackman_state is None:
        return out

    lookup = trackman_state["lookup"]
    feature_names = trackman_state["feature_names"]
    n = len(out)
    values = {
        col: np.full(n, np.nan, dtype=np.float32)
        for col in feature_names
    }
    if "tm_past_n" in values:
        values["tm_past_n"][:] = 0.0
    if "tm_past_available" in values:
        values["tm_past_available"][:] = 0.0

    seasons = pd.to_numeric(out["season"], errors="coerce").fillna(-1).astype(int).to_numpy()
    ph = (
        out["pitcher_hand"]
        .map(HAND_MAP)
        .fillna(pd.to_numeric(out["pitcher_hand"], errors="coerce"))
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )
    bh = (
        out["batter_hand"]
        .map(HAND_MAP)
        .fillna(pd.to_numeric(out["batter_hand"], errors="coerce"))
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )

    for i, (season, p_hand, b_hand) in enumerate(zip(seasons, ph, bh)):
        stats = lookup.get((int(season), int(p_hand), int(b_hand)))
        if stats is None:
            continue
        for col, value in stats.items():
            values[col][i] = value

    for col, array in values.items():
        out[col] = array
    return out


def build_features(df: pd.DataFrame, feature_state) -> pd.DataFrame:
    out = df.copy()
    prior = float(feature_state["smoothing_prior"])
    pitcher_strength = float(feature_state["pitcher_prior_strength"])
    batter_strength = float(feature_state["batter_prior_strength"])

    out["count_state"] = (
        out["balls_before"].fillna(-1).astype(str)
        + "-"
        + out["strikes_before"].fillna(-1).astype(str)
    )
    out["is_hitter_count"] = (
        (out["balls_before"] >= 2) & (out["strikes_before"] <= 1)
    ).astype(np.int8)
    out["is_pitcher_count"] = (
        (out["strikes_before"] == 2) & (out["balls_before"] <= 1)
    ).astype(np.int8)
    out["is_full_count"] = (
        (out["balls_before"] == 3) & (out["strikes_before"] == 2)
    ).astype(np.int8)
    out["is_two_strike"] = (out["strikes_before"] == 2).astype(np.int8)
    out["is_three_ball"] = (out["balls_before"] == 3).astype(np.int8)
    out["count_leverage"] = (
        out["balls_before"].fillna(0) - out["strikes_before"].fillna(0)
    ).astype(np.int8)

    out["is_scoring_position"] = (
        (out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)
    ).astype(np.int8)
    out["is_bases_loaded"] = (out["num_runners_on"] == 3).astype(np.int8)
    out["abs_score_diff"] = out["score_diff_pitcher_team"].abs()
    out["is_close_game"] = (out["abs_score_diff"] <= 2).astype(np.int8)
    out["is_late_game"] = (out["inning"] >= 7).astype(np.int8)
    out["li_log"] = np.log1p(out["li"].clip(lower=0))
    out["is_high_leverage"] = (out["li"] >= 2.0).astype(np.int8)
    out["is_clutch"] = (
        (out["inning"] >= 7)
        & (out["abs_score_diff"] <= 2)
        & (out["li"] > 1.2)
    ).astype(np.int8)
    out["win_expectancy_diff"] = (
        out["home_win_expectancy"] - out["away_win_expectancy"]
    )
    out["stadium_owner_team"] = np.where(
        out["top_bottom"].eq("T"),
        out["pitcher_team_id"],
        out["batter_team_id"],
    )
    out["same_hand"] = (
        out["pitcher_hand"].astype(str) == out["batter_hand"].astype(str)
    ).astype(np.int8)

    pitcher_n = pd.to_numeric(out["asof_pitcher_n"], errors="coerce").fillna(0).astype(np.float32)
    for col in PITCHER_RATE_COLS:
        raw = pd.to_numeric(out[col], errors="coerce").fillna(prior).astype(np.float32)
        out[f"{col}_smoothed"] = (
            (pitcher_n * raw + pitcher_strength * prior)
            / (pitcher_n + pitcher_strength)
        ).astype(np.float32)

    batter_n = pd.to_numeric(out["asof_batter_n"], errors="coerce").fillna(0).astype(np.float32)
    for col in BATTER_RATE_COLS:
        raw = pd.to_numeric(out[col], errors="coerce").fillna(prior).astype(np.float32)
        out[f"{col}_smoothed"] = (
            (batter_n * raw + batter_strength * prior)
            / (batter_n + batter_strength)
        ).astype(np.float32)

    out["pitcher_history_log_n"] = np.log1p(out["asof_pitcher_n"].fillna(0))
    out["batter_history_log_n"] = np.log1p(out["asof_batter_n"].fillna(0))
    out["pitchmix_history_log_n"] = np.log1p(out["asof_pitcher_pitchmix_n"].fillna(0))
    out["pitcher_cold_start"] = (out["asof_pitcher_n"].fillna(0) < 50).astype(np.int8)
    out["batter_cold_start"] = (out["asof_batter_n"].fillna(0) < 50).astype(np.int8)

    baseline_success = out["asof_pitcher_success_rate_smoothed"]
    baseline_middle = out["asof_pitcher_middle_rate_smoothed"]
    for k in (1, 3, 5):
        out[f"diff_prev{k}_success"] = (
            out[f"asof_pitcher_prev{k}_game_success_rate"] - baseline_success
        )
        out[f"diff_prev{k}_middle"] = (
            out[f"asof_pitcher_prev{k}_game_middle_rate"] - baseline_middle
        )

    out["pitcher_volatility"] = (
        out["diff_prev1_success"].abs() + out["diff_prev3_success"].abs()
    )
    out["pressure_interaction"] = out["pitcher_volatility"] * out["li_log"]
    out["recent_form_leverage"] = out["diff_prev3_success"] * out["li_log"]

    eps = 1e-5
    out["strike_to_ball_ratio"] = (
        out["asof_pitcher_strike_rate_smoothed"]
        / (out["asof_pitcher_ball_rate_smoothed"] + eps)
    )
    out["reverse_to_success_ratio"] = (
        out["asof_pitcher_reverse_rate_smoothed"]
        / (out["asof_pitcher_success_rate_smoothed"] + eps)
    )
    out["middle_to_success_ratio"] = (
        out["asof_pitcher_middle_rate_smoothed"]
        / (out["asof_pitcher_success_rate_smoothed"] + eps)
    )

    pitchmix_cols = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    p = out[pitchmix_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    p_entropy = p.clip(lower=eps, upper=1.0)
    out["pitchmix_entropy"] = -(p_entropy * np.log(p_entropy)).sum(axis=1)
    out["pitchmix_max_share"] = p.max(axis=1)

    return add_trackman_features(out, feature_state.get("trackman"))


def preprocess(df: pd.DataFrame, state) -> pd.DataFrame:
    result = {}
    for col in state["cat_cols"]:
        values = df[col].fillna("__MISSING__").astype(str)
        result[col] = (
            values.map(state["category_maps"][col]).fillna(-1).astype(np.int32)
        )
    for col in state["num_cols"]:
        values = pd.to_numeric(df[col], errors="coerce")
        result[col] = values.fillna(state["numeric_medians"][col]).astype(np.float32)
    X = pd.DataFrame(result, index=df.index)
    return X[state["feature_names"]]


def validate_inputs(test: pd.DataFrame, sample: pd.DataFrame, bundle):
    if ID_COL not in test.columns:
        raise ValueError(f"test.csv missing {ID_COL}")
    if TARGET_COL in test.columns:
        raise ValueError(f"test.csv must not contain {TARGET_COL}")
    if test[ID_COL].duplicated().any():
        raise ValueError("test row_id must be unique")
    if list(sample.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Invalid sample_submission columns: {list(sample.columns)}")
    if sample[ID_COL].duplicated().any():
        raise ValueError("sample_submission row_id must be unique")

    expected = set(bundle["expected_raw_columns"])
    missing = expected - set(test.columns)
    if missing:
        raise ValueError(f"test.csv missing trained raw columns: {sorted(missing)}")

    if len(test) != len(sample):
        raise ValueError(f"row count mismatch: test={len(test)}, sample={len(sample)}")
    if set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("test/sample row_id sets do not match exactly")


def main():
    start = time.perf_counter()
    test_path = "./data/test.csv"
    sample_path = "./data/sample_submission.csv"
    model_path = "./model/lgb_model.txt"
    bundle_path = "./model/bundle.pkl"
    output_path = "./output/submission.csv"

    print("[1/6] Load model bundle")
    bundle = joblib.load(bundle_path)
    if bundle.get("bundle_version") != 1:
        raise ValueError(f"Unsupported bundle version: {bundle.get('bundle_version')}")
    model = lgb.Booster(model_file=model_path)

    print("[2/6] Load test data")
    test = load_csv(test_path)
    sample = load_csv(sample_path)
    validate_inputs(test, sample, bundle)

    print(f"[3/6] Build safe row-wise features: rows={len(test):,}")
    ids = test[ID_COL].copy()
    features = build_features(test, bundle["feature_state"])
    del test
    gc.collect()

    print("[4/6] Apply train-fitted preprocessing")
    X = preprocess(features, bundle["preprocessor_state"])
    del features
    gc.collect()

    print(f"[5/6] LightGBM inference: features={X.shape[1]}")
    pred = np.asarray(model.predict(X, num_threads=6), dtype=np.float64)
    del X
    gc.collect()

    if pred.shape != (len(ids),):
        raise ValueError(f"prediction shape mismatch: {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("predictions contain NaN/Inf")
    if ((pred < 0.0) | (pred > 1.0)).any():
        raise ValueError("predictions are outside [0, 1]")

    print("[6/6] Build submission in official sample order")
    pred_df = pd.DataFrame({ID_COL: ids.to_numpy(), TARGET_COL: pred})
    submission = sample[[ID_COL]].merge(
        pred_df,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if submission[TARGET_COL].isna().any():
        raise ValueError("missing predictions after row_id merge")
    if len(submission) != len(sample):
        raise ValueError("submission row count changed unexpectedly")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(
        f"[DONE] {output_path} rows={len(submission):,} "
        f"pred_mean={submission[TARGET_COL].mean():.6f} "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()

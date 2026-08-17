from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_PRE_PITCH_COLUMNS = {
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
}


def load_csv(path: Path | str) -> pd.DataFrame:
    print(f"[DATA] Loading {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def validate_train_schema(df: pd.DataFrame, target_col: str, id_col: str) -> None:
    missing = (REQUIRED_PRE_PITCH_COLUMNS | {target_col, id_col}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing train columns: {sorted(missing)}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{id_col} must be unique.")
    labels = set(df[target_col].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{target_col} must be binary; found {sorted(labels)}")
    seasons = sorted(int(value) for value in df["season"].dropna().unique())
    if seasons != [2019, 2020, 2021, 2022, 2023, 2024]:
        raise ValueError(f"Unexpected train seasons: {seasons}")


def validate_test_schema(df: pd.DataFrame, target_col: str, id_col: str) -> None:
    if target_col in df.columns:
        raise ValueError(f"{target_col} must not exist in test.csv")
    missing = REQUIRED_PRE_PITCH_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{id_col} must be unique in test.csv")

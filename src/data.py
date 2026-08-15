# src/data.py

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def load_train(
    path: Path,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    print(f"[DATA] Loading train: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=usecols,
        low_memory=False,
    )

    print(
        f"[DATA] train shape={df.shape}, "
        f"season={df['season'].min()}~{df['season'].max()}"
    )

    return df


def load_test(
    path: Path,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    print(f"[DATA] Loading test: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=usecols,
        low_memory=False,
    )

    print(f"[DATA] test shape={df.shape}")

    return df


def validate_train_schema(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> None:
    required = {
        id_col,
        target_col,
        "season",
        "balls_before",
        "strikes_before",
        "inning",
        "li",
        "pitcher_id",
        "batter_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required train columns: {sorted(missing)}"
        )

    if df[id_col].duplicated().any():
        raise ValueError(f"{id_col} must be unique.")

    if not set(df[target_col].dropna().unique()).issubset({0, 1}):
        raise ValueError(
            f"{target_col} must contain only binary labels."
        )


def validate_test_schema(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
) -> None:
    if target_col in df.columns:
        raise ValueError(
            f"{target_col} must not exist in test data."
        )

    if id_col not in df.columns:
        raise ValueError(
            f"{id_col} is missing in test data."
        )

    if df[id_col].duplicated().any():
        raise ValueError(
            f"{id_col} must be unique in test data."
        )
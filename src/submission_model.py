from __future__ import annotations

from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ModelConfig


def train_lightgbm_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    categorical_cols: Sequence[str],
    config: ModelConfig,
    num_boost_round: int,
):
    categorical_feature = [c for c in categorical_cols if c in X_train.columns]
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        categorical_feature=categorical_feature,
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": config.learning_rate,
        "num_leaves": config.lgb_num_leaves,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": config.random_seed,
        "feature_fraction_seed": config.random_seed,
        "bagging_seed": config.random_seed,
        "data_random_seed": config.random_seed,
        "num_threads": 6,
        "force_col_wise": True,
        "deterministic": True,
    }
    return lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=int(num_boost_round),
        callbacks=[lgb.log_evaluation(period=50)],
    )

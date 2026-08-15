from __future__ import annotations

import gc
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from catboost import (
    CatBoostClassifier,
)

from src.config import ModelConfig


def _xgb_brier_metric(
    predt,
    dtrain,
):
    y = dtrain.get_label()

    score = np.mean(
        (predt - y) ** 2
    )

    return (
        "brier",
        float(score),
    )


def _lgb_brier_metric(
    pred,
    dataset,
):
    y = dataset.get_label()

    score = np.mean(
        (pred - y) ** 2
    )

    return (
        "brier",
        float(score),
        False,
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    config: ModelConfig,
):

    print(
        "[XGB] Building "
        "QuantileDMatrix..."
    )

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        weight=sample_weight,
        feature_names=list(
            X_train.columns
        ),
    )

    dvalid = xgb.QuantileDMatrix(
        X_valid,
        label=y_valid,
        feature_names=list(
            X_valid.columns
        ),
        ref=dtrain,
    )

    params = {
        "objective": (
            "binary:logistic"
        ),
        "learning_rate": (
            config.learning_rate
        ),
        "max_depth": (
            config.xgb_max_depth
        ),
        "min_child_weight": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "seed": (
            config.random_seed
        ),
        "verbosity": 1,
        "disable_default_eval_metric": 1,
    }

    try:
        import torch

        if torch.cuda.is_available():
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"

    except Exception:
        params["device"] = "cpu"

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=(
            config
            .xgb_num_boost_round
        ),
        evals=[
            (
                dvalid,
                "valid",
            )
        ],
        custom_metric=(
            _xgb_brier_metric
        ),
        maximize=False,
        early_stopping_rounds=(
            config
            .early_stopping_rounds
        ),
        verbose_eval=50,
    )

    best_iteration = (
        model.best_iteration
        if model.best_iteration
        is not None
        else config
        .xgb_num_boost_round
        - 1
    )

    pred = model.predict(
        dvalid,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )

    del dtrain
    del dvalid

    gc.collect()

    return model, pred


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    categorical_cols: Sequence[str],
    config: ModelConfig,
):

    categorical_feature = [
        c
        for c in categorical_cols
        if c in X_train.columns
    ]

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        categorical_feature=(
            categorical_feature
        ),
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=(
            categorical_feature
        ),
        reference=train_set,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": (
            config.learning_rate
        ),
        "num_leaves": (
            config.lgb_num_leaves
        ),
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": (
            config.random_seed
        ),
        "num_threads": 6,
        "force_col_wise": True,
    }

    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=(
            config
            .lgb_num_boost_round
        ),
        valid_sets=[
            valid_set
        ],
        valid_names=[
            "valid"
        ],
        feval=(
            _lgb_brier_metric
        ),
        callbacks=[
            lgb.early_stopping(
                config
                .early_stopping_rounds
            ),
            lgb.log_evaluation(
                period=50
            ),
        ],
    )

    pred = model.predict(
        X_valid,
        num_iteration=(
            model.best_iteration
        ),
    )

    del train_set
    del valid_set

    gc.collect()

    return model, pred


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    categorical_indices: Sequence[int],
    config: ModelConfig,
):

    params = {
        "iterations": (
            config.cat_iterations
        ),
        "learning_rate": (
            config.learning_rate
        ),
        "depth": (
            config.cat_depth
        ),
        "loss_function": (
            "Logloss"
        ),
        "eval_metric": (
            "BrierScore"
        ),
        "random_seed": (
            config.random_seed
        ),
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "verbose": 50,
        "allow_writing_files": False,
    }

    try:
        import torch

        if torch.cuda.is_available():
            params[
                "task_type"
            ] = "GPU"
        else:
            params[
                "task_type"
            ] = "CPU"

    except Exception:
        params[
            "task_type"
        ] = "CPU"

    model = CatBoostClassifier(
        **params
    )

    model.fit(
        X_train,
        y_train,
        cat_features=list(
            categorical_indices
        ),
        sample_weight=(
            sample_weight
        ),
        eval_set=(
            X_valid,
            y_valid,
        ),
        early_stopping_rounds=(
            config
            .early_stopping_rounds
        ),
        use_best_model=True,
    )

    pred = (
        model
        .predict_proba(
            X_valid
        )[:, 1]
    )

    return model, pred
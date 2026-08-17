from __future__ import annotations

import gc
from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from src.config import ModelConfig


MODEL_ORDER = ("xgb", "cat")


def _xgb_brier_metric(prediction, dmatrix):
    target = dmatrix.get_label()
    score = np.mean((np.asarray(prediction) - target) ** 2)
    return "brier", float(score)


def _xgb_params(config: ModelConfig) -> dict:
    return {
        "objective": "binary:logistic",
        "disable_default_eval_metric": 1,
        "learning_rate": config.xgb_learning_rate,
        "max_depth": config.xgb_max_depth,
        "min_child_weight": 30.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 8.0,
        "reg_alpha": 0.05,
        "gamma": 0.0,
        "tree_method": "hist",
        "max_bin": 255,
        "device": "cpu",
        "seed": config.random_seed,
        "nthread": config.num_threads,
        "verbosity": 1,
    }


def train_xgboost_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    config: ModelConfig,
):
    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        weight=sample_weight,
        feature_names=list(X_train.columns),
    )
    dvalid = xgb.QuantileDMatrix(
        X_valid,
        label=y_valid,
        feature_names=list(X_valid.columns),
        ref=dtrain,
    )
    model = xgb.train(
        params=_xgb_params(config),
        dtrain=dtrain,
        num_boost_round=config.xgb_num_boost_round,
        evals=[(dvalid, "valid")],
        custom_metric=_xgb_brier_metric,
        maximize=False,
        early_stopping_rounds=config.xgb_early_stopping_rounds,
        verbose_eval=50,
    )
    best_iteration = (
        int(model.best_iteration)
        if model.best_iteration is not None
        else config.xgb_num_boost_round - 1
    )
    prediction = model.predict(dvalid, iteration_range=(0, best_iteration + 1))
    del dtrain, dvalid
    gc.collect()
    return model, np.asarray(prediction, dtype=np.float64), best_iteration + 1


def train_xgboost_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    config: ModelConfig,
    num_boost_round: int,
):
    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        weight=sample_weight,
        feature_names=list(X_train.columns),
    )
    model = xgb.train(
        params=_xgb_params(config),
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        verbose_eval=50,
    )
    del dtrain
    gc.collect()
    return model


def _catboost_params(config: ModelConfig, iterations: int | None = None) -> dict:
    params = {
        "iterations": int(iterations or config.cat_iterations),
        "learning_rate": config.cat_learning_rate,
        "depth": config.cat_depth,
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "random_seed": config.random_seed,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.35,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "verbose": 50,
        "allow_writing_files": False,
        "thread_count": config.num_threads,
        "task_type": config.cat_task_type.upper(),
    }
    return params


def train_catboost_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    categorical_indices: Sequence[int],
    config: ModelConfig,
):
    model = CatBoostClassifier(**_catboost_params(config))
    model.fit(
        X_train,
        y_train,
        cat_features=list(categorical_indices),
        sample_weight=sample_weight,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=config.cat_early_stopping_rounds,
        use_best_model=True,
    )
    best_iteration = int(model.get_best_iteration())
    if best_iteration < 0:
        best_iteration = config.cat_iterations - 1
    prediction = model.predict_proba(X_valid)[:, 1]
    return model, np.asarray(prediction, dtype=np.float64), best_iteration + 1


def train_catboost_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    categorical_indices: Sequence[int],
    config: ModelConfig,
    iterations: int,
):
    model = CatBoostClassifier(**_catboost_params(config, iterations=iterations))
    model.fit(
        X_train,
        y_train,
        cat_features=list(categorical_indices),
        sample_weight=sample_weight,
    )
    return model

from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import pandas as pd
try:  # Legacy ablation backend; not required by the submitted runtime.
    import xgboost as xgb
except ImportError:  # pragma: no cover - exercised in lightweight test images
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - exercised in lightweight test images
    CatBoostClassifier = None

from src.config import ModelConfig


# The submitted ensemble intentionally keeps one tree family and one
# out-of-family learner.  XGBoost helpers remain below for reproducible legacy
# ablations, but XGBoost is not trained or packaged by the TabM strategy.
GBDT_MODEL_ORDER = ("cat",)
RESNET_MODEL_ORDER = (*GBDT_MODEL_ORDER, "resnet")
MODEL_ORDER = (*GBDT_MODEL_ORDER, "tabm")


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
        "min_child_weight": config.xgb_min_child_weight,
        "subsample": config.xgb_subsample,
        "colsample_bytree": config.xgb_colsample_bytree,
        "reg_lambda": config.xgb_reg_lambda,
        "reg_alpha": config.xgb_reg_alpha,
        "gamma": config.xgb_gamma,
        "tree_method": "hist",
        "max_bin": int(config.xgb_max_bin),
        "device": config.xgb_device.lower(),
        "seed": config.random_seed,
        "nthread": config.num_threads,
        "verbosity": 1,
    }


def _quantile_dmatrix(
    X: pd.DataFrame,
    config: ModelConfig,
    *,
    label=None,
    weight=None,
    ref=None,
):
    """Construct every quantized matrix with the Booster's bin contract.

    XGBoost stores the quantization cut structure in QuantileDMatrix. Its
    max_bin must match the hist Booster parameter, including validation
    matrices constructed with a training reference.
    """
    if xgb is None:
        raise ImportError("xgboost is required for the legacy XGBoost ablation.")
    return xgb.QuantileDMatrix(
        X,
        label=label,
        weight=weight,
        feature_names=list(X.columns),
        ref=ref,
        max_bin=int(config.xgb_max_bin),
        nthread=int(config.num_threads),
    )


def validate_xgboost_backend(config: ModelConfig) -> None:
    """Fail fast on an incompatible XGBoost matrix/Booster contract.

    This one-round check runs before the 1.47M-row feature build. It exercises
    the same QuantileDMatrix + hist path used by temporal and final training.
    """
    if xgb is None:
        raise ImportError("xgboost is not installed.")
    if int(config.xgb_max_bin) < 2:
        raise ValueError(f"xgb_max_bin must be >= 2; got {config.xgb_max_bin}")
    if config.xgb_device.lower() not in {"cpu", "cuda"}:
        raise ValueError(f"xgb_device must be cpu or cuda; got {config.xgb_device}")

    X_probe = pd.DataFrame(
        {
            "probe_a": np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
            "probe_b": np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.float32),
        }
    )
    y_probe = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.float32)
    dprobe = _quantile_dmatrix(X_probe, config, label=y_probe)
    params = _xgb_params(config)
    params.update({"verbosity": 0, "min_child_weight": 1.0})
    probe_model = xgb.train(params=params, dtrain=dprobe, num_boost_round=1)
    prediction = np.asarray(probe_model.predict(dprobe), dtype=np.float64)
    if prediction.shape != y_probe.shape or not np.isfinite(prediction).all():
        raise RuntimeError("XGBoost backend preflight returned invalid predictions.")
    print(
        f"[BACKEND] XGBoost {xgb.__version__} preflight PASS "
        f"(device={config.xgb_device.lower()}, "
        f"QuantileDMatrix/hist max_bin={config.xgb_max_bin})"
    )
    del probe_model, dprobe, X_probe, y_probe, prediction
    gc.collect()


def train_xgboost_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    config: ModelConfig,
):
    dtrain = _quantile_dmatrix(
        X_train,
        config,
        label=y_train,
        weight=sample_weight,
    )
    dvalid = _quantile_dmatrix(
        X_valid,
        config,
        label=y_valid,
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
    dtrain = _quantile_dmatrix(
        X_train,
        config,
        label=y_train,
        weight=sample_weight,
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
        "l2_leaf_reg": config.cat_l2_leaf_reg,
        "random_strength": config.cat_random_strength,
        "bootstrap_type": config.cat_bootstrap_type,
        "bagging_temperature": config.cat_bagging_temperature,
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
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed.")
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
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed.")
    model = CatBoostClassifier(**_catboost_params(config, iterations=iterations))
    model.fit(
        X_train,
        y_train,
        cat_features=list(categorical_indices),
        sample_weight=sample_weight,
    )
    return model

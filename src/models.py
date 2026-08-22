from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.config import ModelConfig
from src.xgb_multiview import (
    apply_numeric_pca_state,
    build_representation_view,
    fit_numeric_pca_state,
    make_representative_sample_weight,
)


GBDT_MODEL_ORDER = ("xgb", "lgb", "cat")
RESNET_MODEL_ORDER = (*GBDT_MODEL_ORDER, "resnet")
MODEL_ORDER = (*GBDT_MODEL_ORDER, "resnet", "ft_transformer")


# ============================================================
# XGBoost
# ============================================================

def _xgb_brier_metric(prediction, dmatrix):
    target = dmatrix.get_label()
    score = np.mean((np.asarray(prediction) - target) ** 2)
    return "brier", float(score)


def _xgb_params(
    config: ModelConfig,
    *,
    seed: int | None = None,
) -> dict:
    resolved_seed = (
        int(config.random_seed)
        if seed is None
        else int(seed)
    )

    return {
        "objective": "binary:logistic",
        "disable_default_eval_metric": 1,
        "learning_rate": (
            config.xgb_learning_rate
        ),
        "max_depth": (
            config.xgb_max_depth
        ),
        "min_child_weight": (
            config.xgb_min_child_weight
        ),
        "subsample": (
            config.xgb_subsample
        ),
        "colsample_bytree": (
            config.xgb_colsample_bytree
        ),
        "reg_lambda": (
            config.xgb_reg_lambda
        ),
        "reg_alpha": (
            config.xgb_reg_alpha
        ),
        "gamma": (
            config.xgb_gamma
        ),
        "tree_method": "hist",
        "max_bin": int(
            config.xgb_max_bin
        ),
        "device": (
            config.xgb_device.lower()
        ),
        "seed": resolved_seed,
        "nthread": (
            config.num_threads
        ),
        "verbosity": 1,
    }


def _validated_xgb_bagging_seeds(
    config: ModelConfig,
) -> tuple[int, ...]:
    seeds = tuple(
        int(seed)
        for seed
        in config.xgb_bagging_seeds
    )

    if not seeds:
        raise ValueError(
            "xgb_bagging_seeds "
            "must not be empty."
        )

    if len(seeds) != len(
        set(seeds)
    ):
        raise ValueError(
            "xgb_bagging_seeds "
            "must be unique."
        )

    if seeds[0] != int(
        config.random_seed
    ):
        raise ValueError(
            "The first XGBoost bagging "
            "seed must equal random_seed "
            "so seed-2026 remains the "
            "validated V10 baseline."
        )

    return seeds


def _quantile_dmatrix(
    X: pd.DataFrame,
    config: ModelConfig,
    *,
    label=None,
    weight=None,
    ref=None,
):
    if xgb is None:
        raise ImportError("xgboost is required.")

    return xgb.QuantileDMatrix(
        X,
        label=label,
        weight=weight,
        feature_names=list(X.columns),
        ref=ref,
        max_bin=int(config.xgb_max_bin),
        nthread=int(config.num_threads),
    )


def select_xgb_gain_features(
    model,
    feature_names: Sequence[str],
    *,
    top_k: int,
) -> list[str]:
    names = [
        str(name)
        for name in feature_names
    ]

    top_k = int(
        top_k
    )

    if (
        top_k <= 0
        or top_k > len(names)
    ):
        raise ValueError(
            "Invalid XGBoost Top-K "
            f"feature count: {top_k}"
        )

    gain = model.get_score(
        importance_type="gain"
    )

    original_index = {
        name: index
        for index, name
        in enumerate(names)
    }

    ranked = sorted(
        names,
        key=lambda name: (
            -float(
                gain.get(
                    name,
                    0.0,
                )
            ),
            original_index[name],
        ),
    )

    selected = ranked[
        :top_k
    ]

    if len(set(selected)) != len(
        selected
    ):
        raise RuntimeError(
            "Duplicate selected "
            "XGBoost features."
        )

    return selected


def validate_xgboost_backend(config: ModelConfig) -> None:
    if xgb is None:
        raise ImportError("xgboost is not installed.")

    X_probe = pd.DataFrame(
        {
            "probe_a": np.asarray(
                [0, 1, 2, 3, 4, 5, 6, 7],
                dtype=np.float32,
            ),
            "probe_b": np.asarray(
                [1, 1, 0, 0, 1, 1, 0, 0],
                dtype=np.float32,
            ),
        }
    )
    y_probe = np.asarray(
        [0, 0, 0, 1, 0, 1, 1, 1],
        dtype=np.float32,
    )

    dprobe = _quantile_dmatrix(
        X_probe,
        config,
        label=y_probe,
    )

    params = _xgb_params(config)
    params.update(
        {
            "verbosity": 0,
            "min_child_weight": 1.0,
        }
    )

    model = xgb.train(
        params=params,
        dtrain=dprobe,
        num_boost_round=1,
    )

    pred = np.asarray(
        model.predict(dprobe),
        dtype=np.float64,
    )

    if pred.shape != y_probe.shape:
        raise RuntimeError("Invalid XGBoost prediction shape.")

    if not np.isfinite(pred).all():
        raise RuntimeError("XGBoost produced non-finite prediction.")

    print(
        f"[BACKEND] XGBoost {xgb.__version__} PASS "
        f"(device={config.xgb_device}, max_bin={config.xgb_max_bin})"
    )

    del model, dprobe, X_probe, y_probe, pred
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

    prediction = model.predict(
        dvalid,
        iteration_range=(0, best_iteration + 1),
    )

    del dtrain, dvalid
    gc.collect()

    return (
        model,
        np.asarray(prediction, dtype=np.float64),
        best_iteration + 1,
    )


def train_xgboost_bagged_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    config: ModelConfig,
):
    seeds = (
        _validated_xgb_bagging_seeds(
            config
        )
    )

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

    # --------------------------------------------------------
    # Seed 2026 remains exactly the V10 model and determines
    # early stopping.  Additional seeds use the same number
    # of rounds so bagging does not confound the experiment
    # with per-seed iteration tuning.
    # --------------------------------------------------------

    base_seed = seeds[0]

    base_model = xgb.train(
        params=_xgb_params(
            config,
            seed=base_seed,
        ),
        dtrain=dtrain,
        num_boost_round=(
            config.xgb_num_boost_round
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
            .xgb_early_stopping_rounds
        ),
        verbose_eval=50,
    )

    base_best_iteration = (
        int(
            base_model.best_iteration
        )
        if (
            base_model.best_iteration
            is not None
        )
        else (
            config.xgb_num_boost_round
            - 1
        )
    )

    num_rounds = (
        base_best_iteration + 1
    )

    base_prediction = (
        np.asarray(
            base_model.predict(
                dvalid,
                iteration_range=(
                    0,
                    num_rounds,
                ),
            ),
            dtype=np.float64,
        )
    )

    models = [
        base_model
    ]

    seed_predictions = [
        base_prediction
    ]

    print(
        "[XGB-BAG] "
        f"seed={base_seed} "
        f"rounds={num_rounds}"
    )

    for seed in seeds[1:]:
        model = xgb.train(
            params=_xgb_params(
                config,
                seed=seed,
            ),
            dtrain=dtrain,
            num_boost_round=(
                num_rounds
            ),
            verbose_eval=False,
        )

        prediction = (
            np.asarray(
                model.predict(
                    dvalid
                ),
                dtype=np.float64,
            )
        )

        if (
            prediction.shape
            != base_prediction.shape
        ):
            raise RuntimeError(
                "XGBoost seed prediction "
                "shape mismatch."
            )

        if not np.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                "XGBoost seed prediction "
                "contains NaN/inf."
            )

        models.append(
            model
        )

        seed_predictions.append(
            prediction
        )

        print(
            "[XGB-BAG] "
            f"seed={seed} "
            f"rounds={num_rounds}"
        )

    stacked = np.vstack(
        seed_predictions
    )

    bagged_prediction = (
        stacked.mean(
            axis=0,
            dtype=np.float64,
        )
    )

    del (
        dtrain,
        dvalid,
        stacked,
        seed_predictions,
    )

    gc.collect()

    return (
        models,
        bagged_prediction,
        int(num_rounds),
        base_prediction,
    )


def train_xgboost_multiview_fold(
    *,
    base_model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    sample_weight: np.ndarray,
    numerical_cols: Sequence[str],
    config: ModelConfig,
    num_boost_round: int,
):
    if not config.xgb_multiview_enabled:
        raise ValueError(
            "XGBoost multi-view is disabled."
        )

    top_raw = (
        select_xgb_gain_features(
            base_model,
            X_train.columns,
            top_k=(
                config
                .xgb_multiview_top_raw_features
            ),
        )
    )

    pca_state = (
        fit_numeric_pca_state(
            X_train,
            numerical_cols,
            n_components=(
                config
                .xgb_multiview_pca_components
            ),
        )
    )

    representation_train = (
        build_representation_view(
            X_train,
            top_raw,
            pca_state,
        )
    )

    representation_valid = (
        build_representation_view(
            X_valid,
            top_raw,
            pca_state,
        )
    )

    if representation_train.shape[1] != (
        config
        .xgb_multiview_top_raw_features
        + config
        .xgb_multiview_pca_components
    ):
        raise RuntimeError(
            "Unexpected representation "
            "view width."
        )

    d_repr_train = (
        _quantile_dmatrix(
            representation_train,
            config,
            label=y_train,
            weight=sample_weight,
        )
    )

    d_repr_valid = (
        _quantile_dmatrix(
            representation_valid,
            config,
            ref=d_repr_train,
        )
    )

    representation_model = (
        xgb.train(
            params=_xgb_params(
                config,
                seed=(
                    config
                    .xgb_multiview_representation_seed
                ),
            ),
            dtrain=d_repr_train,
            num_boost_round=int(
                num_boost_round
            ),
            verbose_eval=False,
        )
    )

    representation_prediction = (
        np.asarray(
            representation_model.predict(
                d_repr_valid
            ),
            dtype=np.float64,
        )
    )

    (
        representative_weight,
        representative_diagnostics,
    ) = (
        make_representative_sample_weight(
            X_train,
            sample_weight,
            group_columns=(
                config
                .xgb_multiview_group_columns
            ),
            leverage_column=(
                config
                .xgb_multiview_leverage_column
            ),
            leverage_bins=(
                config
                .xgb_multiview_leverage_bins
            ),
            clip_low=(
                config
                .xgb_multiview_repr_weight_clip_low
            ),
            clip_high=(
                config
                .xgb_multiview_repr_weight_clip_high
            ),
        )
    )

    d_representative_train = (
        _quantile_dmatrix(
            X_train,
            config,
            label=y_train,
            weight=(
                representative_weight
            ),
        )
    )

    d_representative_valid = (
        _quantile_dmatrix(
            X_valid,
            config,
            ref=(
                d_representative_train
            ),
        )
    )

    representative_model = (
        xgb.train(
            params=_xgb_params(
                config,
                seed=(
                    config
                    .xgb_multiview_representative_seed
                ),
            ),
            dtrain=(
                d_representative_train
            ),
            num_boost_round=int(
                num_boost_round
            ),
            verbose_eval=False,
        )
    )

    representative_prediction = (
        np.asarray(
            representative_model.predict(
                d_representative_valid
            ),
            dtype=np.float64,
        )
    )

    del (
        d_repr_train,
        d_repr_valid,
        d_representative_train,
        d_representative_valid,
        representation_train,
        representation_valid,
        representative_weight,
    )

    gc.collect()

    return (
        representation_model,
        representative_model,
        representation_prediction,
        representative_prediction,
        top_raw,
        pca_state,
        representative_diagnostics,
    )


def train_xgboost_full_bagged(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    config: ModelConfig,
    num_boost_round: int,
):
    seeds = (
        _validated_xgb_bagging_seeds(
            config
        )
    )

    dtrain = _quantile_dmatrix(
        X_train,
        config,
        label=y_train,
        weight=sample_weight,
    )

    models = []

    for seed in seeds:
        print(
            "[XGB-BAG/FINAL] "
            f"seed={seed} "
            f"rounds={num_boost_round}"
        )

        model = xgb.train(
            params=_xgb_params(
                config,
                seed=seed,
            ),
            dtrain=dtrain,
            num_boost_round=int(
                num_boost_round
            ),
            verbose_eval=False,
        )

        models.append(
            model
        )

    del dtrain
    gc.collect()

    return models


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


def train_xgboost_multiview_full(
    *,
    base_model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    numerical_cols: Sequence[str],
    config: ModelConfig,
    num_boost_round: int,
):
    top_raw = (
        select_xgb_gain_features(
            base_model,
            X_train.columns,
            top_k=(
                config
                .xgb_multiview_top_raw_features
            ),
        )
    )

    pca_state = (
        fit_numeric_pca_state(
            X_train,
            numerical_cols,
            n_components=(
                config
                .xgb_multiview_pca_components
            ),
        )
    )

    representation_train = (
        build_representation_view(
            X_train,
            top_raw,
            pca_state,
        )
    )

    d_repr = _quantile_dmatrix(
        representation_train,
        config,
        label=y_train,
        weight=sample_weight,
    )

    representation_model = xgb.train(
        params=_xgb_params(
            config,
            seed=(
                config
                .xgb_multiview_representation_seed
            ),
        ),
        dtrain=d_repr,
        num_boost_round=int(
            num_boost_round
        ),
        verbose_eval=False,
    )

    (
        representative_weight,
        representative_diagnostics,
    ) = (
        make_representative_sample_weight(
            X_train,
            sample_weight,
            group_columns=(
                config
                .xgb_multiview_group_columns
            ),
            leverage_column=(
                config
                .xgb_multiview_leverage_column
            ),
            leverage_bins=(
                config
                .xgb_multiview_leverage_bins
            ),
            clip_low=(
                config
                .xgb_multiview_repr_weight_clip_low
            ),
            clip_high=(
                config
                .xgb_multiview_repr_weight_clip_high
            ),
        )
    )

    d_representative = (
        _quantile_dmatrix(
            X_train,
            config,
            label=y_train,
            weight=(
                representative_weight
            ),
        )
    )

    representative_model = xgb.train(
        params=_xgb_params(
            config,
            seed=(
                config
                .xgb_multiview_representative_seed
            ),
        ),
        dtrain=d_representative,
        num_boost_round=int(
            num_boost_round
        ),
        verbose_eval=False,
    )

    del (
        d_repr,
        d_representative,
        representation_train,
        representative_weight,
    )

    gc.collect()

    return (
        representation_model,
        representative_model,
        top_raw,
        pca_state,
        representative_diagnostics,
    )

# ============================================================
# LightGBM
# ============================================================

def _lgb_brier_metric(prediction, dataset):
    target = dataset.get_label()
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    score = np.mean((prediction - target) ** 2)

    return "brier", float(score), False


def _lgb_params(config: ModelConfig) -> dict:
    return {
        "objective": "binary",
        "metric": "None",
        "learning_rate": float(config.lgb_learning_rate),
        "num_leaves": int(config.lgb_num_leaves),
        "max_depth": int(config.lgb_max_depth),
        "min_data_in_leaf": int(config.lgb_min_data_in_leaf),
        "feature_fraction": float(config.lgb_feature_fraction),
        "bagging_fraction": float(config.lgb_bagging_fraction),
        "bagging_freq": int(config.lgb_bagging_freq),
        "lambda_l1": float(config.lgb_lambda_l1),
        "lambda_l2": float(config.lgb_lambda_l2),
        "min_gain_to_split": float(config.lgb_min_gain_to_split),
        "max_bin": int(config.lgb_max_bin),
        "device_type": str(config.lgb_device_type),
        "seed": int(config.random_seed),
        "feature_fraction_seed": int(config.random_seed + 1),
        "bagging_seed": int(config.random_seed + 2),
        "data_random_seed": int(config.random_seed + 3),
        "num_threads": int(config.num_threads),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def validate_lightgbm_backend(config: ModelConfig) -> None:
    if lgb is None:
        raise ImportError("lightgbm is not installed.")

    X_probe = pd.DataFrame(
        {
            "probe_a": np.asarray(
                [0, 1, 2, 3, 4, 5, 6, 7],
                dtype=np.float32,
            ),
            "probe_b": np.asarray(
                [1, 1, 0, 0, 1, 1, 0, 0],
                dtype=np.float32,
            ),
        }
    )
    y_probe = np.asarray(
        [0, 0, 0, 1, 0, 1, 1, 1],
        dtype=np.float32,
    )

    dataset = lgb.Dataset(
        X_probe,
        label=y_probe,
        free_raw_data=False,
    )

    params = _lgb_params(config)
    params["min_data_in_leaf"] = 1

    model = lgb.train(
        params=params,
        train_set=dataset,
        num_boost_round=2,
    )

    prediction = np.asarray(
        model.predict(X_probe),
        dtype=np.float64,
    )

    if prediction.shape != y_probe.shape:
        raise RuntimeError("Invalid LightGBM prediction shape.")

    if not np.isfinite(prediction).all():
        raise RuntimeError("LightGBM produced non-finite prediction.")

    print(
        f"[BACKEND] LightGBM {lgb.__version__} PASS "
        f"(device={config.lgb_device_type})"
    )

    del model, dataset, X_probe, y_probe, prediction
    gc.collect()


def train_lightgbm_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    sample_weight: np.ndarray,
    categorical_indices: Sequence[int],
    config: ModelConfig,
):
    if lgb is None:
        raise ImportError("lightgbm is not installed.")

    categorical_indices = list(categorical_indices)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )

    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=dtrain,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )

    model = lgb.train(
        params=_lgb_params(config),
        train_set=dtrain,
        num_boost_round=int(config.lgb_num_boost_round),
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=_lgb_brier_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(
                    config.lgb_early_stopping_rounds
                ),
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iteration = int(model.best_iteration)

    if best_iteration <= 0:
        best_iteration = int(config.lgb_num_boost_round)

    prediction = np.asarray(
        model.predict(
            X_valid,
            num_iteration=best_iteration,
        ),
        dtype=np.float64,
    )

    del dtrain, dvalid
    gc.collect()

    return model, prediction, best_iteration


def train_lightgbm_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    categorical_indices: Sequence[int],
    config: ModelConfig,
    num_boost_round: int,
):
    if lgb is None:
        raise ImportError("lightgbm is not installed.")

    dataset = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        categorical_feature=list(categorical_indices),
        free_raw_data=False,
    )

    model = lgb.train(
        params=_lgb_params(config),
        train_set=dataset,
        num_boost_round=int(num_boost_round),
    )

    del dataset
    gc.collect()

    return model


# ============================================================
# CatBoost
# ============================================================

def _catboost_params(
    config: ModelConfig,
    iterations: int | None = None,
) -> dict:
    return {
        "iterations": int(
            iterations or config.cat_iterations
        ),
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

    model = CatBoostClassifier(
        **_catboost_params(config)
    )

    model.fit(
        X_train,
        y_train,
        cat_features=list(categorical_indices),
        sample_weight=sample_weight,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=(
            config.cat_early_stopping_rounds
        ),
        use_best_model=True,
    )

    best_iteration = int(model.get_best_iteration())

    if best_iteration < 0:
        best_iteration = (
            config.cat_iterations - 1
        )

    prediction = model.predict_proba(
        X_valid
    )[:, 1]

    return (
        model,
        np.asarray(prediction, dtype=np.float64),
        best_iteration + 1,
    )


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

    model = CatBoostClassifier(
        **_catboost_params(
            config,
            iterations=iterations,
        )
    )

    model.fit(
        X_train,
        y_train,
        cat_features=list(categorical_indices),
        sample_weight=sample_weight,
    )

    return model
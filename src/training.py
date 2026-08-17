from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, Mapping

import joblib
import numpy as np
import pandas as pd

from src.calibration import (
    fit_prequential_bias_calibrator,
    select_stable_weights,
)
from src.config import ExperimentConfig
from src.features import LeakageSafeFeatureEngineer, StrictPastTrackmanFeatures
from src.metrics import evaluate_probabilities
from src.models import (
    GBDT_MODEL_ORDER,
    train_catboost_fold,
    train_catboost_full,
    train_xgboost_fold,
    train_xgboost_full,
)
from src.preprocessing import TabularPreprocessor
from src.runtime import apply_probability_bias, blend_predictions
from src.splits import make_temporal_folds, uniform_weights


NEURAL_MODEL_ORDER = ("resnet", "ft_transformer")


def _active_neural_models(config: ExperimentConfig) -> tuple[str, ...]:
    if not config.neural.enabled:
        return ()
    requested = tuple(config.neural.models)
    if not requested:
        raise ValueError("Neural training is enabled but no neural models were selected.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate neural models are not allowed: {requested}")
    unknown = [name for name in requested if name not in NEURAL_MODEL_ORDER]
    if unknown:
        raise ValueError(f"Unknown neural models: {unknown}")
    return requested


def _strategy_name(model_order: tuple[str, ...]) -> str:
    if model_order == GBDT_MODEL_ORDER:
        return "temporal_xgb_cat_probability_bias_cpu_fallback_v1"
    if model_order == (*GBDT_MODEL_ORDER, "resnet"):
        return "temporal_xgb_cat_resnet_probability_bias_v4"
    if model_order == (*GBDT_MODEL_ORDER, *NEURAL_MODEL_ORDER):
        return "temporal_gbdt_resnet_ftt_probability_bias_v3"
    raise ValueError(f"Unsupported model order: {model_order}")


def _print_metrics(label: str, metrics: Mapping[str, float]) -> None:
    print(
        f"[{label}] Brier={metrics['brier']:.8f} "
        f"AUC={metrics.get('auc', float('nan')):.6f} "
        f"target_mean={metrics['target_mean']:.6f} "
        f"pred_mean={metrics['pred_mean']:.6f} "
        f"bias={metrics['calibration_bias']:+.6f}"
    )


def build_feature_table(
    train: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    trackman = None
    if config.use_trackman:
        if not config.paths.trackman_path.exists():
            raise FileNotFoundError(config.paths.trackman_path)
        trackman = StrictPastTrackmanFeatures().fit_from_csv(config.paths.trackman_path)

    engineer = LeakageSafeFeatureEngineer(config.features, trackman_features=trackman)
    state = engineer.export_runtime_state()
    print("[FEATURE] Building row-wise main-table features...")
    features = engineer.transform(train)
    return features, state


def run_temporal_validation(
    train: pd.DataFrame,
    features: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[Dict[str, object], Dict[str, object]]:
    target = config.features.target_col
    fold_results = []
    neural_model_order = _active_neural_models(config)
    model_order = (*GBDT_MODEL_ORDER, *neural_model_order)
    print(f"[ENSEMBLE] active_models={list(model_order)}")

    for fold in make_temporal_folds(train, config.temporal_folds):
        print("\n" + "=" * 88)
        print(
            f"[FOLD] {fold.name}: n_train={len(fold.train_idx):,}, "
            f"n_valid={len(fold.valid_idx):,}"
        )
        print("=" * 88)
        y_train = train.iloc[fold.train_idx][target].to_numpy(dtype=np.float32, copy=True)
        y_valid = train.iloc[fold.valid_idx][target].to_numpy(dtype=np.float32, copy=True)

        preprocessor = TabularPreprocessor(
            categorical_cols=config.features.categorical_cols,
            excluded_cols=config.features.excluded_cols,
        )
        X_train = preprocessor.fit_transform(features.iloc[fold.train_idx])
        X_valid = preprocessor.transform(features.iloc[fold.valid_idx])
        sample_weight = uniform_weights(len(y_train))

        predictions: Dict[str, np.ndarray] = {}
        best_iterations: Dict[str, int] = {}
        per_model_metrics: Dict[str, Dict[str, float]] = {}

        start = time.perf_counter()
        xgb_model, predictions["xgb"], best_iterations["xgb"] = train_xgboost_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            sample_weight,
            config.models,
        )
        elapsed = time.perf_counter() - start
        per_model_metrics["xgb"] = evaluate_probabilities(y_valid, predictions["xgb"])
        per_model_metrics["xgb"]["train_seconds"] = float(elapsed)
        _print_metrics(f"{fold.valid_season}/xgb", per_model_metrics["xgb"])
        del xgb_model
        gc.collect()

        start = time.perf_counter()
        cat_model, predictions["cat"], best_iterations["cat"] = train_catboost_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            sample_weight,
            preprocessor.categorical_indices,
            config.models,
        )
        elapsed = time.perf_counter() - start
        per_model_metrics["cat"] = evaluate_probabilities(y_valid, predictions["cat"])
        per_model_metrics["cat"]["train_seconds"] = float(elapsed)
        _print_metrics(f"{fold.valid_season}/cat", per_model_metrics["cat"])
        del cat_model
        gc.collect()

        if neural_model_order:
            from src.neural import (
                NeuralPreprocessor,
                prepare_neural_arrays,
                release_torch_memory,
                train_neural_fold,
            )

            neural_preprocessor = NeuralPreprocessor(
                categorical_cols=preprocessor.cat_cols_,
                numerical_cols=preprocessor.num_cols_,
            ).fit(X_train)
            neural_state = neural_preprocessor.export_state()
            train_arrays = prepare_neural_arrays(X_train, neural_state)
            valid_arrays = prepare_neural_arrays(X_valid, neural_state)

            for offset, kind in enumerate(neural_model_order, start=1):
                start = time.perf_counter()
                neural_model, prediction, best_epoch, _ = train_neural_fold(
                    kind=kind,
                    train_arrays=train_arrays,
                    y_train=y_train,
                    sample_weight=sample_weight,
                    valid_arrays=valid_arrays,
                    y_valid=y_valid,
                    neural_state=neural_state,
                    config=config.neural,
                    random_seed=config.models.random_seed + 100 * offset + fold.valid_season,
                )
                elapsed = time.perf_counter() - start
                predictions[kind] = prediction
                best_iterations[kind] = int(best_epoch)
                per_model_metrics[kind] = evaluate_probabilities(y_valid, prediction)
                per_model_metrics[kind]["train_seconds"] = float(elapsed)
                _print_metrics(f"{fold.valid_season}/{kind}", per_model_metrics[kind])
                del neural_model, prediction
                release_torch_memory()

            del train_arrays, valid_arrays, neural_state

        del X_train, X_valid, sample_weight
        gc.collect()

        fold_results.append(
            {
                "name": fold.name,
                "valid_season": fold.valid_season,
                "y_true": y_valid,
                "predictions": predictions,
                "best_iterations": best_iterations,
                "metrics": per_model_metrics,
                "n_train": int(len(fold.train_idx)),
                "n_valid": int(len(fold.valid_idx)),
            }
        )

    if len(fold_results) != 2:
        raise RuntimeError("This submission strategy requires 2023 and 2024 holdouts.")

    weights, weight_report = select_stable_weights(
        fold_results,
        fold_importance=config.temporal_fold_importance,
        step=config.ensemble_grid_step,
        model_order=model_order,
    )
    calibrator = fit_prequential_bias_calibrator(
        earlier_fold=fold_results[0],
        recent_fold=fold_results[1],
        weights=weights,
        model_order=model_order,
    )

    fold_summaries = []
    for fold in fold_results:
        ensemble = blend_predictions(
            [fold["predictions"][name] for name in model_order], weights
        )
        ensemble_metrics = evaluate_probabilities(fold["y_true"], ensemble)
        _print_metrics(f"{fold['valid_season']}/ensemble", ensemble_metrics)
        fold_summaries.append(
            {
                "name": fold["name"],
                "valid_season": int(fold["valid_season"]),
                "n_train": fold["n_train"],
                "n_valid": fold["n_valid"],
                "best_iterations": fold["best_iterations"],
                "models": fold["metrics"],
                "ensemble": ensemble_metrics,
            }
        )

    recent = fold_results[1]
    recent_ensemble = blend_predictions(
        [recent["predictions"][name] for name in model_order], weights
    )
    transferred = apply_probability_bias(
        recent_ensemble, calibrator["earlier_bias"]
    )
    transfer_metrics = evaluate_probabilities(recent["y_true"], transferred)
    print(
        f"[CALIBRATION] accepted={calibrator['accepted']} "
        f"2024 raw={calibrator['recent_raw_brier']:.8f} "
        f"2023-bias transfer={calibrator['recent_transferred_brier']:.8f} "
        f"final_bias={calibrator['bias']:+.6f}"
    )

    recent_iterations = fold_results[-1]["best_iterations"]
    final_iterations = {
        "xgb": int(
            min(
                config.models.xgb_num_boost_round,
                max(50, round(recent_iterations["xgb"] * 1.05)),
            )
        ),
        "cat": int(
            min(
                config.models.cat_iterations,
                max(50, round(recent_iterations["cat"] * 1.05)),
            )
        ),
    }
    neural_epoch_caps = {
        "resnet": int(config.neural.resnet_max_epochs),
        "ft_transformer": int(config.neural.ft_max_epochs),
    }
    for kind in neural_model_order:
        final_iterations[kind] = int(
            min(
                neural_epoch_caps[kind],
                max(2, round(recent_iterations[kind] * 1.05)),
            )
        )

    ensemble_state = {
        "model_order": list(model_order),
        "weights": weights.tolist(),
        "calibration": calibrator,
        "final_iterations": final_iterations,
    }
    report = {
        "strategy": _strategy_name(model_order),
        "folds": fold_summaries,
        "weight_selection": weight_report,
        "calibration": calibrator,
        "calibration_transfer_metrics": transfer_metrics,
        "final_iterations": final_iterations,
    }
    return ensemble_state, report


def train_and_save_final_models(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_state: Mapping[str, object],
    ensemble_state: Mapping[str, object],
    config: ExperimentConfig,
    model_dir: Path,
) -> Dict[str, object]:
    model_dir.mkdir(parents=True, exist_ok=True)
    target = config.features.target_col
    y = train[target].to_numpy(dtype=np.float32, copy=True)
    sample_weight = uniform_weights(len(y))

    preprocessor = TabularPreprocessor(
        categorical_cols=config.features.categorical_cols,
        excluded_cols=config.features.excluded_cols,
    )
    X = preprocessor.fit_transform(features)

    xgb_rounds = int(ensemble_state["final_iterations"]["xgb"])
    print(f"[FINAL] Training XGBoost for {xgb_rounds} rounds...")
    xgb_model = train_xgboost_full(
        X,
        y,
        sample_weight,
        config.models,
        num_boost_round=xgb_rounds,
    )
    xgb_path = model_dir / "xgb_model.json"
    xgb_model.save_model(str(xgb_path))
    del xgb_model
    gc.collect()

    cat_iterations = int(ensemble_state["final_iterations"]["cat"])
    print(f"[FINAL] Training CatBoost for {cat_iterations} iterations...")
    cat_model = train_catboost_full(
        X,
        y,
        sample_weight,
        preprocessor.categorical_indices,
        config.models,
        iterations=cat_iterations,
    )
    cat_path = model_dir / "cat_model.cbm"
    cat_model.save_model(str(cat_path))
    del cat_model
    gc.collect()

    neural_state = None
    neural_paths: Dict[str, Path] = {}
    neural_model_order = tuple(ensemble_state["model_order"])[len(GBDT_MODEL_ORDER) :]
    if neural_model_order:
        from src.neural import (
            NeuralPreprocessor,
            prepare_neural_arrays,
            release_torch_memory,
            save_checkpoint,
            train_neural_full,
        )

        neural_preprocessor = NeuralPreprocessor(
            categorical_cols=preprocessor.cat_cols_,
            numerical_cols=preprocessor.num_cols_,
        ).fit(X)
        neural_state = neural_preprocessor.export_state()
        train_arrays = prepare_neural_arrays(X, neural_state)
        for offset, kind in enumerate(neural_model_order, start=1):
            epochs = int(ensemble_state["final_iterations"][kind])
            print(f"[FINAL] Training {kind} for {epochs} epochs...")
            neural_model, model_spec = train_neural_full(
                kind=kind,
                train_arrays=train_arrays,
                y_train=y,
                sample_weight=sample_weight,
                neural_state=neural_state,
                config=config.neural,
                random_seed=config.models.random_seed + 100 * offset + 2025,
                epochs=epochs,
            )
            neural_path = model_dir / f"{kind}.pt"
            save_checkpoint(neural_path, neural_model, model_spec)
            neural_paths[kind] = neural_path
            del neural_model
            release_torch_memory()
        del train_arrays

    del X, sample_weight
    gc.collect()

    raw_feature_cols = [col for col in train.columns if col != target]
    bundle = {
        "bundle_version": 3,
        "id_col": config.features.id_col,
        "target_col": target,
        "expected_raw_columns": raw_feature_cols,
        "feature_state": dict(feature_state),
        "preprocessor_state": preprocessor.export_state(),
        "neural_preprocessor_state": neural_state,
        "ensemble": dict(ensemble_state),
        "training": {
            "train_seasons": [int(train["season"].min()), int(train["season"].max())],
            "n_rows": int(len(train)),
            "target_mean": float(y.mean()),
            "random_seed": int(config.models.random_seed),
        },
    }
    bundle_path = model_dir / "bundle.pkl"
    joblib.dump(bundle, bundle_path, compress=3)

    manifest = {
        "strategy": _strategy_name(tuple(ensemble_state["model_order"])),
        "bundle_version": 3,
        "model_order": list(ensemble_state["model_order"]),
        "weights": list(ensemble_state["weights"]),
        "calibration": dict(ensemble_state["calibration"]),
        "final_iterations": dict(ensemble_state["final_iterations"]),
        "n_features": int(len(preprocessor.feature_names_)),
        "model_files": [
            xgb_path.name,
            cat_path.name,
            *[neural_paths[name].name for name in neural_model_order],
            bundle_path.name,
        ],
    }
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest

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
    brier_score,
    fit_prequential_bias_calibrator,
    select_stable_weights,
)
from src.config import ExperimentConfig
from src.features import LeakageSafeFeatureEngineer, StrictPastTrackmanFeatures
from src.metrics import evaluate_probabilities
from src.models import (
    MODEL_ORDER,
    train_catboost_fold,
    train_catboost_full,
    train_xgboost_fold,
    train_xgboost_full,
)
from src.preprocessing import TabularPreprocessor
from src.runtime import apply_probability_bias, blend_predictions
from src.splits import make_temporal_folds, uniform_weights


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
        del cat_model, X_train, X_valid, sample_weight
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

    weights, weight_report = select_stable_weights(fold_results, step=0.01)
    calibrator = fit_prequential_bias_calibrator(
        earlier_fold=fold_results[0],
        recent_fold=fold_results[1],
        weights=weights,
    )

    fold_summaries = []
    for fold in fold_results:
        ensemble = blend_predictions(
            [fold["predictions"][name] for name in MODEL_ORDER], weights
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
        [recent["predictions"][name] for name in MODEL_ORDER], weights
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

    ensemble_state = {
        "model_order": list(MODEL_ORDER),
        "weights": weights.tolist(),
        "calibration": calibrator,
        "final_iterations": final_iterations,
    }
    report = {
        "strategy": "temporal_xgb_cat_probability_bias_v2",
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
    del cat_model, X, sample_weight
    gc.collect()

    raw_feature_cols = [col for col in train.columns if col != target]
    bundle = {
        "bundle_version": 2,
        "id_col": config.features.id_col,
        "target_col": target,
        "expected_raw_columns": raw_feature_cols,
        "feature_state": dict(feature_state),
        "preprocessor_state": preprocessor.export_state(),
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
        "strategy": "temporal_xgb_cat_probability_bias_v2",
        "bundle_version": 2,
        "model_order": list(ensemble_state["model_order"]),
        "weights": list(ensemble_state["weights"]),
        "calibration": dict(ensemble_state["calibration"]),
        "final_iterations": dict(ensemble_state["final_iterations"]),
        "n_features": int(len(preprocessor.feature_names_)),
        "model_files": [xgb_path.name, cat_path.name, bundle_path.name],
    }
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest

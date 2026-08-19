from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.calibration import (
    brier_score,
    fit_prequential_logit_calibrator,
    select_stable_weights,
)
from src.config import ExperimentConfig
from src.metrics import evaluate_probabilities
from src.models import train_xgboost_fold, train_xgboost_full
from src.preprocessing import TabularPreprocessor
from src.runtime import apply_logit_intercept, blend_predictions
from src.splits import make_abs_late_fold, make_temporal_folds, uniform_weights
from src.tabfm.config import TabDPTExperimentConfig
from src.tabfm.context import (
    select_recent_context_indices,
    tabdpt_feature_names,
    to_tabdpt_array,
)
from src.tabfm.tabdpt_backend import predict_tabdpt, validate_tabdpt_weight


TABDPT_MODEL_ORDER = ("xgb", "tabdpt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_metrics(label: str, metrics: Mapping[str, float]) -> None:
    print(
        f"[{label}] Brier={metrics['brier']:.8f} "
        f"AUC={metrics.get('auc', float('nan')):.6f} "
        f"pred_mean={metrics['pred_mean']:.6f} "
        f"bias={metrics['calibration_bias']:+.6f}"
    )


def _global_to_fold_positions(
    sorted_fold_indices: np.ndarray,
    selected_global_indices: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(sorted_fold_indices, selected_global_indices)
    if (
        (positions < 0).any()
        or (positions >= len(sorted_fold_indices)).any()
        or not np.array_equal(sorted_fold_indices[positions], selected_global_indices)
    ):
        raise RuntimeError("TabDPT context is not contained in the fold training rows.")
    return positions


def run_tabdpt_temporal_validation(
    train: pd.DataFrame,
    features: pd.DataFrame,
    *,
    experiment_config: ExperimentConfig,
    tabdpt_config: TabDPTExperimentConfig,
    weight_path: Path | str,
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compare XGBoost with a recent-context TabDPT candidate on forward folds."""
    tabdpt_config.validate()
    checkpoint = validate_tabdpt_weight(weight_path)
    target_col = experiment_config.features.target_col
    folds = list(make_temporal_folds(train, experiment_config.temporal_folds))
    folds.append(
        make_abs_late_fold(
            train,
            train_month_max=experiment_config.abs_late_train_month_max,
            valid_months=experiment_config.abs_late_valid_months,
        )
    )
    fold_results: list[dict[str, object]] = []

    for fold_number, fold in enumerate(folds):
        print("\n" + "=" * 88)
        print(
            f"[TABDPT-FOLD] {fold.name}: n_train={len(fold.train_idx):,}, "
            f"n_valid={len(fold.valid_idx):,}"
        )
        print("=" * 88)
        preprocessor = TabularPreprocessor(
            categorical_cols=experiment_config.features.categorical_cols,
            excluded_cols=experiment_config.features.excluded_cols,
        )
        X_train = preprocessor.fit_transform(features.iloc[fold.train_idx])
        X_valid = preprocessor.transform(features.iloc[fold.valid_idx])
        y_train = train.iloc[fold.train_idx][target_col].to_numpy(
            dtype=np.float32, copy=True
        )
        y_valid = train.iloc[fold.valid_idx][target_col].to_numpy(
            dtype=np.float32, copy=True
        )

        start = time.perf_counter()
        xgb_model, xgb_prediction, best_iteration = train_xgboost_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            uniform_weights(len(y_train)),
            experiment_config.models,
        )
        xgb_seconds = time.perf_counter() - start
        del xgb_model
        gc.collect()

        context_global = select_recent_context_indices(
            train,
            fold.train_idx,
            target_col=target_col,
            context_size=tabdpt_config.context_size,
            seed=tabdpt_config.random_seed + 100 * fold_number,
        )
        context_positions = _global_to_fold_positions(fold.train_idx, context_global)
        feature_names = tabdpt_feature_names(
            preprocessor.num_cols_, max_features=tabdpt_config.max_features
        )
        context_x = to_tabdpt_array(X_train.iloc[context_positions], feature_names)
        context_y = train.iloc[context_global][target_col].to_numpy(
            dtype=np.int64, copy=True
        )
        query_x = to_tabdpt_array(X_valid, feature_names)
        start = time.perf_counter()
        tabdpt_prediction = predict_tabdpt(
            context_x,
            context_y,
            query_x,
            weight_path=checkpoint,
            config=tabdpt_config,
            device=device,
        )
        tabdpt_seconds = time.perf_counter() - start

        predictions = {
            "xgb": xgb_prediction,
            "tabdpt": tabdpt_prediction,
        }
        metrics = {
            name: evaluate_probabilities(y_valid, prediction)
            for name, prediction in predictions.items()
        }
        metrics["xgb"]["train_seconds"] = float(xgb_seconds)
        metrics["tabdpt"]["predict_seconds"] = float(tabdpt_seconds)
        _print_metrics(f"{fold.validation_label}/xgb", metrics["xgb"])
        _print_metrics(f"{fold.validation_label}/tabdpt", metrics["tabdpt"])
        print(
            f"[TABDPT-CONTEXT] season="
            f"{int(train.iloc[context_global]['season'].max())} "
            f"rows={len(context_global):,} features={len(feature_names)} "
            f"target_mean={context_y.mean():.6f}"
        )

        fold_results.append(
            {
                "name": fold.name,
                "valid_season": int(fold.valid_season),
                "validation_label": str(fold.validation_label),
                "n_train": int(len(fold.train_idx)),
                "n_valid": int(len(fold.valid_idx)),
                "context_rows": int(len(context_global)),
                "context_season": int(train.iloc[context_global]["season"].max()),
                "feature_names": feature_names,
                "y_true": y_valid,
                "predictions": predictions,
                "metrics": metrics,
                "best_iterations": {"xgb": int(best_iteration)},
            }
        )
        del X_train, X_valid, context_x, context_y, query_x, y_train
        gc.collect()

    weights, weight_report = select_stable_weights(
        fold_results,
        fold_importance=tabdpt_config.fold_importance,
        step=tabdpt_config.ensemble_grid_step,
        model_order=TABDPT_MODEL_ORDER,
        protected_fold_labels=tabdpt_config.protected_folds,
        non_degradation_tolerance=tabdpt_config.non_degradation_tolerance,
    )
    calibrator = fit_prequential_logit_calibrator(
        earlier_fold=fold_results[0],
        recent_fold=fold_results[1],
        weights=weights,
        model_order=TABDPT_MODEL_ORDER,
    )

    fold_summaries: list[dict[str, object]] = []
    blend_scores: list[float] = []
    packaged_scores: list[float] = []
    xgb_scores: list[float] = []
    for fold in fold_results:
        raw_blend = blend_predictions(
            [fold["predictions"][name] for name in TABDPT_MODEL_ORDER], weights
        )
        calibrated_blend = (
            apply_logit_intercept(raw_blend, calibrator["intercept"])
            if calibrator["accepted"]
            else raw_blend
        )
        raw_metrics = evaluate_probabilities(fold["y_true"], raw_blend)
        calibrated_metrics = evaluate_probabilities(fold["y_true"], calibrated_blend)
        _print_metrics(f"{fold['validation_label']}/raw_blend", raw_metrics)
        if calibrator["accepted"]:
            _print_metrics(
                f"{fold['validation_label']}/calibrated_blend", calibrated_metrics
            )
        blend_scores.append(float(raw_metrics["brier"]))
        packaged_scores.append(float(calibrated_metrics["brier"]))
        xgb_scores.append(float(fold["metrics"]["xgb"]["brier"]))
        fold_summaries.append(
            {
                key: fold[key]
                for key in (
                    "name",
                    "valid_season",
                    "validation_label",
                    "n_train",
                    "n_valid",
                    "context_rows",
                    "context_season",
                    "best_iterations",
                    "metrics",
                )
            }
            | {
                "raw_blend": raw_metrics,
                "calibrated_blend": calibrated_metrics,
            }
        )

    importance = np.asarray(tabdpt_config.fold_importance, dtype=np.float64)
    importance /= importance.sum()
    weighted_xgb = float(np.dot(importance, np.asarray(xgb_scores)))
    weighted_blend = float(np.dot(importance, np.asarray(blend_scores)))
    weighted_gain = weighted_xgb - weighted_blend
    labels = [str(fold["validation_label"]) for fold in fold_results]
    protected_checks = {
        label: bool(blend_scores[labels.index(label)] <= xgb_scores[labels.index(label)] + 1e-15)
        for label in tabdpt_config.protected_folds
    }
    late_index = labels.index("2024_late_abs")
    calibration_late_safe = bool(
        not calibrator["accepted"]
        or packaged_scores[late_index] <= blend_scores[late_index] + 1e-15
    )
    tabdpt_weight = float(weights[TABDPT_MODEL_ORDER.index("tabdpt")])
    full_2024_tabdpt_seconds = float(
        fold_results[1]["metrics"]["tabdpt"]["predict_seconds"]
    )
    estimated_l4_seconds = (
        full_2024_tabdpt_seconds * tabdpt_config.l4_runtime_multiplier
        + tabdpt_config.non_tabdpt_runtime_reserve_seconds
    )
    checks = {
        "tabdpt_has_material_weight": (
            tabdpt_weight + 1e-15 >= tabdpt_config.minimum_tabdpt_weight
        ),
        "weighted_gain": weighted_gain + 1e-15 >= tabdpt_config.minimum_weighted_gain,
        "2023_guard": blend_scores[0] <= tabdpt_config.maximum_2023_brier,
        "estimated_l4_runtime": (
            estimated_l4_seconds
            <= tabdpt_config.maximum_estimated_runtime_seconds
        ),
        "calibration_late_transfer": calibration_late_safe,
        **{f"non_degradation_{key}": value for key, value in protected_checks.items()},
    }
    gate = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "weights": weights.tolist(),
            "weighted_xgb_brier": weighted_xgb,
            "weighted_blend_brier": weighted_blend,
            "weighted_gain": weighted_gain,
            "full_2024_rtx3090_tabdpt_seconds": full_2024_tabdpt_seconds,
            "estimated_l4_total_seconds": estimated_l4_seconds,
            "fold_xgb_brier": xgb_scores,
            "fold_blend_brier": blend_scores,
            "fold_packaged_brier": packaged_scores,
        },
        "thresholds": {
            "minimum_tabdpt_weight": tabdpt_config.minimum_tabdpt_weight,
            "minimum_weighted_gain": tabdpt_config.minimum_weighted_gain,
            "maximum_2023_brier": tabdpt_config.maximum_2023_brier,
            "l4_runtime_multiplier": tabdpt_config.l4_runtime_multiplier,
            "non_tabdpt_runtime_reserve_seconds": (
                tabdpt_config.non_tabdpt_runtime_reserve_seconds
            ),
            "maximum_estimated_runtime_seconds": (
                tabdpt_config.maximum_estimated_runtime_seconds
            ),
        },
    }
    recent_iterations = int(fold_results[-1]["best_iterations"]["xgb"])
    final_iterations = int(
        min(
            experiment_config.models.xgb_num_boost_round,
            max(50, round(recent_iterations * 1.05)),
        )
    )
    state = {
        "model_order": list(TABDPT_MODEL_ORDER),
        "weights": weights.tolist(),
        "calibration": calibrator,
        "xgb_final_iterations": final_iterations,
    }
    report = {
        "strategy": "recent_context_tabdpt_turbo_xgb_v1",
        "tabdpt_config": tabdpt_config.__dict__,
        "folds": fold_summaries,
        "weight_selection": weight_report,
        "calibration": calibrator,
        "submission_gate": gate,
    }
    return state, report


def train_and_save_tabdpt_candidate(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_state: Mapping[str, object],
    candidate_state: Mapping[str, object],
    *,
    experiment_config: ExperimentConfig,
    tabdpt_config: TabDPTExperimentConfig,
    source_weight_path: Path | str,
    model_dir: Path,
) -> dict[str, object]:
    """Train the final XGBoost and bundle a fixed 2024 TabDPT context."""
    model_dir.mkdir(parents=True, exist_ok=True)
    target_col = experiment_config.features.target_col
    y = train[target_col].to_numpy(dtype=np.float32, copy=True)
    preprocessor = TabularPreprocessor(
        categorical_cols=experiment_config.features.categorical_cols,
        excluded_cols=experiment_config.features.excluded_cols,
    )
    X = preprocessor.fit_transform(features)

    rounds = int(candidate_state["xgb_final_iterations"])
    print(f"[FINAL] Training TabDPT-candidate XGBoost for {rounds} rounds...")
    model = train_xgboost_full(
        X,
        y,
        uniform_weights(len(y)),
        experiment_config.models,
        num_boost_round=rounds,
    )
    xgb_path = model_dir / "xgb_model.json"
    model.save_model(str(xgb_path))
    del model
    gc.collect()

    feature_names = tabdpt_feature_names(
        preprocessor.num_cols_, max_features=tabdpt_config.max_features
    )
    context_indices = select_recent_context_indices(
        train,
        np.arange(len(train), dtype=np.int64),
        target_col=target_col,
        context_size=tabdpt_config.context_size,
        seed=tabdpt_config.random_seed,
    )
    context_x = to_tabdpt_array(X.iloc[context_indices], feature_names)
    context_y = train.iloc[context_indices][target_col].to_numpy(
        dtype=np.int8, copy=True
    )
    context_path = model_dir / "tabdpt_context.npz"
    np.savez_compressed(
        context_path,
        X=context_x,
        y=context_y,
        feature_names=np.asarray(feature_names),
    )

    source_checkpoint = validate_tabdpt_weight(source_weight_path)
    checkpoint_path = model_dir / "tabdpt1_2.safetensors"
    shutil.copy2(source_checkpoint, checkpoint_path)

    raw_feature_cols = [col for col in train.columns if col != target_col]
    bundle = {
        "bundle_version": 1,
        "strategy": "recent_context_tabdpt_turbo_xgb_v1",
        "id_col": experiment_config.features.id_col,
        "target_col": target_col,
        "expected_raw_columns": raw_feature_cols,
        "feature_state": dict(feature_state),
        "preprocessor_state": preprocessor.export_state(),
        "tabdpt_feature_names": feature_names,
        "tabdpt_config": tabdpt_config.__dict__,
        "ensemble": dict(candidate_state),
        "training": {
            "n_rows": int(len(train)),
            "train_seasons": [int(train["season"].min()), int(train["season"].max())],
            "target_mean": float(y.mean()),
            "context_rows": int(len(context_indices)),
            "context_season": int(train.iloc[context_indices]["season"].max()),
            "context_target_mean": float(context_y.mean()),
        },
    }
    bundle_path = model_dir / "bundle.pkl"
    joblib.dump(bundle, bundle_path, compress=3)

    files = [xgb_path, context_path, checkpoint_path, bundle_path]
    manifest = {
        "strategy": bundle["strategy"],
        "bundle_version": bundle["bundle_version"],
        "model_order": list(candidate_state["model_order"]),
        "weights": list(candidate_state["weights"]),
        "calibration": dict(candidate_state["calibration"]),
        "xgb_final_iterations": rounds,
        "tabdpt_context_rows": int(len(context_indices)),
        "tabdpt_features": int(len(feature_names)),
        "files": {
            path.name: {
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in files
        },
    }
    with (model_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    del X, context_x, context_y
    gc.collect()
    return manifest

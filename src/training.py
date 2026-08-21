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
    fit_abs_regime_logit_calibrator,
    select_stable_weights,
)
from src.config import ExperimentConfig
from src.features import (
    LeakageSafeFeatureEngineer,
    StrictPastMainHistoryFeatures,
    StrictPastTrackmanFeatures,
)
from src.metrics import evaluate_probabilities
from src.models import (
    GBDT_MODEL_ORDER,
    train_catboost_fold,
    train_catboost_full,
    train_lightgbm_fold,
    train_lightgbm_full,
    train_xgboost_fold,
    train_xgboost_full,
)
from src.preprocessing import TabularPreprocessor
from src.runtime import apply_logit_intercept, blend_predictions
from src.splits import make_abs_late_fold, make_temporal_folds, uniform_weights


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


def _strategy_name(
    model_order: tuple[str, ...]
) -> str:
    if model_order == GBDT_MODEL_ORDER:
        return (
            "temporal_xgb_lgb_cat_"
            "abs_calibration_v9"
        )

    if model_order == (
        *GBDT_MODEL_ORDER,
        "resnet",
    ):
        return (
            "temporal_xgb_lgb_cat_"
            "resnet_abs_calibration_v9"
        )

    if model_order == (
        *GBDT_MODEL_ORDER,
        "resnet",
        "ft_transformer",
    ):
        return (
            "temporal_xgb_lgb_cat_"
            "resnet_ftt_abs_calibration_v9"
        )

    raise ValueError(
        f"Unsupported model order: {model_order}"
    )


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
        trackman = StrictPastTrackmanFeatures(config.features).fit_from_csv(
            config.paths.trackman_path,
            main_train=train,
        )

    main_history = None
    if config.use_main_history:
        main_history = StrictPastMainHistoryFeatures(config.features).fit(
            train,
            target_col=config.features.target_col,
        )
    engineer = LeakageSafeFeatureEngineer(
        config.features,
        trackman_features=trackman,
        main_history_features=main_history,
    )
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
    entity_gate_required = bool(
        config.use_trackman and config.features.trackman_entity_enabled
    )
    fold_results = []
    neural_model_order = _active_neural_models(config)
    model_order = (*GBDT_MODEL_ORDER, *neural_model_order)
    print(f"[ENSEMBLE] active_models={list(model_order)}")

    folds = list(make_temporal_folds(train, config.temporal_folds))
    folds.append(
        make_abs_late_fold(
            train,
            train_month_max=config.abs_late_train_month_max,
            valid_months=config.abs_late_valid_months,
        )
    )
    for fold in folds:
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
        entity_ablation: Dict[str, object] | None = None

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
        _print_metrics(f"{fold.validation_label}/xgb", per_model_metrics["xgb"])

        entity_columns = [
            column for column in X_train.columns if column.startswith("tm_entity_")
        ]
        if entity_gate_required:
            if not entity_columns:
                raise RuntimeError(
                    "Trackman entity resolution is enabled but no tm_entity_* "
                    "features reached the preprocessor."
                )
            baseline_columns = [
                column for column in X_train.columns if column not in entity_columns
            ]
            start_ablation = time.perf_counter()
            baseline_model, baseline_prediction, baseline_iterations = (
                train_xgboost_fold(
                    X_train[baseline_columns],
                    y_train,
                    X_valid[baseline_columns],
                    y_valid,
                    sample_weight,
                    config.models,
                )
            )
            baseline_metrics = evaluate_probabilities(y_valid, baseline_prediction)
            baseline_metrics["train_seconds"] = float(
                time.perf_counter() - start_ablation
            )
            entity_gain = float(
                baseline_metrics["brier"] - per_model_metrics["xgb"]["brier"]
            )
            entity_ablation = {
                "baseline_without_entity": baseline_metrics,
                "with_entity": dict(per_model_metrics["xgb"]),
                "brier_gain": entity_gain,
                "n_entity_features": int(len(entity_columns)),
                "baseline_best_iterations": int(baseline_iterations),
            }
            print(
                f"[{fold.validation_label}/entity-ablation] "
                f"baseline={baseline_metrics['brier']:.8f} "
                f"with_entity={per_model_metrics['xgb']['brier']:.8f} "
                f"gain={entity_gain:+.8f} features={len(entity_columns)}"
            )
            del baseline_model, baseline_prediction
        del xgb_model
        gc.collect()

        # --------------------------------------------------------
        # LightGBM
        # --------------------------------------------------------
        start = time.perf_counter()

        (
            lgb_model,
            predictions["lgb"],
            best_iterations["lgb"],
        ) = train_lightgbm_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            sample_weight,
            preprocessor.categorical_indices,
            config.models,
        )

        elapsed = (
            time.perf_counter() - start
        )

        per_model_metrics["lgb"] = (
            evaluate_probabilities(
                y_valid,
                predictions["lgb"],
            )
        )

        per_model_metrics["lgb"][
            "train_seconds"
        ] = float(elapsed)

        _print_metrics(
            f"{fold.validation_label}/lgb",
            per_model_metrics["lgb"],
        )

        del lgb_model
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
        _print_metrics(f"{fold.validation_label}/cat", per_model_metrics["cat"])
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
                _print_metrics(f"{fold.validation_label}/{kind}", per_model_metrics[kind])
                del neural_model, prediction
                release_torch_memory()

            del train_arrays, valid_arrays, neural_state

        del X_train, X_valid, sample_weight
        gc.collect()

        fold_results.append(
            {
                "name": fold.name,
                "valid_season": fold.valid_season,
                "validation_label": (
                    fold.validation_label
                ),
                "y_true": y_valid,
                "valid_months": (
                    pd.to_numeric(
                        train.iloc[
                            fold.valid_idx
                        ]["game_month"],
                        errors="raise",
                    )
                    .to_numpy(
                        dtype=np.int8,
                        copy=True,
                    )
                ),
                "predictions": predictions,
                "best_iterations": (
                    best_iterations
                ),
                "metrics": per_model_metrics,
                "entity_ablation": (
                    entity_ablation
                ),
                "n_train": int(
                    len(fold.train_idx)
                ),
                "n_valid": int(
                    len(fold.valid_idx)
                ),
            }
        )

    if len(fold_results) != 3:
        raise RuntimeError(
            "This submission strategy requires 2023, 2024 and late-2024 holdouts."
        )

    weights, weight_report = select_stable_weights(
        fold_results,
        fold_importance=config.temporal_fold_importance,
        step=config.ensemble_grid_step,
        model_order=model_order,
        protected_fold_labels=config.ensemble_protected_folds,
        non_degradation_tolerance=config.ensemble_non_degradation_tolerance,
    )
    calibrator = (
        fit_abs_regime_logit_calibrator(
            full_2024_fold=fold_results[1],
            late_2024_fold=fold_results[2],
            weights=weights,
            model_order=model_order,
            early_month_max=(
                config.abs_late_train_month_max
            ),
        )
    )

    fold_summaries = []
    for fold in fold_results:
        ensemble = blend_predictions(
            [fold["predictions"][name] for name in model_order], weights
        )
        ensemble_metrics = evaluate_probabilities(fold["y_true"], ensemble)
        _print_metrics(f"{fold['validation_label']}/ensemble", ensemble_metrics)
        fold_summaries.append(
            {
                "name": fold["name"],
                "valid_season": int(fold["valid_season"]),
                "validation_label": str(fold["validation_label"]),
                "n_train": fold["n_train"],
                "n_valid": fold["n_valid"],
                "best_iterations": fold["best_iterations"],
                "models": fold["metrics"],
                "entity_ablation": fold["entity_ablation"],
                "ensemble": ensemble_metrics,
            }
        )

    late_fold = fold_results[2]

    late_ensemble = blend_predictions(
        [
            late_fold["predictions"][name]
            for name in model_order
        ],
        weights,
    )

    late_transferred = (
        apply_logit_intercept(
            late_ensemble,
            calibrator["early_intercept"],
        )
    )

    transfer_metrics = (
        evaluate_probabilities(
            late_fold["y_true"],
            late_transferred,
        )
    )

    print(
        "[CALIBRATION] "
        f"accepted={calibrator['accepted']} "
        f"late_raw="
        f"{calibrator['late_raw_brier']:.8f} "
        f"early->late="
        f"{calibrator['late_transferred_brier']:.8f} "
        f"gain="
        f"{calibrator['transfer_gain']:+.8f} "
        f"final_intercept="
        f"{calibrator['intercept']:+.6f}"
    )

    recent_iterations = (
        fold_results[-1]["best_iterations"]
    )

    final_iterations = {
        "xgb": int(
            min(
                config.models.xgb_num_boost_round,
                max(
                    50,
                    round(
                        recent_iterations["xgb"]
                        * 1.05
                    ),
                ),
            )
        ),
        "lgb": int(
            min(
                config.models.lgb_num_boost_round,
                max(
                    50,
                    round(
                        recent_iterations["lgb"]
                        * 1.05
                    ),
                ),
            )
        ),
        "cat": int(
            min(
                config.models.cat_iterations,
                max(
                    50,
                    round(
                        recent_iterations["cat"]
                        * 1.05
                    ),
                ),
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
    by_label = {item["validation_label"]: item for item in fold_summaries}
    guard_2023 = float(by_label["2023"]["ensemble"]["brier"])
    anchor_2024_raw = float(by_label["2024"]["ensemble"]["brier"])
    guard_2023 = float(
        by_label["2023"][
            "ensemble"
        ]["brier"]
    )

    anchor_2024 = float(
        by_label["2024"][
            "ensemble"
        ]["brier"]
    )

    late = (
        by_label[
            "2024_late_abs"
        ]
    )

    late_ensemble = float(
        late["ensemble"]["brier"]
    )

    late_transferred_brier = float(
        transfer_metrics["brier"]
    )

    late_best_component = min(
        float(metrics["brier"])
        for metrics
        in late["models"].values()
    )

    late_blend_gain = (
        late_best_component
        - late_ensemble
    )
    # Use the calibrated score only when the intercept was estimated on 2023
    # and improved 2024 without seeing 2024 labels. This is a genuine
    # one-season-forward result, not in-fold calibration.
    anchor_2024 = float(
        by_label["2024"]["ensemble"]["brier"]
    )
    late = by_label["2024_late_abs"]
    late_ensemble = float(late["ensemble"]["brier"])
    late_best_component = min(
        float(metrics["brier"]) for metrics in late["models"].values()
    )
    late_blend_gain = late_best_component - late_ensemble
    entity_2024_gain = (
        float(by_label["2024"]["entity_ablation"]["brier_gain"])
        if entity_gate_required
        else None
    )
    entity_late_gain = (
        float(late["entity_ablation"]["brier_gain"])
        if entity_gate_required
        else None
    )
    checks = {
        "2023_guard": (
            guard_2023
            <= config.submission_gate_2023_max_brier
        ),
        "2024_target": (
            anchor_2024
            <= config.submission_gate_2024_max_brier
        ),
        "late_abs_blend": (
            late_blend_gain
            >= config.submission_gate_late_min_blend_gain
        ),
        "calibration_late_transfer": (
            not calibrator["accepted"]
            or calibrator["transfer_gain"] > 0.0
        ),
        "entity_2024_paired_ablation": (
            not entity_gate_required
            or entity_2024_gain
            >= config.submission_gate_entity_2024_min_gain
        ),
        "entity_late_paired_ablation": (
            not entity_gate_required
            or entity_late_gain
            >= config.submission_gate_entity_late_min_gain
        ),
    }
    report["submission_gate"] = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "2023_brier": (
                guard_2023
            ),
            "2024_raw_brier": (
                anchor_2024
            ),
            "2024_gate_brier": (
                anchor_2024
            ),
            "2024_gate_uses_calibration": False,

            "2024_late_raw_brier": (
                late_ensemble
            ),
            "2024_late_transferred_brier": (
                late_transferred_brier
            ),
            "2024_late_best_component_brier": (
                late_best_component
            ),
            "2024_late_blend_gain": (
                late_blend_gain
            ),

            "calibration_accepted": bool(
                calibrator["accepted"]
            ),
            "calibration_transfer_gain": float(
                calibrator[
                    "transfer_gain"
                ]
            ),

            "entity_2024_xgb_brier_gain": (
                entity_2024_gain
            ),
            "entity_late_xgb_brier_gain": (
                entity_late_gain
            ),
        },
        "thresholds": {
            "2023_max_brier": config.submission_gate_2023_max_brier,
            "2024_max_brier": config.submission_gate_2024_max_brier,
            "2024_late_min_blend_gain": config.submission_gate_late_min_blend_gain,
            "entity_2024_min_gain": config.submission_gate_entity_2024_min_gain,
            "entity_late_min_gain": config.submission_gate_entity_late_min_gain,
        },
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

    lgb_rounds = int(
        ensemble_state[
            "final_iterations"
        ]["lgb"]
    )

    print(
        "[FINAL] Training LightGBM "
        f"for {lgb_rounds} rounds..."
    )

    lgb_model = train_lightgbm_full(
        X,
        y,
        sample_weight,
        preprocessor.categorical_indices,
        config.models,
        num_boost_round=lgb_rounds,
    )

    lgb_path = (
        model_dir / "lgb_model.txt"
    )

    lgb_model.save_model(
        str(lgb_path)
    )

    del lgb_model
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
        "bundle_version": 7,
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
        "bundle_version": 7,
        "model_order": list(ensemble_state["model_order"]),
        "weights": list(ensemble_state["weights"]),
        "calibration": dict(ensemble_state["calibration"]),
        "final_iterations": dict(ensemble_state["final_iterations"]),
        "n_features": int(len(preprocessor.feature_names_)),
        "model_files": [
            xgb_path.name,
            lgb_path.name,
            cat_path.name,
            *[
                neural_paths[name].name
                for name in neural_model_order
            ],
            bundle_path.name,
        ],
    }
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest

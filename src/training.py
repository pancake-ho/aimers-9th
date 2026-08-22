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
    evaluate_fixed_calibrated_weights,
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
    train_xgboost_bagged_fold,
    train_xgboost_full_bagged,
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
    model_order: tuple[str, ...],
) -> str:
    if model_order == GBDT_MODEL_ORDER:
        return (
            "temporal_xgbbag3_lgb_cat_"
            "fixedchamp_calibrated_v11b"
        )

    if model_order == (
        *GBDT_MODEL_ORDER,
        "resnet",
    ):
        return (
            "temporal_xgbbag3_lgb_cat_"
            "resnet_fixedchamp_"
            "calibrated_v11b"
        )

    if model_order == (
        *GBDT_MODEL_ORDER,
        "resnet",
        "ft_transformer",
    ):
        return (
            "temporal_xgbbag3_lgb_cat_"
            "resnet_ftt_fixedchamp_"
            "calibrated_v11b"
        )

    raise ValueError(
        "Unsupported model order: "
        f"{model_order}"
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

        # ----------------------------------------------------
        # XGBoost 3-seed bagging.
        # ----------------------------------------------------

        start = time.perf_counter()

        (
            xgb_models,
            predictions["xgb"],
            best_iterations["xgb"],
            xgb_single_prediction,
        ) = (
            train_xgboost_bagged_fold(
                X_train,
                y_train,
                X_valid,
                y_valid,
                sample_weight,
                config.models,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        per_model_metrics[
            "xgb"
        ] = evaluate_probabilities(
            y_valid,
            predictions["xgb"],
        )

        per_model_metrics[
            "xgb"
        ][
            "train_seconds"
        ] = float(elapsed)

        xgb_single_metrics = (
            evaluate_probabilities(
                y_valid,
                xgb_single_prediction,
            )
        )

        xgb_bagging_gain = float(
            xgb_single_metrics["brier"]
            - per_model_metrics[
                "xgb"
            ]["brier"]
        )

        xgb_bagging_ablation = {
            "single_seed": (
                int(
                    config.models
                    .xgb_bagging_seeds[0]
                )
            ),
            "seeds": [
                int(seed)
                for seed
                in config.models
                .xgb_bagging_seeds
            ],
            "single_seed_metrics": (
                xgb_single_metrics
            ),
            "bagged_metrics": dict(
                per_model_metrics[
                    "xgb"
                ]
            ),
            "brier_gain": (
                xgb_bagging_gain
            ),
        }

        _print_metrics(
            f"{fold.validation_label}/xgb",
            per_model_metrics["xgb"],
        )

        print(
            "[XGB-BAG-ABLATION] "
            f"fold="
            f"{fold.validation_label} "
            f"single="
            f"{xgb_single_metrics['brier']:.8f} "
            f"bagged="
            f"{per_model_metrics['xgb']['brier']:.8f} "
            f"gain="
            f"{xgb_bagging_gain:+.8f}"
        )

        del (
            xgb_models,
            xgb_single_prediction,
        )

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
                "xgb_bagging_ablation": (
                    xgb_bagging_ablation
                ),
            }
        )

    if len(fold_results) != 3:
        raise RuntimeError(
            "This submission strategy requires 2023, 2024 and late-2024 holdouts."
        )

    (
        weights,
        weight_report,
        calibrator,
    ) = (
        evaluate_fixed_calibrated_weights(
            fold_results,
            weights=(
                config
                .ensemble_fixed_champion_weights
            ),
            fold_importance=(
                config
                .temporal_fold_importance
            ),
            model_order=model_order,
            early_month_max=(
                config
                .abs_late_train_month_max
            ),
        )
    )

    print(
        "[ENSEMBLE-POLICY] "
        f"policy={weight_report['method']} "
        f"weights={weights.tolist()} "
        f"forward_brier="
        f"{weight_report['forward_weighted_brier']:.8f} "
        f"calibration_accepted="
        f"{weight_report['calibration_accepted']} "
        f"calibration_transfer_gain="
        f"{weight_report['calibration_transfer_gain']:+.8f}"
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
                "xgb_bagging_ablation": (
                    fold[
                        "xgb_bagging_ablation"
                    ]
                ),                
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
    by_label = {
        item["validation_label"]: item
        for item in fold_summaries
    }


    # --------------------------------------------------------
    # V11 deployment-aware submission gate
    #
    # Protect:
    # 1) the current bagged XGB reference,
    # 2) the previous V10 champion,
    # 3) forward ABS calibration transfer.
    # --------------------------------------------------------

    guard_2023 = float(
        by_label[
            "2023"
        ]["ensemble"]["brier"]
    )

    anchor_2024 = float(
        by_label[
            "2024"
        ]["ensemble"]["brier"]
    )

    late_summary = (
        by_label[
            "2024_late_abs"
        ]
    )

    late_raw_brier = float(
        late_summary[
            "ensemble"
        ]["brier"]
    )

    late_transferred_brier = float(
        transfer_metrics[
            "brier"
        ]
    )

    reference_model = (
        config
        .ensemble_shrinkage_reference_model
    )

    reference_2024_brier = float(
        by_label[
            "2024"
        ]["models"][
            reference_model
        ]["brier"]
    )

    reference_late_brier = float(
        by_label[
            "2024_late_abs"
        ]["models"][
            reference_model
        ]["brier"]
    )

    gain_2024_vs_reference = float(
        reference_2024_brier
        - anchor_2024
    )

    gain_late_raw_vs_reference = float(
        reference_late_brier
        - late_raw_brier
    )

    gain_late_calibrated_vs_reference = float(
        reference_late_brier
        - late_transferred_brier
    )

    entity_2024_gain = (
        float(
            by_label["2024"][
                "entity_ablation"
            ]["brier_gain"]
        )
        if entity_gate_required
        else None
    )

    entity_late_gain = (
        float(
            late_summary[
                "entity_ablation"
            ]["brier_gain"]
        )
        if entity_gate_required
        else None
    )

    xgb_bagging_2024_gain = float(
        by_label[
            "2024"
        ][
            "xgb_bagging_ablation"
        ][
            "brier_gain"
        ]
    )

    xgb_bagging_late_gain = float(
        by_label[
            "2024_late_abs"
        ][
            "xgb_bagging_ablation"
        ][
            "brier_gain"
        ]
    )

    forward_brier = float(
        weight_report[
            "forward_weighted_brier"
        ]
    )

    previous_forward_brier = float(
        config
        .submission_gate_previous_forward_brier
    )

    forward_improvement = float(
        previous_forward_brier
        - forward_brier
    )

    previous_2024_raw_brier = float(
        config
        .submission_gate_previous_2024_raw_brier
    )

    previous_late_calibrated_brier = float(
        config
        .submission_gate_previous_late_calibrated_brier
    )

    champion_2024_gain = float(
        previous_2024_raw_brier
        - anchor_2024
    )

    champion_late_calibrated_gain = float(
        previous_late_calibrated_brier
        - late_transferred_brier
    )

    checks = {
        "fixed_weight_policy": bool(
            weight_report[
                "selection_passed"
            ]
        ),

        "2023_guard": (
            guard_2023
            <= config
            .submission_gate_2023_max_brier
        ),

        "2024_non_degradation": (
            gain_2024_vs_reference
            + 1.0e-15
            >= config
            .submission_gate_2024_min_gain_vs_reference
        ),

        "late_raw_non_degradation": (
            gain_late_raw_vs_reference
            + 1.0e-15
            >= config
            .submission_gate_late_raw_min_gain_vs_reference
        ),

        "calibration_accepted": bool(
            calibrator["accepted"]
        ),

        "calibration_transfer_gain": (
            float(
                calibrator[
                    "transfer_gain"
                ]
            )
            + 1.0e-15
            >= config
            .submission_gate_calibration_min_transfer_gain
        ),

        "late_calibrated_gain_vs_reference": (
            gain_late_calibrated_vs_reference
            + 1.0e-15
            >= config
            .submission_gate_late_calibrated_min_gain_vs_reference
        ),

        "entity_2024_paired_ablation": (
            not entity_gate_required
            or entity_2024_gain
            >= config
            .submission_gate_entity_2024_min_gain
        ),

        "entity_late_paired_ablation": (
            not entity_gate_required
            or entity_late_gain
            >= config
            .submission_gate_entity_late_min_gain
        ),
        "xgb_bagging_2024": (
            xgb_bagging_2024_gain
            + 1.0e-15
            >= config
            .submission_gate_xgb_bagging_2024_min_gain
        ),

        "xgb_bagging_late": (
            xgb_bagging_late_gain
            + 1.0e-15
            >= config
            .submission_gate_xgb_bagging_late_min_gain
        ),

        "beats_previous_forward_objective": (
            forward_improvement
            + 1.0e-15
            >= config
            .submission_gate_min_forward_improvement
        ),
        "beats_previous_2024_raw": (
            champion_2024_gain
            + 1.0e-15
            >= config
            .submission_gate_previous_2024_min_gain
        ),

        "beats_previous_late_calibrated": (
            champion_late_calibrated_gain
            + 1.0e-15
            >= config
            .submission_gate_previous_late_calibrated_min_gain
        ),              
    }

    report[
        "artifact_recommendation"
    ] = {
        "quality_gate_passed": bool(
            report[
                "submission_gate"
            ]["passed"]
        ),
        "quality_gate_is_advisory": True,
        "package_artifact": True,
    }

    report[
        "submission_gate"
    ] = {
        "passed": bool(
            all(
                checks.values()
            )
        ),

        "checks": checks,

        "observed": {
            "weight_policy": (
                weight_report[
                    "method"
                ]
            ),

            "weights": list(
                weights
            ),

            "reference_model": (
                reference_model
            ),

            "2023_brier": (
                guard_2023
            ),

            "2024_raw_brier": (
                anchor_2024
            ),

            "2024_reference_brier": (
                reference_2024_brier
            ),

            "2024_gain_vs_reference": (
                gain_2024_vs_reference
            ),

            "late_raw_brier": (
                late_raw_brier
            ),

            "late_reference_brier": (
                reference_late_brier
            ),

            "late_raw_gain_vs_reference": (
                gain_late_raw_vs_reference
            ),

            "late_calibrated_brier": (
                late_transferred_brier
            ),

            "late_calibrated_gain_vs_reference": (
                gain_late_calibrated_vs_reference
            ),

            "calibration_transfer_gain": float(
                calibrator[
                    "transfer_gain"
                ]
            ),

            "calibration_final_intercept": float(
                calibrator[
                    "intercept"
                ]
            ),

            "entity_2024_xgb_brier_gain": (
                entity_2024_gain
            ),

            "entity_late_xgb_brier_gain": (
                entity_late_gain
            ),
            "xgb_bagging_2024_gain": (
                xgb_bagging_2024_gain
            ),

            "xgb_bagging_late_gain": (
                xgb_bagging_late_gain
            ),

            "forward_weighted_brier": (
                forward_brier
            ),

            "previous_forward_brier": (
                previous_forward_brier
            ),

            "forward_improvement": (
                forward_improvement
            ),     
            "previous_2024_raw_brier": (
                previous_2024_raw_brier
            ),

            "champion_2024_gain": (
                champion_2024_gain
            ),

            "previous_late_calibrated_brier": (
                previous_late_calibrated_brier
            ),

            "champion_late_calibrated_gain": (
                champion_late_calibrated_gain
            ),                   
        },

        "thresholds": {
            "2023_max_brier": (
                config
                .submission_gate_2023_max_brier
            ),

            "2024_min_gain_vs_reference": (
                config
                .submission_gate_2024_min_gain_vs_reference
            ),

            "late_raw_min_gain_vs_reference": (
                config
                .submission_gate_late_raw_min_gain_vs_reference
            ),

            "calibration_min_transfer_gain": (
                config
                .submission_gate_calibration_min_transfer_gain
            ),

            "late_calibrated_min_gain_vs_reference": (
                config
                .submission_gate_late_calibrated_min_gain_vs_reference
            ),

            "entity_2024_min_gain": (
                config
                .submission_gate_entity_2024_min_gain
            ),

            "entity_late_min_gain": (
                config
                .submission_gate_entity_late_min_gain
            ),
            "xgb_bagging_2024_min_gain": (
                config
                .submission_gate_xgb_bagging_2024_min_gain
            ),

            "xgb_bagging_late_min_gain": (
                config
                .submission_gate_xgb_bagging_late_min_gain
            ),

            "minimum_forward_improvement": (
                config
                .submission_gate_min_forward_improvement
            ),            
            "previous_2024_min_gain": (
                config
                .submission_gate_previous_2024_min_gain
            ),

            "previous_late_calibrated_min_gain": (
                config
                .submission_gate_previous_late_calibrated_min_gain
            ),            
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

    xgb_rounds = int(
        ensemble_state[
            "final_iterations"
        ]["xgb"]
    )

    print(
        "[FINAL] Training "
        "3-seed XGBoost bag "
        f"for {xgb_rounds} rounds..."
    )

    xgb_models = (
        train_xgboost_full_bagged(
            X,
            y,
            sample_weight,
            config.models,
            num_boost_round=(
                xgb_rounds
            ),
        )
    )

    xgb_seeds = tuple(
        int(seed)
        for seed
        in config.models
        .xgb_bagging_seeds
    )

    if len(xgb_models) != len(
        xgb_seeds
    ):
        raise RuntimeError(
            "Unexpected number of "
            "XGBoost bagging models."
        )

    xgb_paths = []

    for index, (
        seed,
        model,
    ) in enumerate(
        zip(
            xgb_seeds,
            xgb_models,
        )
    ):
        filename = (
            "xgb_model.json"
            if index == 0
            else (
                f"xgb_model_seed"
                f"{seed}.json"
            )
        )

        path = (
            model_dir
            / filename
        )

        model.save_model(
            str(path)
        )

        xgb_paths.append(
            path
        )

    del xgb_models
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
        "quality_assessment": dict(
            ensemble_state.get(
                "quality_assessment",
                {},
            )
        ),        
        "training": {
            "train_seasons": [int(train["season"].min()), int(train["season"].max())],
            "n_rows": int(len(train)),
            "target_mean": float(y.mean()),
            "random_seed": int(config.models.random_seed),
        },
        "xgb_bagging": {
            "seeds": [
                int(seed)
                for seed
                in xgb_seeds
            ],
            "model_files": [
                path.name
                for path
                in xgb_paths
            ],
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
        "quality_assessment": dict(
            ensemble_state.get(
                "quality_assessment",
                {},
            )
        ),        
        "model_files": [
            *[
                path.name
                for path
                in xgb_paths
            ],
            lgb_path.name,
            cat_path.name,
            *[
                neural_paths[name].name
                for name in neural_model_order
            ],
            bundle_path.name,
        ],
        "xgb_bagging": {
            "seeds": [
                int(seed)
                for seed
                in xgb_seeds
            ],
            "model_files": [
                path.name
                for path
                in xgb_paths
            ],
        },        
    }
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest

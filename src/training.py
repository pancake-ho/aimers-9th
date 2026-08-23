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
    select_calibration_aware_shrunk_weights,
)
from src.config import ExperimentConfig
from src.features import (
    LeakageSafeFeatureEngineer,
    StrictPastMainHistoryFeatures,
    StrictPastTrackmanFeatures,
)
from src.privileged_trackman import (
    build_distillation_target,
    select_lupi_weight,
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
    train_xgboost_multiview_fold,
    train_xgboost_multiview_full,
    train_xgboost_temporal_fold,
    train_xgboost_temporal_full,    
    train_xgboost_native_cat_fold,
    train_xgboost_native_cat_full,    
)
from src.xgb_multiview import (
    blend_xgb_views,
    select_xgb_multiview_weights,
)
from src.xgb_temporal import (
    blend_temporal_views,
    recent_window_positions,
    select_temporal_view_weights,
)
from src.preprocessing import TabularPreprocessor
from src.runtime import (
    apply_logit_intercept,
    blend_predictions,
    preprocess_xgb_native_frame,
)
from src.xgb_native_cat import (
    blend_xgb_native_cat,
    select_xgb_native_cat_weight,
)
from src.splits import make_abs_late_fold, make_temporal_folds, uniform_weights


NEURAL_MODEL_ORDER = ("resnet", "ft_transformer")

WEIGHT_EPS = 1.0e-12


def _outer_weight_map(
    ensemble_state: Mapping[
        str,
        object,
    ],
) -> Dict[str, float]:
    model_order = tuple(
        str(name)
        for name
        in ensemble_state[
            "model_order"
        ]
    )

    weights = np.asarray(
        ensemble_state[
            "weights"
        ],
        dtype=np.float64,
    )

    if weights.shape != (
        len(model_order),
    ):
        raise ValueError(
            "Outer ensemble weight count "
            "does not match model_order."
        )

    if not np.isfinite(
        weights
    ).all():
        raise ValueError(
            "Outer ensemble weights "
            "must be finite."
        )

    if (
        weights < 0.0
    ).any():
        raise ValueError(
            "Outer ensemble weights "
            "must be non-negative."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Outer ensemble weights "
            "must sum to one."
        )

    return {
        name: float(weight)
        for name, weight
        in zip(
            model_order,
            weights,
        )
    }


def _outer_active(
    outer_weights: Mapping[
        str,
        float,
    ],
    name: str,
) -> bool:
    return (
        abs(
            float(
                outer_weights.get(
                    name,
                    0.0,
                )
            )
        )
        > WEIGHT_EPS
    )

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
    model_order: tuple[
        str,
        ...,
    ],
    *,
    ensemble_policy: str,
    lupi_enabled: bool,
    native_cat_enabled: bool,
) -> str:
    parts = [
        "temporal_xgb",
        "entityv2",
        "timeviews",
        "bag3",
        "recent1",
        "recent2",
    ]

    if native_cat_enabled:
        parts.append(
            "nativecat"
        )

    if lupi_enabled:
        parts.append(
            "lupi"
        )

    parts.extend(
        [
            str(
                ensemble_policy
            ),
            "calibrated",
            "v16",
        ]
    )

    if model_order == (
        *GBDT_MODEL_ORDER,
        "resnet",
        "ft_transformer",
    ):
        parts.extend(
            [
                "lgb",
                "cat",
                "resnet",
                "ftt",
            ]
        )

    elif model_order == (
        *GBDT_MODEL_ORDER,
        "resnet",
    ):
        parts.extend(
            [
                "lgb",
                "cat",
                "resnet",
            ]
        )

    elif model_order == (
        GBDT_MODEL_ORDER
    ):
        parts.extend(
            [
                "lgb",
                "cat",
            ]
        )

    else:
        raise ValueError(
            "Unsupported model order: "
            f"{model_order}"
        )

    return "_".join(
        parts
    )


def _print_metrics(label: str, metrics: Mapping[str, float]) -> None:
    print(
        f"[{label}] Brier={metrics['brier']:.8f} "
        f"AUC={metrics.get('auc', float('nan')):.6f} "
        f"target_mean={metrics['target_mean']:.6f} "
        f"pred_mean={metrics['pred_mean']:.6f} "
        f"bias={metrics['calibration_bias']:+.6f}"
    )


def _split_entity_feature_view(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    entity_columns = [
        column
        for column
        in X.columns
        if str(column).startswith(
            "tm_entity_"
        )
    ]

    if not entity_columns:
        raise ValueError(
            "Trackman entity ablation requested "
            "but no tm_entity_* columns exist."
        )

    baseline = X.drop(
        columns=entity_columns
    )

    if baseline.shape[1] == 0:
        raise ValueError(
            "Trackman entity ablation "
            "removed every feature."
        )

    return (
        baseline,
        entity_columns,
    )


def _apply_temporal_xgb_blend(
    fold: Dict[str, object],
    temporal_weights: np.ndarray,
) -> Dict[str, float]:
    """Install the selected temporal-XGB prediction into one fold.

    The fold-local prediction is computed only from that fold's
    validation predictions.  No prediction array may be reused
    across folds.
    """

    y_true = np.asarray(
        fold["y_true"],
        dtype=np.float64,
    )

    temporal_prediction = (
        blend_temporal_views(
            fold[
                "xgb_temporal_views"
            ],
            temporal_weights,
        )
    )

    temporal_prediction = np.asarray(
        temporal_prediction,
        dtype=np.float64,
    )

    if (
        temporal_prediction.shape
        != y_true.shape
    ):
        raise ValueError(
            "Temporal-XGB prediction/target "
            "shape mismatch: "
            f"fold={fold.get('validation_label')} "
            f"prediction={temporal_prediction.shape} "
            f"target={y_true.shape}"
        )

    if not np.isfinite(
        temporal_prediction
    ).all():
        raise ValueError(
            "Temporal-XGB prediction contains "
            "NaN or infinity: "
            f"fold={fold.get('validation_label')}"
        )

    metrics = (
        evaluate_probabilities(
            y_true,
            temporal_prediction,
        )
    )

    fold[
        "predictions"
    ][
        "xgb"
    ] = temporal_prediction

    fold[
        "metrics"
    ][
        "xgb"
    ] = metrics

    return metrics
    

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
    *,
    privileged_state: Mapping[
        str,
        object,
    ]
    | None = None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    target = config.features.target_col
    entity_gate_required = bool(
        config.use_trackman and config.features.trackman_entity_enabled
    )
    lupi_enabled = bool(
        config
        .privileged
        .enabled
    )

    if (
        lupi_enabled
        and privileged_state
        is None
    ):
        raise ValueError(
            "Privileged distillation is enabled "
            "but privileged_state is absent."
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

        lupi_prediction = None
        lupi_fold_state = None

        if lupi_enabled:
            teacher_probability = np.asarray(
                privileged_state[
                    "teacher_probability"
                ],
                dtype=np.float32,
            )

            teacher_available = np.asarray(
                privileged_state[
                    "teacher_available"
                ],
                dtype=bool,
            )

            fold_teacher_probability = (
                teacher_probability[
                    fold.train_idx
                ]
            )

            fold_teacher_available = (
                teacher_available[
                    fold.train_idx
                ]
            )

            y_distill = (
                build_distillation_target(
                    y_train,
                    fold_teacher_probability,
                    fold_teacher_available,
                    strength=(
                        config
                        .privileged
                        .distill_strength
                    ),
                )
            )

            (
                lupi_model,
                lupi_prediction,
                lupi_rounds,
            ) = (
                train_xgboost_temporal_fold(
                    X_train,
                    y_distill,
                    X_valid,
                    y_valid,
                    sample_weight,
                    config.models,
                    seed=(
                        config
                        .privileged
                        .student_seed
                    ),
                )
            )

            best_iterations[
                "xgb_lupi_student"
            ] = int(
                lupi_rounds
            )

            lupi_metrics = (
                evaluate_probabilities(
                    y_valid,
                    lupi_prediction,
                )
            )

            valid_teacher_probability = (
                teacher_probability[
                    fold.valid_idx
                ]
            )

            valid_teacher_available = (
                teacher_available[
                    fold.valid_idx
                ]
            )

            teacher_subset_metrics = None
            base_subset_metrics = None

            if (
                valid_teacher_available.sum()
                > 0
            ):
                teacher_subset_metrics = (
                    evaluate_probabilities(
                        y_valid[
                            valid_teacher_available
                        ],
                        valid_teacher_probability[
                            valid_teacher_available
                        ],
                    )
                )

                base_subset_metrics = (
                    evaluate_probabilities(
                        y_valid[
                            valid_teacher_available
                        ],
                        predictions[
                            "xgb"
                        ][
                            valid_teacher_available
                        ],
                    )
                )

            lupi_fold_state = {
                "student_metrics": (
                    lupi_metrics
                ),
                "distilled_train_rows": int(
                    fold_teacher_available.sum()
                ),
                "distilled_train_fraction": float(
                    fold_teacher_available.mean()
                ),
                "teacher_subset_metrics": (
                    teacher_subset_metrics
                ),
                "base_subset_metrics": (
                    base_subset_metrics
                ),
            }

            print(
                "[LUPI-STUDENT] "
                f"fold={fold.validation_label} "
                f"distilled_rows="
                f"{fold_teacher_available.sum():,} "
                f"rounds={lupi_rounds} "
                f"brier="
                f"{lupi_metrics['brier']:.8f}"
            )

            if (
                teacher_subset_metrics
                is not None
                and base_subset_metrics
                is not None
            ):
                teacher_gain = float(
                    base_subset_metrics[
                        "brier"
                    ]
                    - teacher_subset_metrics[
                        "brier"
                    ]
                )

                print(
                    "[LUPI-TEACHER-UPPER] "
                    f"fold={fold.validation_label} "
                    f"rows="
                    f"{valid_teacher_available.sum():,} "
                    f"base="
                    f"{base_subset_metrics['brier']:.8f} "
                    f"teacher="
                    f"{teacher_subset_metrics['brier']:.8f} "
                    f"gain="
                    f"{teacher_gain:+.8f}"
                )

            del (
                lupi_model,
                y_distill,
            )

            gc.collect()

        # ----------------------------------------------------
        # Paired Trackman-entity ablation.
        #
        # Same fold, same XGBoost configuration, same bagging seeds.
        # The only difference is removal of tm_entity_* columns.
        #
        # Run only on the two protected future regimes to avoid
        # unnecessary extra training cost.
        # ----------------------------------------------------
        if (
            entity_gate_required
            and fold.validation_label
            in {
                "2024",
                "2024_late_abs",
            }
        ):
            (
                X_train_no_entity,
                entity_columns,
            ) = _split_entity_feature_view(
                X_train
            )

            X_valid_no_entity = (
                X_valid.drop(
                    columns=entity_columns
                )
            )

            start = time.perf_counter()

            (
                no_entity_models,
                no_entity_prediction,
                _,
                no_entity_single_prediction,
            ) = train_xgboost_bagged_fold(
                X_train_no_entity,
                y_train,
                X_valid_no_entity,
                y_valid,
                sample_weight,
                config.models,
            )

            no_entity_seconds = (
                time.perf_counter()
                - start
            )

            no_entity_metrics = (
                evaluate_probabilities(
                    y_valid,
                    no_entity_prediction,
                )
            )

            no_entity_metrics[
                "train_seconds"
            ] = float(
                no_entity_seconds
            )

            entity_brier_gain = float(
                no_entity_metrics["brier"]
                - per_model_metrics[
                    "xgb"
                ][
                    "brier"
                ]
            )

            entity_ablation = {
                "entity_feature_count": int(
                    len(
                        entity_columns
                    )
                ),
                "without_entity_feature_count": int(
                    X_train_no_entity.shape[1]
                ),
                "with_entity_feature_count": int(
                    X_train.shape[1]
                ),
                "without_entity_metrics": (
                    no_entity_metrics
                ),
                "with_entity_metrics": dict(
                    per_model_metrics[
                        "xgb"
                    ]
                ),
                "brier_gain": (
                    entity_brier_gain
                ),
            }

            print(
                "[ENTITY-ABLATION] "
                f"fold={fold.validation_label} "
                f"entity_features="
                f"{len(entity_columns)} "
                f"without="
                f"{no_entity_metrics['brier']:.8f} "
                f"with="
                f"{per_model_metrics['xgb']['brier']:.8f} "
                f"gain="
                f"{entity_brier_gain:+.8f}"
            )

            del (
                no_entity_models,
                no_entity_prediction,
                no_entity_single_prediction,
                X_train_no_entity,
                X_valid_no_entity,
            )
            gc.collect()

        base_xgb_prediction = (
            predictions[
                "xgb"
            ].copy()
        )

        (
            representation_model,
            representative_model,
            representation_prediction,
            representative_prediction,
            representation_raw_features,
            representation_pca_state,
            representative_diagnostics,
        ) = (
            train_xgboost_multiview_fold(
                base_model=(
                    xgb_models[0]
                ),
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                sample_weight=(
                    sample_weight
                ),
                numerical_cols=(
                    preprocessor.num_cols_
                ),
                config=config.models,
                num_boost_round=(
                    best_iterations[
                        "xgb"
                    ]
                ),
            )
        )

        xgb_views = {
            "base": (
                base_xgb_prediction
            ),
            "representation": (
                representation_prediction
            ),
            "representative": (
                representative_prediction
            ),
        }
        xgb_view_metrics = {
            name: evaluate_probabilities(
                y_valid,
                prediction,
            )
            for name, prediction
            in xgb_views.items()
        }

        for (
            view_name,
            view_metrics,
        ) in xgb_view_metrics.items():
            _print_metrics(
                (
                    f"{fold.validation_label}"
                    f"/xgb_view_{view_name}"
                ),
                view_metrics,
            )        

        print(
            "[XGB-MULTIVIEW] "
            f"fold={fold.validation_label} "
            f"repr_width="
            f"{len(representation_raw_features) + config.models.xgb_multiview_pca_components} "
            f"pca_explained="
            f"{sum(representation_pca_state['explained_variance_ratio']):.4f} "
            f"repr_weight_std="
            f"{representative_diagnostics['factor_std']:.4f}"
        )

        del (
            representation_model,
            representative_model,
            representation_prediction,
            representative_prediction,
        )
        del (
            xgb_models,
            xgb_single_prediction,
        )

        # ----------------------------------------------------
        # V13 temporal-distribution XGBoost experts.
        # ----------------------------------------------------

        fold_train_seasons = (
            pd.to_numeric(
                train.iloc[
                    fold.train_idx
                ]["season"],
                errors="raise",
            )
            .to_numpy(
                dtype=np.int16,
                copy=True,
            )
        )

        (
            recent1_positions,
            recent1_window,
        ) = (
            recent_window_positions(
                fold_train_seasons,
                n_seasons=(
                    config.models
                    .xgb_temporal_recent1_seasons
                ),
            )
        )

        (
            recent2_positions,
            recent2_window,
        ) = (
            recent_window_positions(
                fold_train_seasons,
                n_seasons=(
                    config.models
                    .xgb_temporal_recent2_seasons
                ),
            )
        )

        temporal_views = {
            "base": (
                predictions[
                    "xgb"
                ].copy()
            ),
        }

        temporal_view_metrics = {}

        # ---------------- recent1 ----------------

        start = time.perf_counter()

        (
            recent1_model,
            recent1_prediction,
            recent1_rounds,
        ) = (
            train_xgboost_temporal_fold(
                X_train.iloc[
                    recent1_positions
                ],
                y_train[
                    recent1_positions
                ],
                X_valid,
                y_valid,
                sample_weight[
                    recent1_positions
                ],
                config.models,
                seed=(
                    config.models
                    .xgb_temporal_recent1_seed
                ),
            )
        )

        recent1_seconds = (
            time.perf_counter()
            - start
        )

        temporal_views[
            "recent1"
        ] = recent1_prediction

        temporal_view_metrics[
            "recent1"
        ] = evaluate_probabilities(
            y_valid,
            recent1_prediction,
        )

        temporal_view_metrics[
            "recent1"
        ][
            "train_seconds"
        ] = float(
            recent1_seconds
        )

        best_iterations[
            "xgb_recent1"
        ] = int(
            recent1_rounds
        )

        del recent1_model

        # ---------------- recent2 ----------------

        start = time.perf_counter()

        (
            recent2_model,
            recent2_prediction,
            recent2_rounds,
        ) = (
            train_xgboost_temporal_fold(
                X_train.iloc[
                    recent2_positions
                ],
                y_train[
                    recent2_positions
                ],
                X_valid,
                y_valid,
                sample_weight[
                    recent2_positions
                ],
                config.models,
                seed=(
                    config.models
                    .xgb_temporal_recent2_seed
                ),
            )
        )

        recent2_seconds = (
            time.perf_counter()
            - start
        )

        temporal_views[
            "recent2"
        ] = recent2_prediction

        temporal_view_metrics[
            "recent2"
        ] = evaluate_probabilities(
            y_valid,
            recent2_prediction,
        )

        temporal_view_metrics[
            "recent2"
        ][
            "train_seconds"
        ] = float(
            recent2_seconds
        )

        best_iterations[
            "xgb_recent2"
        ] = int(
            recent2_rounds
        )

        del recent2_model

        temporal_view_metrics[
            "base"
        ] = evaluate_probabilities(
            y_valid,
            temporal_views[
                "base"
            ],
        )

        for (
            view_name,
            view_metrics,
        ) in temporal_view_metrics.items():
            _print_metrics(
                (
                    f"{fold.validation_label}"
                    f"/xgb_temporal_{view_name}"
                ),
                view_metrics,
            )

        print(
            "[XGB-TEMPORAL] "
            f"fold={fold.validation_label} "
            f"recent1_seasons="
            f"{recent1_window['selected_seasons']} "
            f"recent1_rows="
            f"{recent1_window['n_rows']:,} "
            f"recent2_seasons="
            f"{recent2_window['selected_seasons']} "
            f"recent2_rows="
            f"{recent2_window['n_rows']:,}"
        )

        gc.collect()

        native_cat_prediction = None
        native_cat_metrics = None

        if (
            config.models
            .xgb_native_cat_enabled
        ):
            native_state = (
                preprocessor
                .export_state()
            )

            X_train_native = (
                preprocess_xgb_native_frame(
                    features.iloc[
                        fold.train_idx
                    ],
                    native_state,
                )
            )

            X_valid_native = (
                preprocess_xgb_native_frame(
                    features.iloc[
                        fold.valid_idx
                    ],
                    native_state,
                )
            )

            start = time.perf_counter()

            (
                native_cat_model,
                native_cat_prediction,
                native_cat_rounds,
            ) = (
                train_xgboost_native_cat_fold(
                    X_train_native,
                    y_train,
                    X_valid_native,
                    y_valid,
                    sample_weight,
                    config.models,
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            native_cat_metrics = (
                evaluate_probabilities(
                    y_valid,
                    native_cat_prediction,
                )
            )

            native_cat_metrics[
                "train_seconds"
            ] = float(
                elapsed
            )

            best_iterations[
                "xgb_native_cat"
            ] = int(
                native_cat_rounds
            )

            _print_metrics(
                (
                    f"{fold.validation_label}"
                    "/xgb_native_cat"
                ),
                native_cat_metrics,
            )

            del (
                native_cat_model,
                X_train_native,
                X_valid_native,
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
        (
            cat_model,
            predictions["cat"],
            best_iterations["cat"],
        ) = train_catboost_fold(
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

            neural_preprocessor = (
                NeuralPreprocessor(
                    categorical_cols=(
                        preprocessor.cat_cols_
                    ),
                    numerical_cols=(
                        preprocessor.num_cols_
                    ),
                )
                .fit(
                    X_train
                )
            )

            neural_state = (
                neural_preprocessor
                .export_state()
            )

            train_arrays = (
                prepare_neural_arrays(
                    X_train,
                    neural_state,
                )
            )

            valid_arrays = (
                prepare_neural_arrays(
                    X_valid,
                    neural_state,
                )
            )

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

        del (
            X_train,
            X_valid,
            sample_weight,
        )
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
                "xgb_views": (
                    xgb_views
                ),

                "xgb_multiview_fold_state": {
                    "representation_raw_features": (
                        representation_raw_features
                    ),
                    "pca_explained_variance_ratio": (
                        representation_pca_state[
                            "explained_variance_ratio"
                        ]
                    ),
                    "representative_diagnostics": (
                        representative_diagnostics
                    ),
                    "view_metrics": (
                        xgb_view_metrics
                    ),                    
                },
                "xgb_lupi_prediction": (
                    lupi_prediction
                ),

                "xgb_lupi_fold_state": (
                    lupi_fold_state
                ),
                "xgb_temporal_views": (
                    temporal_views
                ),

                "xgb_temporal_fold_state": {
                    "recent1_window": (
                        recent1_window
                    ),

                    "recent2_window": (
                        recent2_window
                    ),

                    "view_metrics": (
                        temporal_view_metrics
                    ),
                },
                "xgb_native_cat_prediction": (
                    native_cat_prediction
                ),

                "xgb_native_cat_metrics": (
                    native_cat_metrics
                ),                              
            }
        )

    if len(fold_results) != 3:
        raise RuntimeError(
            "This submission strategy requires 2023, 2024 and late-2024 holdouts."
        )
    
    (
        xgb_view_weights,
        xgb_view_report,
    ) = (
        select_xgb_multiview_weights(
            fold_results,
            fold_importance=(
                config
                .temporal_fold_importance
            ),
            grid_step=(
                config.models
                .xgb_multiview_grid_step
            ),
            minimum_base_weight=(
                config.models
                .xgb_multiview_minimum_base_weight
            ),
            maximum_aux_weight=(
                config.models
                .xgb_multiview_maximum_aux_weight
            ),
            protected_labels=(
                "2024",
                "2024_late_abs",
            ),
            protected_tolerance=(
                config.models
                .xgb_multiview_protected_tolerance
            ),
        )
    )

    print(
        "[XGB-MULTIVIEW-SELECT] "
        f"weights={xgb_view_weights.tolist()} "
        f"forward_brier="
        f"{xgb_view_report['forward_weighted_brier']:.8f} "
        f"gains="
        f"{xgb_view_report['gain_vs_base']}"
    )

    for fold in fold_results:
        upgraded = (
            blend_xgb_views(
                fold[
                    "xgb_views"
                ],
                xgb_view_weights,
            )
        )

        fold[
            "predictions"
        ][
            "xgb"
        ] = upgraded

    # --------------------------------------------------------
    # V13 temporal-view selection.
    #
    # The base is the already validated bag3/feature-view
    # logical XGB.  recent1/recent2 then compete against it.
    # --------------------------------------------------------

    for fold in fold_results:
        fold[
            "xgb_temporal_views"
        ][
            "base"
        ] = (
            fold[
                "predictions"
            ][
                "xgb"
            ].copy()
        )

    (
        temporal_weights,
        temporal_report,
    ) = (
        select_temporal_view_weights(
            fold_results,
            fold_importance=(
                config
                .temporal_fold_importance
            ),

            grid_step=(
                config.models
                .xgb_temporal_grid_step
            ),

            minimum_base_weight=(
                config.models
                .xgb_temporal_minimum_base_weight
            ),

            maximum_aux_weight=(
                config.models
                .xgb_temporal_maximum_aux_weight
            ),

            protected_labels=(
                "2024",
                "2024_late_abs",
            ),

            protected_tolerance=(
                config.models
                .xgb_temporal_protected_tolerance
            ),
        )
    )

    print(
        "[XGB-TEMPORAL-SELECT] "
        f"weights="
        f"{temporal_weights.tolist()} "
        f"forward_brier="
        f"{temporal_report['forward_weighted_brier']:.8f} "
        f"gains="
        f"{temporal_report['gain_vs_base']}"
    )

    for fold in fold_results:
        temporal_metrics = (
            _apply_temporal_xgb_blend(
                fold,
                temporal_weights,
            )
        )

        _print_metrics(
            (
                f"{fold['validation_label']}"
                "/xgb_temporal_blend"
            ),
            temporal_metrics,
        )
    
    lupi_weight = 0.0

    lupi_report = {
        "enabled": bool(
            lupi_enabled
        ),
        "selected_weight": 0.0,
        "accepted_nonzero": False,
    }

    if lupi_enabled:
        (
            lupi_weight,
            lupi_report,
        ) = (
            select_lupi_weight(
                fold_results,
                candidate_weights=(
                    config
                    .privileged
                    .student_weight_candidates
                ),
                fold_importance=(
                    config
                    .temporal_fold_importance
                ),
                minimum_forward_gain=(
                    config
                    .privileged
                    .minimum_forward_gain
                ),
                maximum_2023_regression=(
                    config
                    .privileged
                    .maximum_2023_regression
                ),
            )
        )

        print(
            "[LUPI-SELECT] "
            f"weight="
            f"{lupi_weight:.3f} "
            f"accepted="
            f"{lupi_report['accepted_nonzero']} "
            f"forward_gain="
            f"{lupi_report['forward_gain']:+.8f} "
            f"gains="
            f"{lupi_report['gain_vs_base']}"
        )

        for fold in fold_results:
            base = np.asarray(
                fold[
                    "predictions"
                ][
                    "xgb"
                ],
                dtype=np.float64,
            )

            student = np.asarray(
                fold[
                    "xgb_lupi_prediction"
                ],
                dtype=np.float64,
            )

            upgraded = (
                (
                    1.0
                    - float(
                        lupi_weight
                    )
                )
                * base
                + float(
                    lupi_weight
                )
                * student
            )

            upgraded = np.clip(
                upgraded,
                1.0e-6,
                1.0 - 1.0e-6,
            )

            fold[
                "predictions"
            ][
                "xgb"
            ] = upgraded

            fold[
                "metrics"
            ][
                "xgb"
            ] = (
                evaluate_probabilities(
                    fold[
                        "y_true"
                    ],
                    upgraded,
                )
            )

    native_cat_weight = 0.0

    native_cat_report = {
        "method": (
            "native_categorical_disabled"
        ),
        "selected_weight": 0.0,
        "accepted_nonzero": False,
    }

    if (
        config.models
        .xgb_native_cat_enabled
    ):
        (
            native_cat_weight,
            native_cat_report,
        ) = (
            select_xgb_native_cat_weight(
                fold_results,
                fold_importance=(
                    config
                    .temporal_fold_importance
                ),
                grid_step=(
                    config.models
                    .xgb_native_cat_grid_step
                ),
                maximum_weight=(
                    config.models
                    .xgb_native_cat_max_weight
                ),
                minimum_material_protected_gain=(
                    config.models
                    .xgb_native_cat_min_material_protected_gain
                ),
                minimum_forward_improvement=(
                    config.models
                    .xgb_native_cat_min_forward_improvement
                ),
                maximum_2023_regression=(
                    config.models
                    .xgb_native_cat_max_2023_regression
                ),
            )
        )

        print(
            "[XGB-NATIVE-CAT-SELECT] "
            f"weight={native_cat_weight:.3f} "
            f"forward_gain="
            f"{native_cat_report['forward_improvement']:+.8f} "
            f"gains="
            f"{native_cat_report['gain_vs_base']}"
        )

        for fold in fold_results:
            upgraded = (
                blend_xgb_native_cat(
                    fold[
                        "predictions"
                    ][
                        "xgb"
                    ],
                    fold[
                        "xgb_native_cat_prediction"
                    ],
                    native_cat_weight,
                )
            )

            fold[
                "predictions"
            ][
                "xgb"
            ] = upgraded

            fold[
                "metrics"
            ][
                "xgb"
            ] = (
                evaluate_probabilities(
                    fold[
                        "y_true"
                    ],
                    upgraded,
                )
            )

            _print_metrics(
                (
                    f"{fold['validation_label']}"
                    "/xgb_native_cat_blend"
                ),
                fold[
                    "metrics"
                ][
                    "xgb"
                ],
            )

    if (
        config
        .ensemble_weight_policy
        == "fixed"
    ):
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

    elif (
        config
        .ensemble_weight_policy
        == "shrinkage"
    ):
        (
            weights,
            weight_report,
            calibrator,
        ) = (
            select_calibration_aware_shrunk_weights(
                fold_results,
                fold_importance=(
                    config
                    .temporal_fold_importance
                ),
                step=(
                    config
                    .ensemble_grid_step
                ),
                model_order=model_order,
                protected_fold_labels=(
                    config
                    .ensemble_protected_folds
                ),
                non_degradation_tolerance=(
                    config
                    .ensemble_non_degradation_tolerance
                ),
                reference_model=(
                    config
                    .ensemble_shrinkage_reference_model
                ),
                alpha_grid=(
                    config
                    .ensemble_shrinkage_alphas
                ),
                early_month_max=(
                    config
                    .abs_late_train_month_max
                ),
                maximum_2023_regression_vs_reference=(
                    config
                    .submission_gate_2023_max_regression_vs_xgb
                ),
                minimum_calibration_transfer_gain=(
                    config
                    .submission_gate_calibration_min_transfer_gain
                ),
            )
        )

    else:
        raise ValueError(
            "Unknown ensemble policy: "
            f"{config.ensemble_weight_policy}"
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
                "xgb_multiview_fold_state": (
                    fold[
                        "xgb_multiview_fold_state"
                    ]
                ),       
                "xgb_temporal_fold_state": (
                    fold[
                        "xgb_temporal_fold_state"
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

    final_iterations[
        "xgb_recent1"
    ] = int(
        min(
            config.models
            .xgb_num_boost_round,
            max(
                30,
                round(
                    recent_iterations[
                        "xgb_recent1"
                    ]
                    * 1.05
                ),
            ),
        )
    )

    final_iterations[
        "xgb_recent2"
    ] = int(
        min(
            config.models
            .xgb_num_boost_round,
            max(
                30,
                round(
                    recent_iterations[
                        "xgb_recent2"
                    ]
                    * 1.05
                ),
            ),
        )
    )

    if (
        config.models
        .xgb_native_cat_enabled
    ):
        final_iterations[
            "xgb_native_cat"
        ] = int(
            min(
                config.models
                .xgb_num_boost_round,
                max(
                    50,
                    round(
                        recent_iterations[
                            "xgb_native_cat"
                        ]
                        * 1.05
                    ),
                ),
            )
        )

    if lupi_enabled:
        final_iterations[
            "xgb_lupi_student"
        ] = int(
            min(
                config
                .models
                .xgb_num_boost_round,

                max(
                    30,
                    round(
                        recent_iterations[
                            "xgb_lupi_student"
                        ]
                        * 1.05
                    ),
                ),
            )
        )

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
        "model_order": list(
            model_order
        ),

        "weights": (
            weights.tolist()
        ),

        "calibration": (
            calibrator
        ),

        "final_iterations": (
            final_iterations
        ),

        "xgb_multiview": {
            "view_order": (
                xgb_view_report[
                    "view_order"
                ]
            ),
            "weights": (
                xgb_view_weights.tolist()
            ),
            "validation": (
                xgb_view_report
            ),
        },
        "xgb_temporal_views": {
            "view_order": (
                temporal_report[
                    "view_order"
                ]
            ),

            "weights": (
                temporal_weights.tolist()
            ),

            "validation": (
                temporal_report
            ),
        },    
        "xgb_lupi": {
            "enabled": bool(
                lupi_enabled
            ),
            "weight": float(
                lupi_weight
            ),
            "validation": (
                lupi_report
            ),
        },         
        "xgb_native_categorical": {
            "enabled": bool(
                config.models
                .xgb_native_cat_enabled
            ),
            "weight": float(
                native_cat_weight
            ),
            "validation": (
                native_cat_report
            ),
        },          
    }
    report = {
        "strategy": _strategy_name(
            tuple(
                ensemble_state[
                    "model_order"
                ]
            ),
            ensemble_policy=(
                config
                .ensemble_weight_policy
            ),
            lupi_enabled=bool(
                config
                .privileged
                .enabled
            ),
            native_cat_enabled=bool(
                config
                .models
                .xgb_native_cat_enabled
            ),
        ),
        "folds": fold_summaries,
        "weight_selection": weight_report,
        "calibration": calibrator,
        "calibration_transfer_metrics": transfer_metrics,
        "final_iterations": final_iterations,
        "xgb_multiview": (
            xgb_view_report
        ),
        "xgb_temporal_views": (
            temporal_report
        ),        
        "xgb_native_categorical": (
            native_cat_report
        ),       
        "xgb_lupi": (
            lupi_report
        ),         
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

    xgb_mv_2024_gain = float(
        xgb_view_report[
            "gain_vs_base"
        ]["2024"]
    )

    xgb_mv_late_gain = float(
        xgb_view_report[
            "gain_vs_base"
        ]["2024_late_abs"]
    )

    xgb_temporal_2024_gain = float(
        temporal_report[
            "gain_vs_base"
        ][
            "2024"
        ]
    )

    xgb_temporal_late_gain = float(
        temporal_report[
            "gain_vs_base"
        ][
            "2024_late_abs"
        ]
    )    

    forward_brier = float(
        weight_report[
            "forward_weighted_brier"
        ]
    )

    reference_2023_brier = float(
        by_label[
            "2023"
        ][
            "models"
        ][
            reference_model
        ][
            "brier"
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
            <= reference_2023_brier
            + config
            .submission_gate_2023_max_regression_vs_xgb
        ),

        "calibration_accepted": bool(
            calibrator[
                "accepted"
            ]
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

        "entity_2024_paired_ablation": (
            not entity_gate_required
            or entity_2024_gain
            >= -config
            .submission_gate_entity_2024_max_regression
        ),

        "entity_late_paired_ablation": (
            not entity_gate_required
            or entity_late_gain
            >= config
            .submission_gate_entity_late_min_gain
        ),

        # The champion bag3 must still be healthy.
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

        "xgb_lupi_material_gain": (
            not lupi_enabled
            or bool(
                lupi_report[
                    "accepted_nonzero"
                ]
            )
        ),        

        # Multi-view selection explicitly permits only a very small
        # protected-fold degradation while searching for complementary
        # signal.
        "xgb_multiview_2024_protected": (
            xgb_mv_2024_gain
            + config.models
            .xgb_multiview_protected_tolerance
            + 1.0e-15
            >= 0.0
        ),

        "xgb_multiview_late_protected": (
            xgb_mv_late_gain
            + config.models
            .xgb_multiview_protected_tolerance
            + 1.0e-15
            >= 0.0
        ),

        "xgb_temporal_2024_protected": (
            xgb_temporal_2024_gain
            + config.models
            .xgb_temporal_protected_tolerance
            + 1.0e-15
            >= 0.0
        ),

        "xgb_temporal_late_protected": (
            xgb_temporal_late_gain
            + config.models
            .xgb_temporal_protected_tolerance
            + 1.0e-15
            >= 0.0
        ),

        # Compare the actually deployed predictor against the
        # frozen 907 validation champion.
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

    diagnostics = {
        "2024_ensemble_gain_vs_current_xgb": float(
            gain_2024_vs_reference
        ),

        "late_raw_ensemble_gain_vs_current_xgb": float(
            gain_late_raw_vs_reference
        ),

        "late_calibrated_gain_vs_current_xgb": float(
            gain_late_calibrated_vs_reference
        ),

        "xgb_multiview_2024_gain_vs_bag3": float(
            xgb_mv_2024_gain
        ),

        "xgb_multiview_late_gain_vs_bag3": float(
            xgb_mv_late_gain
        ),

        "xgb_temporal_weights": (
            temporal_weights.tolist()
        ),

        "xgb_temporal_2024_gain_vs_base": float(
            xgb_temporal_2024_gain
        ),

        "xgb_temporal_late_gain_vs_base": float(
            xgb_temporal_late_gain
        ),

        "xgb_temporal_forward_brier": float(
            temporal_report[
                "forward_weighted_brier"
            ]
        ),        
    }

    # --------------------------------------------------------
    # Build quality-gate result first.
    #
    # IMPORTANT:
    # Do not read report["submission_gate"] before assigning it.
    #
    # Quality thresholds are advisory in the default production
    # path. scripts/train_submit.py decides whether a failed gate
    # blocks packaging; --require-quality-gate enables strict mode.
    # --------------------------------------------------------

    gate_passed = bool(
        all(
            checks.values()
        )
    )

    report[
        "submission_gate"
    ] = {
        "passed": gate_passed,

        "checks": checks,

        "diagnostics": (
            diagnostics
        ),        

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

            "xgb_multiview_weights": (
                xgb_view_weights.tolist()
            ),

            "xgb_multiview_2024_gain": (
                xgb_mv_2024_gain
            ),

            "xgb_multiview_late_gain": (
                xgb_mv_late_gain
            ),

            "xgb_temporal_weights": (
                temporal_weights.tolist()
            ),

            "xgb_temporal_2024_gain": (
                xgb_temporal_2024_gain
            ),

            "xgb_temporal_late_gain": (
                xgb_temporal_late_gain
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
            "2023_max_regression_vs_xgb": float(
                config
                .submission_gate_2023_max_regression_vs_xgb
            ),

            "2024_min_gain_vs_reference": float(
                config
                .submission_gate_2024_min_gain_vs_reference
            ),

            "late_raw_min_gain_vs_reference": float(
                config
                .submission_gate_late_raw_min_gain_vs_reference
            ),

            "calibration_min_transfer_gain": float(
                config
                .submission_gate_calibration_min_transfer_gain
            ),

            "late_calibrated_min_gain_vs_reference": float(
                config
                .submission_gate_late_calibrated_min_gain_vs_reference
            ),

            "entity_2024_max_regression": float(
                config
                .submission_gate_entity_2024_max_regression
            ),

            "entity_late_min_gain": float(
                config
                .submission_gate_entity_late_min_gain
            ),

            "xgb_bagging_2024_min_gain": float(
                config
                .submission_gate_xgb_bagging_2024_min_gain
            ),

            "xgb_bagging_late_min_gain": float(
                config
                .submission_gate_xgb_bagging_late_min_gain
            ),

            "minimum_forward_improvement": float(
                config
                .submission_gate_min_forward_improvement
            ),

            "previous_2024_min_gain": float(
                config
                .submission_gate_previous_2024_min_gain
            ),

            "previous_late_calibrated_min_gain": float(
                config
                .submission_gate_previous_late_calibrated_min_gain
            ),
        },
    }

    # --------------------------------------------------------
    # Informational recommendation only.
    #
    # Actual artifact policy belongs to scripts/train_submit.py.
    # Default production behavior is package_always.
    # --------------------------------------------------------

    report[
        "artifact_recommendation"
    ] = {
        "quality_gate_passed": (
            gate_passed
        ),
        "quality_gate_is_advisory": True,
        "package_artifact": True,
    }

    return ensemble_state, report


def train_and_save_final_models(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_state: Mapping[str, object],
    ensemble_state: Mapping[str, object],
    config: ExperimentConfig,
    model_dir: Path,
    *,
    privileged_state=None,
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
    outer_weights = (
        _outer_weight_map(
            ensemble_state
        )
    )

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

    (
        representation_model,
        representative_model,
        representation_raw_features,
        representation_pca_state,
        representative_diagnostics,
    ) = (
        train_xgboost_multiview_full(
            base_model=xgb_models[0],
            X_train=X,
            y_train=y,
            sample_weight=(
                sample_weight
            ),
            numerical_cols=(
                preprocessor.num_cols_
            ),
            config=config.models,
            num_boost_round=(
                xgb_rounds
            ),
        )
    )

    representation_path = (
        model_dir
        / "xgb_representation.json"
    )

    representative_path = (
        model_dir
        / "xgb_representative.json"
    )

    representation_model.save_model(
        str(
            representation_path
        )
    )

    representative_model.save_model(
        str(
            representative_path
        )
    )

    del (
        representation_model,
        representative_model,
    )
    del xgb_models
    gc.collect()

    final_seasons = (
        pd.to_numeric(
            train["season"],
            errors="raise",
        )
        .to_numpy(
            dtype=np.int16,
            copy=True,
        )
    )

    (
        recent1_positions,
        recent1_window,
    ) = recent_window_positions(
        final_seasons,
        n_seasons=(
            config.models
            .xgb_temporal_recent1_seasons
        ),
    )

    (
        recent2_positions,
        recent2_window,
    ) = recent_window_positions(
        final_seasons,
        n_seasons=(
            config.models
            .xgb_temporal_recent2_seasons
        ),
    )

    print(
        "[FINAL] Training temporal recent1 "
        f"seasons="
        f"{recent1_window['selected_seasons']} "
        f"rows="
        f"{recent1_window['n_rows']:,}"
    )

    recent1_model = (
        train_xgboost_temporal_full(
            X.iloc[
                recent1_positions
            ],
            y[
                recent1_positions
            ],
            sample_weight[
                recent1_positions
            ],
            config.models,
            seed=(
                config.models
                .xgb_temporal_recent1_seed
            ),
            num_boost_round=int(
                ensemble_state[
                    "final_iterations"
                ][
                    "xgb_recent1"
                ]
            ),
        )
    )

    recent1_path = (
        model_dir
        / "xgb_recent1.json"
    )

    recent1_model.save_model(
        str(
            recent1_path
        )
    )

    del recent1_model

    print(
        "[FINAL] Training temporal recent2 "
        f"seasons="
        f"{recent2_window['selected_seasons']} "
        f"rows="
        f"{recent2_window['n_rows']:,}"
    )

    recent2_model = (
        train_xgboost_temporal_full(
            X.iloc[
                recent2_positions
            ],
            y[
                recent2_positions
            ],
            sample_weight[
                recent2_positions
            ],
            config.models,
            seed=(
                config.models
                .xgb_temporal_recent2_seed
            ),
            num_boost_round=int(
                ensemble_state[
                    "final_iterations"
                ][
                    "xgb_recent2"
                ]
            ),
        )
    )

    recent2_path = (
        model_dir
        / "xgb_recent2.json"
    )

    recent2_model.save_model(
        str(
            recent2_path
        )
    )

    del recent2_model

    gc.collect()

    lupi_path = None

    lupi_state = (
        ensemble_state.get(
            "xgb_lupi",
            {},
        )
    )

    lupi_weight = float(
        lupi_state.get(
            "weight",
            0.0,
        )
    )

    if (
        config.privileged.enabled
        and lupi_weight > 1.0e-12
    ):
        if privileged_state is None:
            raise RuntimeError(
                "Final LUPI training "
                "requires privileged_state."
            )

        teacher_probability = np.asarray(
            privileged_state[
                "teacher_probability"
            ],
            dtype=np.float32,
        )

        teacher_available = np.asarray(
            privileged_state[
                "teacher_available"
            ],
            dtype=bool,
        )

        y_distill = (
            build_distillation_target(
                y,
                teacher_probability,
                teacher_available,
                strength=(
                    config
                    .privileged
                    .distill_strength
                ),
            )
        )

        lupi_rounds = int(
            ensemble_state[
                "final_iterations"
            ][
                "xgb_lupi_student"
            ]
        )

        print(
            "[FINAL] Training "
            "pre-pitch LUPI XGBoost "
            f"for {lupi_rounds} rounds "
            f"distilled_rows="
            f"{teacher_available.sum():,}"
        )

        lupi_model = (
            train_xgboost_temporal_full(
                X,
                y_distill,
                uniform_weights(
                    len(y_distill)
                ),
                config.models,
                seed=(
                    config
                    .privileged
                    .student_seed
                ),
                num_boost_round=(
                    lupi_rounds
                ),
            )
        )

        lupi_path = (
            model_dir
            / "xgb_lupi_student.json"
        )

        lupi_model.save_model(
            str(
                lupi_path
            )
        )

        del (
            lupi_model,
            y_distill,
        )

        gc.collect()    

    native_cat_path = None

    if (
        config.models
        .xgb_native_cat_enabled
    ):
        print(
            "[FINAL] Building native "
            "categorical XGBoost frame..."
        )

        X_native = (
            preprocess_xgb_native_frame(
                features,
                preprocessor
                .export_state(),
            )
        )

        native_rounds = int(
            ensemble_state[
                "final_iterations"
            ][
                "xgb_native_cat"
            ]
        )

        print(
            "[FINAL] Training native "
            "categorical XGBoost "
            f"for {native_rounds} rounds..."
        )

        native_cat_model = (
            train_xgboost_native_cat_full(
                X_native,
                y,
                sample_weight,
                config.models,
                num_boost_round=(
                    native_rounds
                ),
            )
        )

        native_cat_path = (
            model_dir
            / "xgb_native_cat.json"
        )

        native_cat_model.save_model(
            str(
                native_cat_path
            )
        )

        del (
            native_cat_model,
            X_native,
        )

        gc.collect()

    lgb_path = None

    if _outer_active(
        outer_weights,
        "lgb",
    ):
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
            model_dir
            / "lgb_model.txt"
        )

        lgb_model.save_model(
            str(lgb_path)
        )

        del lgb_model
        gc.collect()

    else:
        print(
            "[FINAL] Skipping LightGBM "
            "because outer ensemble weight is zero."
        )

    cat_path = None

    if _outer_active(
        outer_weights,
        "cat",
    ):
        cat_iterations = int(
            ensemble_state[
                "final_iterations"
            ]["cat"]
        )

        print(
            "[FINAL] Training CatBoost "
            f"for {cat_iterations} iterations..."
        )

        cat_model = train_catboost_full(
            X,
            y,
            sample_weight,
            preprocessor.categorical_indices,
            config.models,
            iterations=cat_iterations,
        )

        cat_path = (
            model_dir
            / "cat_model.cbm"
        )

        cat_model.save_model(
            str(cat_path)
        )

        del cat_model
        gc.collect()

    else:
        print(
            "[FINAL] Skipping CatBoost "
            "because outer ensemble weight is zero."
        )

    neural_state = None
    neural_paths: Dict[str, Path] = {}
    neural_model_order = tuple(
        name
        for name
        in ensemble_state[
            "model_order"
        ]
        if name
        in {
            "resnet",
            "ft_transformer",
        }
        and _outer_active(
            outer_weights,
            name,
        )
    )
    if neural_model_order:
        from src.neural import (
            NeuralPreprocessor,
            prepare_neural_arrays,
            release_torch_memory,
            save_checkpoint,
            train_neural_full,
        )

        neural_preprocessor = (
            NeuralPreprocessor(
                categorical_cols=(
                    preprocessor.cat_cols_
                ),
                numerical_cols=(
                    preprocessor.num_cols_
                ),
            )
            .fit(X)
        )

        neural_state = (
            neural_preprocessor
            .export_state()
        )

        train_arrays = (
            prepare_neural_arrays(
                X,
                neural_state,
            )
        )
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
        "preprocessor_state": (
            preprocessor.export_state()
        ),

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
        "xgb_multiview": {
            "view_order": [
                "base",
                "representation",
                "representative",
            ],

            "weights": (
                ensemble_state[
                    "xgb_multiview"
                ]["weights"]
            ),

            "representation_model_file": (
                "xgb_representation.json"
            ),

            "representative_model_file": (
                "xgb_representative.json"
            ),

            "representation_raw_features": (
                representation_raw_features
            ),

            "pca_state": (
                representation_pca_state
            ),

            "representative_diagnostics": (
                representative_diagnostics
            ),
        },
        "xgb_lupi": {
            "enabled": bool(
                lupi_path
                is not None
                and lupi_weight
                > 1.0e-12
            ),
            "weight": float(
                lupi_weight
            ),
            "model_file": (
                lupi_path.name
                if lupi_path
                is not None
                else None
            ),
        },        
        "xgb_native_categorical": {
            "enabled": bool(
                config.models
                .xgb_native_cat_enabled
            ),
            "weight": float(
                ensemble_state[
                    "xgb_native_categorical"
                ][
                    "weight"
                ]
            ),
            "model_file": (
                native_cat_path.name
                if native_cat_path
                is not None
                else None
            ),
            "feature_count": int(
                len(
                    preprocessor
                    .feature_names_
                )
            ),
            "validation": (
                ensemble_state[
                    "xgb_native_categorical"
                ][
                    "validation"
                ]
            ),
        },        
        "xgb_temporal_views": {
            "view_order": [
                "base",
                "recent1",
                "recent2",
            ],

            "weights": (
                ensemble_state[
                    "xgb_temporal_views"
                ][
                    "weights"
                ]
            ),

            "model_files": {
                "recent1": (
                    recent1_path.name
                ),

                "recent2": (
                    recent2_path.name
                ),
            },

            "recent1_window": (
                recent1_window
            ),

            "recent2_window": (
                recent2_window
            ),
        },        
    }
    bundle_path = model_dir / "bundle.pkl"
    joblib.dump(bundle, bundle_path, compress=3)

    model_files = [
        *[
            path.name
            for path
            in xgb_paths
        ],

        # Internal XGB experts currently remain
        # packaged unconditionally for the first
        # V16 correctness run.
        representation_path.name,
        representative_path.name,
        recent1_path.name,
        recent2_path.name,

        bundle_path.name,
    ]

    if lgb_path is not None:
        model_files.append(
            lgb_path.name
        )

    if cat_path is not None:
        model_files.append(
            cat_path.name
        )

    for name in neural_model_order:
        model_files.append(
            neural_paths[
                name
            ].name
        )

    if native_cat_path is not None:
        model_files.append(
            native_cat_path.name
        )

    if lupi_path is not None:
        model_files.append(
            lupi_path.name
        )

    model_files = list(
        dict.fromkeys(
            model_files
        )
    )

    manifest = {
        "strategy": _strategy_name(
            tuple(
                ensemble_state[
                    "model_order"
                ]
            ),
            ensemble_policy=(
                config
                .ensemble_weight_policy
            ),
            lupi_enabled=bool(
                config
                .privileged
                .enabled
            ),
            native_cat_enabled=bool(
                config
                .models
                .xgb_native_cat_enabled
            ),
        ),
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
        "model_files": (
            model_files
        ),
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
        "xgb_multiview": {
            "view_order": list(
                ensemble_state[
                    "xgb_multiview"
                ][
                    "view_order"
                ]
            ),

            "weights": list(
                ensemble_state[
                    "xgb_multiview"
                ][
                    "weights"
                ]
            ),

            "representation_model_file": (
                representation_path.name
            ),

            "representative_model_file": (
                representative_path.name
            ),

            "representation_feature_count": int(
                len(
                    representation_raw_features
                )
                + config.models
                .xgb_multiview_pca_components
            ),

            "representation_raw_feature_count": int(
                len(
                    representation_raw_features
                )
            ),

            "pca_component_count": int(
                config.models
                .xgb_multiview_pca_components
            ),
        }, 
        "xgb_native_categorical": {
            "enabled": bool(
                config.models
                .xgb_native_cat_enabled
            ),
            "weight": float(
                ensemble_state[
                    "xgb_native_categorical"
                ][
                    "weight"
                ]
            ),
            "model_file": (
                native_cat_path.name
                if native_cat_path
                is not None
                else None
            ),
            "feature_count": int(
                len(
                    preprocessor
                    .feature_names_
                )
            ),
        },        
        "xgb_temporal_views": {
            "view_order": [
                "base",
                "recent1",
                "recent2",
            ],

            "weights": list(
                ensemble_state[
                    "xgb_temporal_views"
                ][
                    "weights"
                ]
            ),

            "model_files": {
                "recent1": (
                    recent1_path.name
                ),

                "recent2": (
                    recent2_path.name
                ),
            },
        },              
    }
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest

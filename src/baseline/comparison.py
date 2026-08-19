from __future__ import annotations

import gc
import time
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.baseline.pitcher_residual import (
    PitcherBaselineConfig,
    pitcher_baseline_probability,
    probability_to_margin,
    select_strict_residual_features,
    train_residual_xgboost_fold,
)
from src.config import ExperimentConfig
from src.metrics import evaluate_probabilities
from src.models import train_xgboost_fold
from src.preprocessing import TabularPreprocessor
from src.splits import make_abs_late_fold, make_temporal_folds, uniform_weights


DIRECT_MODEL_NAME = "direct_xgb"


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        return None
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def clustered_brier_delta_bootstrap(
    y_true: np.ndarray,
    candidate_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    clusters: Sequence[object],
    *,
    n_bootstrap: int = 2000,
    random_seed: int = 2026,
) -> Dict[str, float | int]:
    """Paired cluster bootstrap for candidate-minus-reference Brier.

    Rows from the same pitcher are resampled together.  Aggregating the paired
    loss differences by cluster makes thousands of replicates inexpensive even
    for the 250k-row 2024 validation fold.
    """
    y = np.asarray(y_true, dtype=np.float64)
    candidate = np.asarray(candidate_prediction, dtype=np.float64)
    reference = np.asarray(reference_prediction, dtype=np.float64)
    cluster_values = pd.Series(clusters).fillna("__MISSING__").astype(str)
    if not (len(y) == len(candidate) == len(reference) == len(cluster_values)):
        raise ValueError("Bootstrap arrays and clusters must have equal length.")
    if int(n_bootstrap) < 100:
        raise ValueError("n_bootstrap must be at least 100.")

    paired_loss = (candidate - y) ** 2 - (reference - y) ** 2
    aggregated = (
        pd.DataFrame({"cluster": cluster_values, "loss": paired_loss})
        .groupby("cluster", sort=False, observed=True)["loss"]
        .agg(["sum", "size"])
    )
    loss_sum = aggregated["sum"].to_numpy(dtype=np.float64)
    cluster_size = aggregated["size"].to_numpy(dtype=np.float64)
    n_clusters = len(aggregated)
    if n_clusters < 2:
        raise ValueError("At least two pitcher clusters are required.")

    rng = np.random.default_rng(int(random_seed))
    samples = np.empty(int(n_bootstrap), dtype=np.float64)
    # Chunk the draws to avoid a large n_bootstrap x n_cluster allocation.
    chunk_size = 256
    for start in range(0, int(n_bootstrap), chunk_size):
        stop = min(start + chunk_size, int(n_bootstrap))
        draw = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        samples[start:stop] = loss_sum[draw].sum(axis=1) / cluster_size[draw].sum(axis=1)

    point_delta = float(paired_loss.mean())
    return {
        "n_clusters": int(n_clusters),
        "n_bootstrap": int(n_bootstrap),
        "brier_delta": point_delta,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_improves": float(np.mean(samples < 0.0)),
    }


def _metric_delta(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
) -> Dict[str, float]:
    return {
        key: float(candidate[key] - reference[key])
        for key in ("brier", "logloss", "calibration_bias")
    }


def _variant_specs(
    strengths: Iterable[float],
    full_strength: float | None,
) -> list[tuple[str, str, float]]:
    unique_strengths = []
    for value in strengths:
        strength = float(value)
        if strength <= 0.0:
            raise ValueError("Every prior strength must be positive.")
        if strength not in unique_strengths:
            unique_strengths.append(strength)
    if not unique_strengths:
        raise ValueError("At least one context residual strength is required.")

    specs = [
        (f"residual_context_s{int(value)}", "context", value)
        for value in unique_strengths
    ]
    if full_strength is not None:
        value = float(full_strength)
        if value <= 0.0:
            raise ValueError("full_strength must be positive.")
        specs.append((f"residual_full_s{int(value)}", "full", value))
    return specs


def run_pitcher_residual_comparison(
    train: pd.DataFrame,
    features: pd.DataFrame,
    config: ExperimentConfig,
    *,
    prior_probability: float = 0.5,
    context_strengths: Sequence[float] = (25.0, 100.0, 500.0),
    full_strength: float | None = 100.0,
    probability_clip: float = 0.02,
    bootstrap_samples: int = 2000,
) -> Dict[str, object]:
    """Compare current direct XGBoost against pitcher-offset alternatives."""
    if len(train) != len(features):
        raise ValueError("train and features must contain the same rows.")
    target_col = config.features.target_col
    variant_specs = _variant_specs(context_strengths, full_strength)
    folds = list(make_temporal_folds(train, config.temporal_folds))
    folds.append(
        make_abs_late_fold(
            train,
            train_month_max=config.abs_late_train_month_max,
            valid_months=config.abs_late_valid_months,
        )
    )
    if len(folds) != len(config.temporal_fold_importance):
        raise ValueError("Fold count and temporal_fold_importance must match.")

    fold_reports = []
    for fold_index, fold in enumerate(folds):
        print("\n" + "=" * 88)
        print(
            f"[RESIDUAL-FOLD] {fold.name}: n_train={len(fold.train_idx):,}, "
            f"n_valid={len(fold.valid_idx):,}"
        )
        print("=" * 88)
        train_rows = train.iloc[fold.train_idx]
        valid_rows = train.iloc[fold.valid_idx]
        y_train = train_rows[target_col].to_numpy(dtype=np.float32, copy=True)
        y_valid = valid_rows[target_col].to_numpy(dtype=np.float32, copy=True)
        sample_weight = uniform_weights(len(y_train))

        preprocessor = TabularPreprocessor(
            categorical_cols=config.features.categorical_cols,
            excluded_cols=config.features.excluded_cols,
        )
        X_train = preprocessor.fit_transform(features.iloc[fold.train_idx])
        X_valid = preprocessor.transform(features.iloc[fold.valid_idx])
        X_train_context = select_strict_residual_features(X_train)
        X_valid_context = X_valid.loc[:, X_train_context.columns]

        model_metrics: Dict[str, Dict[str, float]] = {}
        best_iterations: Dict[str, int] = {}
        timings: Dict[str, float] = {}
        comparisons: Dict[str, Dict[str, object]] = {}
        baseline_metrics: Dict[str, Dict[str, float]] = {}

        started = time.perf_counter()
        direct_model, direct_prediction, direct_iterations = train_xgboost_fold(
            X_train,
            y_train,
            X_valid,
            y_valid,
            sample_weight,
            config.models,
        )
        timings[DIRECT_MODEL_NAME] = float(time.perf_counter() - started)
        best_iterations[DIRECT_MODEL_NAME] = int(direct_iterations)
        model_metrics[DIRECT_MODEL_NAME] = evaluate_probabilities(
            y_valid, direct_prediction
        )
        print(
            f"[RESULT] {fold.validation_label}/{DIRECT_MODEL_NAME} "
            f"brier={model_metrics[DIRECT_MODEL_NAME]['brier']:.8f} "
            f"auc={model_metrics[DIRECT_MODEL_NAME]['auc']:.6f}"
        )
        del direct_model
        gc.collect()

        baseline_cache: Dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for variant_offset, (name, mode, strength) in enumerate(variant_specs, start=1):
            if strength not in baseline_cache:
                baseline_config = PitcherBaselineConfig(
                    prior_probability=float(prior_probability),
                    prior_strength=float(strength),
                    probability_clip=float(probability_clip),
                )
                train_probability = pitcher_baseline_probability(
                    features.iloc[fold.train_idx], baseline_config
                )
                valid_probability = pitcher_baseline_probability(
                    features.iloc[fold.valid_idx], baseline_config
                )
                baseline_cache[strength] = (
                    probability_to_margin(train_probability),
                    probability_to_margin(valid_probability),
                )
                baseline_metrics[f"pitcher_baseline_s{int(strength)}"] = (
                    evaluate_probabilities(y_valid, valid_probability)
                )

            train_margin, valid_margin = baseline_cache[strength]
            candidate_X_train = X_train_context if mode == "context" else X_train
            candidate_X_valid = X_valid_context if mode == "context" else X_valid
            started = time.perf_counter()
            model, prediction, iterations = train_residual_xgboost_fold(
                candidate_X_train,
                y_train,
                train_margin,
                candidate_X_valid,
                y_valid,
                valid_margin,
                sample_weight,
                config.models,
            )
            timings[name] = float(time.perf_counter() - started)
            best_iterations[name] = int(iterations)
            metrics = evaluate_probabilities(y_valid, prediction)
            model_metrics[name] = metrics
            bootstrap = clustered_brier_delta_bootstrap(
                y_valid,
                prediction,
                direct_prediction,
                valid_rows["pitcher_id"].to_numpy(copy=False),
                n_bootstrap=int(bootstrap_samples),
                random_seed=int(config.models.random_seed + 100 * fold_index + variant_offset),
            )
            comparisons[name] = {
                "delta_vs_direct": _metric_delta(
                    metrics, model_metrics[DIRECT_MODEL_NAME]
                ),
                "prediction_correlation": _safe_correlation(
                    prediction, direct_prediction
                ),
                "error_correlation": _safe_correlation(
                    prediction - y_valid, direct_prediction - y_valid
                ),
                "pitcher_cluster_bootstrap": bootstrap,
                "feature_mode": mode,
                "n_features": int(candidate_X_train.shape[1]),
                "prior_strength": float(strength),
            }
            print(
                f"[RESULT] {fold.validation_label}/{name} "
                f"brier={metrics['brier']:.8f} "
                f"delta={metrics['brier'] - model_metrics[DIRECT_MODEL_NAME]['brier']:+.8f} "
                f"CI=[{bootstrap['ci_low']:+.8f},{bootstrap['ci_high']:+.8f}]"
            )
            del model, prediction
            gc.collect()

        fold_reports.append(
            {
                "fold_name": fold.name,
                "validation_label": fold.validation_label,
                "n_train": int(len(fold.train_idx)),
                "n_valid": int(len(fold.valid_idx)),
                "n_all_features": int(X_train.shape[1]),
                "n_context_features": int(X_train_context.shape[1]),
                "models": model_metrics,
                "baseline_only": baseline_metrics,
                "comparisons": comparisons,
                "best_iterations": best_iterations,
                "train_seconds": timings,
            }
        )
        del (
            X_train,
            X_valid,
            X_train_context,
            X_valid_context,
            direct_prediction,
            sample_weight,
            baseline_cache,
        )
        gc.collect()

    importance = np.asarray(config.temporal_fold_importance, dtype=np.float64)
    importance = importance / importance.sum()
    candidate_names = [name for name, _, _ in variant_specs]
    weighted_brier = {
        name: float(
            np.dot(
                importance,
                [fold["models"][name]["brier"] for fold in fold_reports],
            )
        )
        for name in (DIRECT_MODEL_NAME, *candidate_names)
    }
    decisions: Dict[str, Dict[str, object]] = {}
    for name in candidate_names:
        weighted_delta = float(
            weighted_brier[name] - weighted_brier[DIRECT_MODEL_NAME]
        )
        protected = [
            fold
            for fold in fold_reports
            if fold["validation_label"] in config.ensemble_protected_folds
        ]
        protected_deltas = {
            fold["validation_label"]: float(
                fold["comparisons"][name]["delta_vs_direct"]["brier"]
            )
            for fold in protected
        }
        protected_non_degradation = all(
            value <= float(config.ensemble_non_degradation_tolerance)
            for value in protected_deltas.values()
        )
        ci_supported = all(
            float(
                fold["comparisons"][name]["pitcher_cluster_bootstrap"]["ci_high"]
            )
            < 0.0
            for fold in protected
        )
        if weighted_delta < 0.0 and protected_non_degradation and ci_supported:
            status = "strong_replace_candidate"
        elif weighted_delta < 0.0 and protected_non_degradation:
            status = "eligible_for_constrained_ensemble_test"
        else:
            status = "reject_as_direct_xgb_replacement"
        decisions[name] = {
            "status": status,
            "weighted_brier_delta_vs_direct": weighted_delta,
            "protected_fold_deltas": protected_deltas,
            "protected_non_degradation": bool(protected_non_degradation),
            "protected_ci_supported": bool(ci_supported),
        }

    return {
        "experiment": "pitcher_logit_offset_residual_xgboost_v1",
        "reference_model": DIRECT_MODEL_NAME,
        "baseline_formula": "(n * asof_rate + strength * prior) / (n + strength)",
        "prior_probability": float(prior_probability),
        "probability_clip": float(probability_clip),
        "context_strengths": [float(value) for value in context_strengths],
        "full_strength": None if full_strength is None else float(full_strength),
        "fold_importance": importance.tolist(),
        "folds": fold_reports,
        "weighted_brier": weighted_brier,
        "decisions": decisions,
    }

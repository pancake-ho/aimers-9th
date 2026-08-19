from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from src.models import MODEL_ORDER
from src.runtime import apply_logit_intercept, apply_probability_bias, blend_predictions


def brier_score(y_true, prediction) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return float(np.mean((pred - y) ** 2))


def probability_bias(y_true, prediction) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    return float(np.mean(pred - y))


def _integer_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in _integer_compositions(total - first, parts - 1):
            yield (first, *remainder)


def _candidate_weights(step: float = 0.05, model_order: Sequence[str] = MODEL_ORDER):
    n_steps = int(round(1.0 / step))
    if n_steps <= 0 or not np.isclose(n_steps * step, 1.0):
        raise ValueError("Ensemble step must divide one exactly.")
    for composition in _integer_compositions(n_steps, len(model_order)):
        yield np.asarray(composition, dtype=np.float64) / n_steps


def select_stable_weights(
    folds: Sequence[Mapping[str, object]],
    fold_importance: Sequence[float],
    step: float = 0.05,
    model_order: Sequence[str] = MODEL_ORDER,
    protected_fold_labels: Sequence[str] = (),
    non_degradation_tolerance: float = 0.0,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Minimize a regime-aware forward Brier objective.

    The hidden target is 2025 and 2024 is the only holdout from the same ABS
    era. The 2023 fold remains in the objective as a stability guard, while
    2024 receives the larger, predeclared importance.
    """
    importance = np.asarray(fold_importance, dtype=np.float64)
    if importance.shape != (len(folds),):
        raise ValueError("fold_importance must match the number of folds.")
    if (importance < 0).any() or not np.isfinite(importance).all() or importance.sum() <= 0:
        raise ValueError("fold_importance must be finite and non-negative.")
    importance /= importance.sum()
    # For each fold, Brier(Pw, y) is a low-dimensional quadratic form.
    # Precomputing it makes a fine simplex grid independent of row count.
    if non_degradation_tolerance < 0.0:
        raise ValueError("non_degradation_tolerance must be non-negative.")

    quadratic_folds = []
    fold_labels = []
    for fold in folds:
        label = str(fold.get("validation_label", fold.get("valid_season", "")))
        fold_labels.append(label)
        prediction_matrix = np.column_stack(
            [
                np.clip(
                    np.asarray(fold["predictions"][name], dtype=np.float64),
                    1e-6,
                    1.0 - 1e-6,
                )
                for name in model_order
            ]
        )
        target = np.asarray(fold["y_true"], dtype=np.float64)
        if prediction_matrix.shape[0] != target.shape[0]:
            raise ValueError("Fold prediction and target lengths do not match.")
        n_rows = float(len(target))
        quadratic_folds.append(
            (
                prediction_matrix.T @ prediction_matrix / n_rows,
                prediction_matrix.T @ target / n_rows,
                float(target @ target / n_rows),
            )
        )

    protected_indices = []
    for label in protected_fold_labels:
        if label not in fold_labels:
            raise ValueError(f"Protected fold is absent: {label}")
        protected_indices.append(fold_labels.index(label))

    # Pick one stable single-model reference using exactly the predeclared
    # forward objective. A one-hot candidate for this model is always feasible,
    # so the constraints can never force a silently arbitrary blend.
    component_fold_scores = np.empty((len(model_order), len(folds)), dtype=np.float64)
    for model_index in range(len(model_order)):
        one_hot = np.zeros(len(model_order), dtype=np.float64)
        one_hot[model_index] = 1.0
        for fold_index, (gram, linear, constant) in enumerate(quadratic_folds):
            component_fold_scores[model_index, fold_index] = float(
                one_hot @ gram @ one_hot - 2.0 * linear @ one_hot + constant
            )
    component_objectives = component_fold_scores @ importance
    reference_index = int(np.argmin(component_objectives))
    reference_scores = component_fold_scores[reference_index]

    best_weights = None
    best_score = np.inf
    best_fold_scores = None
    evaluated_candidates = 0
    feasible_candidates = 0

    for weights in _candidate_weights(step, model_order=model_order):
        evaluated_candidates += 1
        fold_scores = []
        for gram, linear, constant in quadratic_folds:
            fold_scores.append(
                float(weights @ gram @ weights - 2.0 * linear @ weights + constant)
            )
        fold_scores_array = np.asarray(fold_scores)
        if any(
            fold_scores_array[index]
            > reference_scores[index] + float(non_degradation_tolerance) + 1e-15
            for index in protected_indices
        ):
            continue
        feasible_candidates += 1
        score = float(np.dot(importance, fold_scores_array))
        if score < best_score:
            best_score = score
            best_weights = weights.copy()
            best_fold_scores = fold_scores

    if best_weights is None:
        raise RuntimeError("No ensemble weight candidate was evaluated.")
    report = {
        "model_order": list(model_order),
        "weights": best_weights.tolist(),
        "forward_weighted_brier": best_score,
        "fold_brier": [float(value) for value in best_fold_scores],
        "fold_importance": importance.tolist(),
        "grid_step": float(step),
        "protected_folds": list(protected_fold_labels),
        "non_degradation_tolerance": float(non_degradation_tolerance),
        "reference_model": str(model_order[reference_index]),
        "reference_fold_brier": reference_scores.tolist(),
        "protected_fold_gain_vs_reference": {
            fold_labels[index]: float(reference_scores[index] - best_fold_scores[index])
            for index in protected_indices
        },
        "evaluated_candidates": int(evaluated_candidates),
        "feasible_candidates": int(feasible_candidates),
    }
    return best_weights, report


def fit_prequential_bias_calibrator(
    earlier_fold: Mapping[str, object],
    recent_fold: Mapping[str, object],
    weights: Sequence[float],
    model_order: Sequence[str] = MODEL_ORDER,
    min_gain: float = 1e-7,
) -> Dict[str, object]:
    """Validate an intercept-only Brier correction across adjacent years.

    The earlier holdout estimates a probability-space bias. That fixed bias is
    applied to the next holdout. Only when Brier improves on the next year is
    the method accepted. The final bias is then refit on the most recent
    holdout for one-step-ahead inference.
    """
    earlier_pred = blend_predictions(
        [earlier_fold["predictions"][name] for name in model_order], weights
    )
    recent_pred = blend_predictions(
        [recent_fold["predictions"][name] for name in model_order], weights
    )
    earlier_bias = probability_bias(earlier_fold["y_true"], earlier_pred)
    recent_bias = probability_bias(recent_fold["y_true"], recent_pred)
    recent_raw_brier = brier_score(recent_fold["y_true"], recent_pred)
    transferred = apply_probability_bias(recent_pred, earlier_bias)
    transferred_brier = brier_score(recent_fold["y_true"], transferred)
    same_direction = earlier_bias == 0.0 or recent_bias == 0.0 or np.sign(earlier_bias) == np.sign(recent_bias)
    accepted = bool(same_direction and transferred_brier <= recent_raw_brier - min_gain)

    final_bias = recent_bias if accepted else 0.0
    return {
        "method": "probability_bias",
        "accepted": accepted,
        "bias": float(final_bias),
        "earlier_bias": float(earlier_bias),
        "recent_bias": float(recent_bias),
        "recent_raw_brier": float(recent_raw_brier),
        "recent_transferred_brier": float(transferred_brier),
        "transfer_gain": float(recent_raw_brier - transferred_brier),
    }


def _best_brier_logit_intercept(y_true, prediction) -> float:
    """Deterministic coarse-to-fine one-dimensional Brier optimization."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    center = 0.0
    radius = 0.30
    best = 0.0
    for _ in range(4):
        candidates = np.linspace(center - radius, center + radius, 121)
        scores = np.asarray(
            [brier_score(y, apply_logit_intercept(pred, value)) for value in candidates]
        )
        best = float(candidates[int(np.argmin(scores))])
        center = best
        radius /= 10.0
    return best


def fit_prequential_logit_calibrator(
    earlier_fold: Mapping[str, object],
    recent_fold: Mapping[str, object],
    weights: Sequence[float],
    model_order: Sequence[str] = MODEL_ORDER,
    min_gain: float = 1e-7,
) -> Dict[str, object]:
    """Accept a logit intercept only after year-ahead transfer succeeds.

    The intercept is estimated on the earlier holdout and frozen before being
    evaluated on the next season.  Only a successful, direction-consistent
    transfer allows refitting on the latest full-season holdout for 2025.
    """
    earlier_pred = blend_predictions(
        [earlier_fold["predictions"][name] for name in model_order], weights
    )
    recent_pred = blend_predictions(
        [recent_fold["predictions"][name] for name in model_order], weights
    )
    earlier_intercept = _best_brier_logit_intercept(
        earlier_fold["y_true"], earlier_pred
    )
    recent_intercept = _best_brier_logit_intercept(
        recent_fold["y_true"], recent_pred
    )
    recent_raw_brier = brier_score(recent_fold["y_true"], recent_pred)
    transferred = apply_logit_intercept(recent_pred, earlier_intercept)
    transferred_brier = brier_score(recent_fold["y_true"], transferred)
    same_direction = (
        earlier_intercept == 0.0
        or recent_intercept == 0.0
        or np.sign(earlier_intercept) == np.sign(recent_intercept)
    )
    accepted = bool(
        same_direction and transferred_brier <= recent_raw_brier - min_gain
    )
    return {
        "method": "logit_intercept",
        "accepted": accepted,
        "intercept": float(recent_intercept if accepted else 0.0),
        "earlier_intercept": float(earlier_intercept),
        "recent_intercept": float(recent_intercept),
        "recent_raw_brier": float(recent_raw_brier),
        "recent_transferred_brier": float(transferred_brier),
        "transfer_gain": float(recent_raw_brier - transferred_brier),
    }

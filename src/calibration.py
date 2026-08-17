from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from src.models import MODEL_ORDER
from src.runtime import apply_probability_bias, blend_predictions


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


def _candidate_weights(step: float = 0.05):
    n_steps = int(round(1.0 / step))
    if n_steps <= 0 or not np.isclose(n_steps * step, 1.0):
        raise ValueError("Ensemble step must divide one exactly.")
    for composition in _integer_compositions(n_steps, len(MODEL_ORDER)):
        yield np.asarray(composition, dtype=np.float64) / n_steps


def select_stable_weights(
    folds: Sequence[Mapping[str, object]],
    fold_importance: Sequence[float],
    step: float = 0.05,
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
    best_weights = None
    best_score = np.inf
    best_fold_scores = None

    for weights in _candidate_weights(step):
        fold_scores = []
        for fold in folds:
            prediction = blend_predictions(
                [fold["predictions"][name] for name in MODEL_ORDER],
                weights,
            )
            fold_scores.append(brier_score(fold["y_true"], prediction))
        score = float(np.dot(importance, np.asarray(fold_scores)))
        if score < best_score:
            best_score = score
            best_weights = weights.copy()
            best_fold_scores = fold_scores

    if best_weights is None:
        raise RuntimeError("No ensemble weight candidate was evaluated.")
    report = {
        "model_order": list(MODEL_ORDER),
        "weights": best_weights.tolist(),
        "forward_weighted_brier": best_score,
        "fold_brier": [float(value) for value in best_fold_scores],
        "fold_importance": importance.tolist(),
        "grid_step": float(step),
    }
    return best_weights, report


def fit_prequential_bias_calibrator(
    earlier_fold: Mapping[str, object],
    recent_fold: Mapping[str, object],
    weights: Sequence[float],
    min_gain: float = 1e-7,
) -> Dict[str, object]:
    """Validate an intercept-only Brier correction across adjacent years.

    The earlier holdout estimates a probability-space bias. That fixed bias is
    applied to the next holdout. Only when Brier improves on the next year is
    the method accepted. The final bias is then refit on the most recent
    holdout for one-step-ahead inference.
    """
    earlier_pred = blend_predictions(
        [earlier_fold["predictions"][name] for name in MODEL_ORDER], weights
    )
    recent_pred = blend_predictions(
        [recent_fold["predictions"][name] for name in MODEL_ORDER], weights
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

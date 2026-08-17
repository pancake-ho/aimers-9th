from __future__ import annotations

from dataclasses import dataclass
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


def _candidate_weights(step: float = 0.01):
    if MODEL_ORDER != ("xgb", "cat"):
        raise RuntimeError("Weight grid is defined for the xgb/cat pair.")
    n_steps = int(round(1.0 / step))
    for index in range(n_steps + 1):
        xgb_weight = index / n_steps
        yield np.asarray([xgb_weight, 1.0 - xgb_weight], dtype=np.float64)


def select_stable_weights(
    folds: Sequence[Mapping[str, object]],
    step: float = 0.01,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Minimize the equally weighted mean Brier across temporal folds."""
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
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_weights = weights.copy()
            best_fold_scores = fold_scores

    if best_weights is None:
        raise RuntimeError("No ensemble weight candidate was evaluated.")
    report = {
        "model_order": list(MODEL_ORDER),
        "weights": best_weights.tolist(),
        "mean_temporal_brier": best_score,
        "fold_brier": [float(value) for value in best_fold_scores],
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

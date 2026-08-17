from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def evaluate_probabilities(y_true, prediction) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    result = {
        "brier": float(np.mean((pred - y) ** 2)),
        "logloss": float(log_loss(y, pred)),
        "target_mean": float(y.mean()),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "calibration_bias": float((pred - y).mean()),
    }
    result["auc"] = float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    return result


def brier_skill_score(y_true, prediction, reference_probability: float) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    model_brier = float(np.mean((pred - y) ** 2))
    reference = np.full(len(y), float(reference_probability), dtype=np.float64)
    reference_brier = float(np.mean((reference - y) ** 2))
    return float(1.0 - model_brier / reference_brier)

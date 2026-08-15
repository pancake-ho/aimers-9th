# src/metrics.py

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)


EPS = 1e-6


def clip_probability(
    pred,
) -> np.ndarray:

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    return np.clip(
        pred,
        EPS,
        1.0 - EPS,
    )


def brier_score(
    y_true,
    y_pred,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = clip_probability(
        y_pred
    )

    return float(
        np.mean(
            (
                y_pred
                - y_true
            ) ** 2
        )
    )


def empirical_baseline_brier(
    y_true,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    p = float(
        y_true.mean()
    )

    return float(
        np.mean(
            (
                p - y_true
            ) ** 2
        )
    )


def brier_skill_score(
    y_true,
    y_pred,
) -> float:
    """
    Internal diagnostic BSS using validation-set
    prevalence as reference.

    Official competition score may use a fixed
    organizer baseline, so model selection should
    primarily use raw Brier Score.
    """

    bs = brier_score(
        y_true,
        y_pred,
    )

    reference = (
        empirical_baseline_brier(
            y_true
        )
    )

    if reference <= 0:
        return float("nan")

    return float(
        1.0
        - bs / reference
    )


def evaluate_probabilities(
    y_true,
    y_pred,
) -> Dict[str, float]:

    y_true = np.asarray(
        y_true,
    )

    pred = clip_probability(
        y_pred
    )

    result = {
        "brier": (
            brier_score(
                y_true,
                pred,
            )
        ),
        "brier_skill_empirical": (
            brier_skill_score(
                y_true,
                pred,
            )
        ),
        "logloss": float(
            log_loss(
                y_true,
                pred,
            )
        ),
        "pred_mean": float(
            pred.mean()
        ),
        "pred_std": float(
            pred.std()
        ),
        "target_mean": float(
            y_true.mean()
        ),
        "calibration_bias": float(
            pred.mean()
            - y_true.mean()
        ),
    }

    if len(
        np.unique(
            y_true
        )
    ) == 2:
        result["auc"] = float(
            roc_auc_score(
                y_true,
                pred,
            )
        )
    else:
        result["auc"] = float(
            "nan"
        )

    return result
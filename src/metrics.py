from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)


EPS = 1e-7


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
            (y_pred - y_true) ** 2
        )
    )


def constant_brier_score(
    y_true,
    probability: float,
) -> float:
    pred = np.full(
        len(y_true),
        float(probability),
        dtype=np.float64,
    )

    return brier_score(
        y_true,
        pred,
    )


def evaluate_probabilities(
    y_true,
    y_pred,
) -> Dict[str, float]:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    pred = clip_probability(
        y_pred
    )

    result = {
        "brier": brier_score(
            y_true,
            pred,
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

    if len(np.unique(y_true)) == 2:
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


def evaluate_constant_baselines(
    y_train,
    y_valid,
    train_seasons=None,
) -> Dict[str, float]:

    y_train = np.asarray(
        y_train,
        dtype=np.float64,
    )

    y_valid = np.asarray(
        y_valid,
        dtype=np.float64,
    )

    train_mean = float(
        y_train.mean()
    )

    result = {
        "train_mean": train_mean,
        "train_mean_brier": (
            constant_brier_score(
                y_valid,
                train_mean,
            )
        ),
    }

    if train_seasons is not None:
        train_seasons = np.asarray(
            train_seasons
        )

        latest_season = int(
            train_seasons.max()
        )

        latest_mask = (
            train_seasons
            == latest_season
        )

        latest_mean = float(
            y_train[
                latest_mask
            ].mean()
        )

        result[
            "latest_train_season"
        ] = latest_season

        result[
            "latest_season_mean"
        ] = latest_mean

        result[
            "latest_season_mean_brier"
        ] = constant_brier_score(
            y_valid,
            latest_mean,
        )

    return result
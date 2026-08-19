from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.models import _xgb_brier_metric, _xgb_params, xgb


BASELINE_COUNT_COL = "asof_pitcher_n"
BASELINE_RATE_COL = "asof_pitcher_success_rate"


@dataclass(frozen=True)
class PitcherBaselineConfig:
    """Leakage-safe empirical-Bayes pitcher probability.

    ``asof_pitcher_*`` fields are official pre-pitch inputs.  No statistic is
    recomputed from validation/test rows.  The fixed prior deliberately avoids
    copying the strong 2019--2024 league-level target drift into a future row.
    """

    prior_probability: float = 0.5
    prior_strength: float = 100.0
    probability_clip: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < float(self.prior_probability) < 1.0:
            raise ValueError("prior_probability must be strictly between 0 and 1.")
        if float(self.prior_strength) <= 0.0:
            raise ValueError("prior_strength must be positive.")
        if not 0.0 < float(self.probability_clip) < 0.5:
            raise ValueError("probability_clip must be between 0 and 0.5.")


def pitcher_baseline_probability(
    frame: pd.DataFrame,
    config: PitcherBaselineConfig,
) -> np.ndarray:
    """Return the row-wise, strictly-as-of pitcher baseline probability."""
    missing = {BASELINE_COUNT_COL, BASELINE_RATE_COL} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing pitcher baseline columns: {sorted(missing)}")

    prior = float(config.prior_probability)
    strength = float(config.prior_strength)
    count = (
        pd.to_numeric(frame[BASELINE_COUNT_COL], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64, copy=False)
    )
    rate = (
        pd.to_numeric(frame[BASELINE_RATE_COL], errors="coerce")
        .fillna(prior)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype=np.float64, copy=False)
    )
    probability = (count * rate + strength * prior) / (count + strength)
    probability = np.clip(
        probability,
        float(config.probability_clip),
        1.0 - float(config.probability_clip),
    )
    if probability.shape != (len(frame),) or not np.isfinite(probability).all():
        raise RuntimeError("Pitcher baseline produced invalid probabilities.")
    return probability.astype(np.float32, copy=False)


def probability_to_margin(probability: np.ndarray, clip: float = 1e-6) -> np.ndarray:
    """Convert probability to the raw log-odds required by XGBoost base_margin."""
    values = np.asarray(probability, dtype=np.float64)
    values = np.clip(values, float(clip), 1.0 - float(clip))
    margin = np.log(values) - np.log1p(-values)
    if not np.isfinite(margin).all():
        raise RuntimeError("Non-finite pitcher baseline margin.")
    return margin.astype(np.float32, copy=False)


def margin_to_probability(margin: np.ndarray) -> np.ndarray:
    values = np.asarray(margin, dtype=np.float64)
    output = np.empty_like(values)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_value = np.exp(values[~nonnegative])
    output[~nonnegative] = exp_value / (1.0 + exp_value)
    return output


# The strict residual model must not relearn the same pitcher intercept that is
# already supplied through base_margin.  Confidence/form-difference features
# remain available, but absolute pitcher-success and pitcher-identity signals
# are excluded.  The full residual ablation intentionally keeps every feature.
STRICT_EXACT_EXCLUSIONS = frozenset(
    {
        "pitcher_id",
        "pitcher_count_combo",
        "pitcher_base_combo",
        "recent_success_blend",
        "count_success_interaction",
        "abs_history_interaction",
    }
)

STRICT_PREFIX_EXCLUSIONS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "cross_season_success_",
    "hist_pitcher_effect",
    "pitcher_count_adjusted_success",
    "pitcher_matchup_adjusted_success",
    "pitcher_count_matchup_adjusted_success",
)


def strict_residual_feature_names(columns: Iterable[str]) -> list[str]:
    selected = []
    for raw_name in columns:
        name = str(raw_name)
        if name in STRICT_EXACT_EXCLUSIONS:
            continue
        if any(name.startswith(prefix) for prefix in STRICT_PREFIX_EXCLUSIONS):
            continue
        selected.append(name)
    if not selected:
        raise ValueError("Strict residual feature selection removed every column.")
    return selected


def select_strict_residual_features(frame: pd.DataFrame) -> pd.DataFrame:
    selected = strict_residual_feature_names(frame.columns)
    return frame.loc[:, selected]


def _residual_quantile_dmatrix(
    frame: pd.DataFrame,
    config: ModelConfig,
    *,
    base_margin: np.ndarray,
    label=None,
    weight=None,
    ref=None,
):
    if xgb is None:
        raise ImportError("xgboost is required for the residual GBDT experiment.")
    margin = np.asarray(base_margin, dtype=np.float32)
    if margin.shape != (len(frame),) or not np.isfinite(margin).all():
        raise ValueError("base_margin must be one finite value per row.")
    return xgb.QuantileDMatrix(
        frame,
        label=label,
        weight=weight,
        base_margin=margin,
        feature_names=list(frame.columns),
        ref=ref,
        max_bin=int(config.xgb_max_bin),
        nthread=int(config.num_threads),
    )


def train_residual_xgboost_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_base_margin: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    valid_base_margin: np.ndarray,
    sample_weight: np.ndarray,
    config: ModelConfig,
):
    """Fit logistic boosting around a row-wise pitcher logit offset.

    The target remains the original binary label.  This is not regression on
    ``y - p0``; XGBoost computes logistic gradients around ``base_margin``.
    Validation prediction receives the matching row-wise offset as required.
    """
    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError("Residual train/validation feature columns must match.")
    dtrain = _residual_quantile_dmatrix(
        X_train,
        config,
        label=np.asarray(y_train, dtype=np.float32),
        weight=np.asarray(sample_weight, dtype=np.float32),
        base_margin=train_base_margin,
    )
    dvalid = _residual_quantile_dmatrix(
        X_valid,
        config,
        label=np.asarray(y_valid, dtype=np.float32),
        base_margin=valid_base_margin,
        ref=dtrain,
    )
    model = xgb.train(
        params=_xgb_params(config),
        dtrain=dtrain,
        num_boost_round=int(config.xgb_num_boost_round),
        evals=[(dvalid, "valid")],
        custom_metric=_xgb_brier_metric,
        maximize=False,
        early_stopping_rounds=int(config.xgb_early_stopping_rounds),
        verbose_eval=50,
    )
    best_iteration = (
        int(model.best_iteration)
        if model.best_iteration is not None
        else int(config.xgb_num_boost_round) - 1
    )
    prediction = np.asarray(
        model.predict(dvalid, iteration_range=(0, best_iteration + 1)),
        dtype=np.float64,
    )
    if prediction.shape != (len(X_valid),) or not np.isfinite(prediction).all():
        raise RuntimeError("Residual XGBoost returned invalid probabilities.")
    del dtrain, dvalid
    gc.collect()
    return model, prediction, best_iteration + 1


def validate_residual_xgboost_backend(config: ModelConfig) -> None:
    """Exercise the exact QuantileDMatrix + base_margin path before data load."""
    frame = pd.DataFrame(
        {
            "count_pressure": np.linspace(0.0, 1.0, 12, dtype=np.float32),
            "is_high_leverage": np.asarray([0, 1] * 6, dtype=np.float32),
        }
    )
    target = np.asarray([0, 0, 1, 0, 1, 1] * 2, dtype=np.float32)
    margin = probability_to_margin(
        np.asarray([0.45, 0.55, 0.50, 0.60, 0.40, 0.65] * 2)
    )
    probe_config = ModelConfig(
        random_seed=config.random_seed,
        num_threads=min(2, int(config.num_threads)),
        xgb_device=config.xgb_device,
        xgb_num_boost_round=2,
        xgb_early_stopping_rounds=1,
        xgb_max_bin=config.xgb_max_bin,
        xgb_min_child_weight=1.0,
    )
    model, prediction, _ = train_residual_xgboost_fold(
        frame.iloc[:8],
        target[:8],
        margin[:8],
        frame.iloc[8:],
        target[8:],
        margin[8:],
        np.ones(8, dtype=np.float32),
        probe_config,
    )
    if not np.logical_and(prediction > 0.0, prediction < 1.0).all():
        raise RuntimeError("Residual XGBoost backend produced invalid probabilities.")
    del model
    gc.collect()
    print(
        "[BACKEND] Residual XGBoost base_margin PASS "
        f"(device={config.xgb_device.lower()})"
    )

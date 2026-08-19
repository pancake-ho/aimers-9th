"""Controlled baseline experiments that do not alter the submitted pipeline."""

from src.baseline.pitcher_residual import (
    PitcherBaselineConfig,
    margin_to_probability,
    pitcher_baseline_probability,
    probability_to_margin,
    select_strict_residual_features,
    train_residual_xgboost_fold,
    validate_residual_xgboost_backend,
)

__all__ = [
    "PitcherBaselineConfig",
    "margin_to_probability",
    "pitcher_baseline_probability",
    "probability_to_margin",
    "select_strict_residual_features",
    "train_residual_xgboost_fold",
    "validate_residual_xgboost_backend",
]

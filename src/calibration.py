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

def fit_abs_regime_logit_calibrator(
    full_2024_fold: Mapping[str, object],
    late_2024_fold: Mapping[str, object],
    weights: Sequence[float],
    model_order: Sequence[str] = MODEL_ORDER,
    early_month_max: int = 6,
    min_gain: float = 1e-7,
) -> Dict[str, object]:
    """
    ABS-regime calibration validation.

    1) A model trained through 2023 predicts all 2024 rows.
    2) Jan-Jun 2024 predictions/labels estimate an intercept.
    3) That frozen intercept is applied to Jul-Sep predictions from the
       independent late-2024 temporal fold.
    4) Only if the forward transfer improves Brier do we accept calibration.
    5) For final 2025 inference, the intercept is refit on all 2024 holdout
       predictions.

    No test-row statistic is used.
    """

    full_prediction = blend_predictions(
        [
            full_2024_fold["predictions"][name]
            for name in model_order
        ],
        weights,
    )

    late_prediction = blend_predictions(
        [
            late_2024_fold["predictions"][name]
            for name in model_order
        ],
        weights,
    )

    full_y = np.asarray(
        full_2024_fold["y_true"],
        dtype=np.float64,
    )
    late_y = np.asarray(
        late_2024_fold["y_true"],
        dtype=np.float64,
    )

    months = np.asarray(
        full_2024_fold["valid_months"],
        dtype=np.int64,
    )

    if months.shape != full_y.shape:
        raise ValueError(
            "2024 validation month vector does not match targets."
        )

    early_mask = months <= int(early_month_max)

    if early_mask.sum() < 1000:
        raise ValueError(
            "Not enough early-2024 rows for calibration transfer."
        )

    early_y = full_y[early_mask]
    early_prediction = full_prediction[early_mask]

    early_intercept = _best_brier_logit_intercept(
        early_y,
        early_prediction,
    )

    full_intercept = _best_brier_logit_intercept(
        full_y,
        full_prediction,
    )

    late_raw_brier = brier_score(
        late_y,
        late_prediction,
    )

    late_transferred_prediction = (
        apply_logit_intercept(
            late_prediction,
            early_intercept,
        )
    )

    late_transferred_brier = brier_score(
        late_y,
        late_transferred_prediction,
    )

    full_raw_brier = brier_score(
        full_y,
        full_prediction,
    )

    full_refit_brier = brier_score(
        full_y,
        apply_logit_intercept(
            full_prediction,
            full_intercept,
        ),
    )

    same_direction = (
        early_intercept == 0.0
        or full_intercept == 0.0
        or np.sign(early_intercept)
        == np.sign(full_intercept)
    )

    accepted = bool(
        same_direction
        and late_transferred_brier
        <= late_raw_brier - float(min_gain)
    )

    return {
        "method": "abs_regime_logit_intercept",
        "accepted": accepted,
        "intercept": float(
            full_intercept if accepted else 0.0
        ),
        "early_intercept": float(
            early_intercept
        ),
        "full_2024_intercept": float(
            full_intercept
        ),
        "early_rows": int(
            early_mask.sum()
        ),
        "full_2024_raw_brier": float(
            full_raw_brier
        ),
        "full_2024_refit_brier": float(
            full_refit_brier
        ),
        "late_raw_brier": float(
            late_raw_brier
        ),
        "late_transferred_brier": float(
            late_transferred_brier
        ),
        "transfer_gain": float(
            late_raw_brier
            - late_transferred_brier
        ),
    }

def _fold_brier_with_weights(
    fold: Mapping[str, object],
    weights: Sequence[float],
    model_order: Sequence[str],
) -> float:
    prediction = blend_predictions(
        [
            fold["predictions"][name]
            for name in model_order
        ],
        weights,
    )

    return brier_score(
        fold["y_true"],
        prediction,
    )


def select_calibration_aware_shrunk_weights(
    folds: Sequence[
        Mapping[str, object]
    ],
    *,
    fold_importance: Sequence[float],
    step: float,
    model_order: Sequence[str],
    protected_fold_labels: Sequence[str],
    non_degradation_tolerance: float,
    reference_model: str,
    alpha_grid: Sequence[float],
    early_month_max: int,
    maximum_2023_regression_vs_reference: float,
    minimum_calibration_transfer_gain: float,
) -> tuple[
    np.ndarray,
    Dict[str, object],
    Dict[str, object],
]:
    """Select a temporally robust calibrated ensemble.

    Procedure
    ---------
    1. Find the normal raw-Brier optimal ensemble on the
       temporal folds with the existing protected-fold gate.

    2. Construct a stable prior consisting of the reference
       model only (XGBoost in V10).

    3. Shrink the raw-optimal weights toward that prior:

           w(alpha)
             = (1-alpha) * w_reference
               + alpha * w_raw_opt

       alpha=0 gives the stable reference.
       alpha=1 recovers the V9 raw-optimal ensemble.

    4. For each alpha, fit the existing ABS-regime
       calibrator:
         early 2024 -> late 2024 transfer.

    5. Select alpha using the deployed probability pipeline:
         2023 raw Brier
         + 2024 raw Brier
         + late-2024 forward-calibrated Brier.

    No test-row statistic is used.
    """

    if len(folds) != 3:
        raise ValueError(
            "Calibration-aware shrinkage "
            "requires exactly three temporal folds."
        )

    labels = [
        str(
            fold.get(
                "validation_label",
                fold.get(
                    "valid_season",
                    "",
                ),
            )
        )
        for fold in folds
    ]

    required_labels = {
        "2023",
        "2024",
        "2024_late_abs",
    }

    if set(labels) != required_labels:
        raise ValueError(
            "Unexpected temporal folds for "
            "calibration-aware shrinkage: "
            f"{labels}"
        )

    if reference_model not in model_order:
        raise ValueError(
            "Shrinkage reference model is absent "
            f"from model_order: {reference_model}"
        )

    alpha_values = tuple(
        float(value)
        for value in alpha_grid
    )

    if not alpha_values:
        raise ValueError(
            "alpha_grid must not be empty."
        )

    if any(
        (
            not np.isfinite(value)
            or value < 0.0
            or value > 1.0
        )
        for value in alpha_values
    ):
        raise ValueError(
            "Every shrinkage alpha must "
            "lie in [0, 1]."
        )

    if len(set(alpha_values)) != len(
        alpha_values
    ):
        raise ValueError(
            "Duplicate shrinkage alpha."
        )

    importance = np.asarray(
        fold_importance,
        dtype=np.float64,
    )

    if importance.shape != (
        len(folds),
    ):
        raise ValueError(
            "fold_importance must match folds."
        )

    if (
        not np.isfinite(
            importance
        ).all()
        or (
            importance < 0.0
        ).any()
        or importance.sum() <= 0.0
    ):
        raise ValueError(
            "Invalid fold importance."
        )

    importance = (
        importance
        / importance.sum()
    )

    by_label = {
        label: folds[index]
        for index, label
        in enumerate(labels)
    }

    # --------------------------------------------------------
    # Raw temporal optimum.
    # --------------------------------------------------------

    (
        raw_optimal_weights,
        raw_weight_report,
    ) = select_stable_weights(
        folds,
        fold_importance=importance,
        step=step,
        model_order=model_order,
        protected_fold_labels=(
            protected_fold_labels
        ),
        non_degradation_tolerance=(
            non_degradation_tolerance
        ),
    )

    # --------------------------------------------------------
    # Stable prior = XGBoost one-hot.
    # --------------------------------------------------------

    reference_weights = np.zeros(
        len(model_order),
        dtype=np.float64,
    )

    reference_index = list(
        model_order
    ).index(
        reference_model
    )

    reference_weights[
        reference_index
    ] = 1.0

    reference_fold_brier = {
        label: _fold_brier_with_weights(
            fold,
            reference_weights,
            model_order,
        )
        for label, fold
        in by_label.items()
    }

    candidate_payloads = []

    for alpha in alpha_values:
        weights = (
            (
                1.0 - alpha
            )
            * reference_weights
            + alpha
            * raw_optimal_weights
        )

        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        if not np.isclose(
            weights.sum(),
            1.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Shrunk ensemble weights "
                "do not sum to one."
            )

        fold_brier = {
            label: _fold_brier_with_weights(
                fold,
                weights,
                model_order,
            )
            for label, fold
            in by_label.items()
        }

        calibrator = (
            fit_abs_regime_logit_calibrator(
                full_2024_fold=(
                    by_label["2024"]
                ),
                late_2024_fold=(
                    by_label[
                        "2024_late_abs"
                    ]
                ),
                weights=weights,
                model_order=model_order,
                early_month_max=(
                    early_month_max
                ),
            )
        )

        if calibrator["accepted"]:
            deployed_late_brier = float(
                calibrator[
                    "late_transferred_brier"
                ]
            )
        else:
            deployed_late_brier = float(
                fold_brier[
                    "2024_late_abs"
                ]
            )

        objective = float(
            importance[
                labels.index("2023")
            ]
            * fold_brier["2023"]
            + importance[
                labels.index("2024")
            ]
            * fold_brier["2024"]
            + importance[
                labels.index(
                    "2024_late_abs"
                )
            ]
            * deployed_late_brier
        )

        gain_2024_vs_reference = float(
            reference_fold_brier[
                "2024"
            ]
            - fold_brier["2024"]
        )

        gain_late_raw_vs_reference = float(
            reference_fold_brier[
                "2024_late_abs"
            ]
            - fold_brier[
                "2024_late_abs"
            ]
        )

        gain_late_deployed_vs_reference = float(
            reference_fold_brier[
                "2024_late_abs"
            ]
            - deployed_late_brier
        )

        reference_2023_brier = float(
            reference_fold_brier["2023"]
        )

        regression_2023_vs_reference = float(
            fold_brier["2023"]
            - reference_2023_brier
        )

        feasible = bool(
            regression_2023_vs_reference
            <= float(
                maximum_2023_regression_vs_reference
            )
            + 1.0e-15

            and gain_2024_vs_reference
            + float(
                non_degradation_tolerance
            )
            >= -1.0e-15

            and gain_late_raw_vs_reference
            + float(
                non_degradation_tolerance
            )
            >= -1.0e-15

            and bool(
                calibrator["accepted"]
            )

            and float(
                calibrator["transfer_gain"]
            )
            + 1.0e-15
            >= float(
                minimum_calibration_transfer_gain
            )
        )

        record = {
            "alpha": float(alpha),
            "weights": weights.tolist(),
            "fold_brier": {
                key: float(value)
                for key, value
                in fold_brier.items()
            },
            "deployed_late_brier": float(
                deployed_late_brier
            ),
            "forward_weighted_brier": float(
                objective
            ),
            "2024_gain_vs_reference": float(
                gain_2024_vs_reference
            ),
            "late_raw_gain_vs_reference": float(
                gain_late_raw_vs_reference
            ),
            "late_deployed_gain_vs_reference": float(
                gain_late_deployed_vs_reference
            ),
            "calibration_accepted": bool(
                calibrator["accepted"]
            ),
            "calibration_transfer_gain": float(
                calibrator[
                    "transfer_gain"
                ]
            ),
            "feasible": bool(
                feasible
            ),
            "2023_regression_vs_reference": float(
                regression_2023_vs_reference
            ),
        }

        candidate_payloads.append(
            (
                record,
                weights,
                calibrator,
            )
        )

    feasible_payloads = [
        payload
        for payload
        in candidate_payloads
        if payload[0]["feasible"]
    ]

    selection_passed = bool(
        feasible_payloads
    )

    selection_pool = (
        feasible_payloads
        if feasible_payloads
        else candidate_payloads
    )

    # On an effectively identical objective,
    # prefer stronger shrinkage / smaller alpha.
    selected_record, selected_weights, (
        selected_calibrator
    ) = min(
        selection_pool,
        key=lambda payload: (
            payload[0][
                "forward_weighted_brier"
            ],
            payload[0]["alpha"],
        ),
    )

    report = {
        "method": (
            "calibration_aware_"
            "reference_shrinkage"
        ),
        "model_order": list(
            model_order
        ),
        "selection_passed": bool(
            selection_passed
        ),
        "reference_model": str(
            reference_model
        ),
        "reference_weights": (
            reference_weights.tolist()
        ),
        "reference_fold_brier": {
            key: float(value)
            for key, value
            in reference_fold_brier.items()
        },
        "raw_optimal_weights": (
            raw_optimal_weights.tolist()
        ),
        "raw_weight_selection": (
            raw_weight_report
        ),
        "alpha_grid": list(
            alpha_values
        ),
        "selected_alpha": float(
            selected_record["alpha"]
        ),
        "weights": (
            selected_weights.tolist()
        ),
        "fold_brier": dict(
            selected_record[
                "fold_brier"
            ]
        ),
        "deployed_late_brier": float(
            selected_record[
                "deployed_late_brier"
            ]
        ),
        "forward_weighted_brier": float(
            selected_record[
                "forward_weighted_brier"
            ]
        ),
        "candidates": [
            payload[0]
            for payload
            in candidate_payloads
        ],
    }

    return (
        np.asarray(
            selected_weights,
            dtype=np.float64,
        ),
        report,
        selected_calibrator,
    )


def evaluate_fixed_calibrated_weights(
    folds: Sequence[
        Mapping[str, object]
    ],
    *,
    weights: Sequence[float],
    fold_importance: Sequence[float],
    model_order: Sequence[str],
    early_month_max: int,
) -> tuple[
    np.ndarray,
    Dict[str, object],
    Dict[str, object],
]:
    """Evaluate a frozen outer ensemble with ABS calibration.

    The weight vector is supplied externally and is not
    optimized on the current folds.

    This is used for the V11b ablation:
        V10 outer weights
        + XGBoost 3-seed bagging.

    No test-row information is used.
    """

    if len(folds) != 3:
        raise ValueError(
            "Fixed calibrated ensemble requires "
            "exactly three temporal folds."
        )

    labels = [
        str(
            fold.get(
                "validation_label",
                fold.get(
                    "valid_season",
                    "",
                ),
            )
        )
        for fold in folds
    ]

    required_labels = {
        "2023",
        "2024",
        "2024_late_abs",
    }

    if set(labels) != required_labels:
        raise ValueError(
            "Unexpected temporal folds: "
            f"{labels}"
        )

    weight_array = np.asarray(
        weights,
        dtype=np.float64,
    )

    if weight_array.shape != (
        len(model_order),
    ):
        raise ValueError(
            "Fixed ensemble weight count "
            "does not match model_order."
        )

    if not np.isfinite(
        weight_array
    ).all():
        raise ValueError(
            "Fixed ensemble weights must "
            "be finite."
        )

    if (
        weight_array < 0.0
    ).any():
        raise ValueError(
            "Fixed ensemble weights must "
            "be non-negative."
        )

    if not np.isclose(
        weight_array.sum(),
        1.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Fixed ensemble weights must "
            "sum to one."
        )

    importance = np.asarray(
        fold_importance,
        dtype=np.float64,
    )

    if importance.shape != (
        len(folds),
    ):
        raise ValueError(
            "fold_importance must "
            "match folds."
        )

    if (
        not np.isfinite(
            importance
        ).all()
        or (
            importance < 0.0
        ).any()
        or importance.sum() <= 0.0
    ):
        raise ValueError(
            "Invalid fold importance."
        )

    importance = (
        importance
        / importance.sum()
    )

    by_label = {
        label: folds[index]
        for index, label
        in enumerate(labels)
    }

    fold_brier = {
        label: _fold_brier_with_weights(
            fold,
            weight_array,
            model_order,
        )
        for label, fold
        in by_label.items()
    }

    calibrator = (
        fit_abs_regime_logit_calibrator(
            full_2024_fold=(
                by_label["2024"]
            ),
            late_2024_fold=(
                by_label[
                    "2024_late_abs"
                ]
            ),
            weights=weight_array,
            model_order=model_order,
            early_month_max=(
                early_month_max
            ),
        )
    )

    if calibrator["accepted"]:
        deployed_late_brier = float(
            calibrator[
                "late_transferred_brier"
            ]
        )
    else:
        deployed_late_brier = float(
            fold_brier[
                "2024_late_abs"
            ]
        )

    objective = float(
        importance[
            labels.index("2023")
        ]
        * fold_brier["2023"]
        + importance[
            labels.index("2024")
        ]
        * fold_brier["2024"]
        + importance[
            labels.index(
                "2024_late_abs"
            )
        ]
        * deployed_late_brier
    )

    report = {
        "method": (
            "fixed_v10_champion_weights"
        ),

        # Structural validity only.
        # Actual Brier gates remain in training.py.
        "selection_passed": True,

        "model_order": list(
            model_order
        ),

        "weights": (
            weight_array.tolist()
        ),

        "fold_importance": (
            importance.tolist()
        ),

        "fold_brier": {
            key: float(value)
            for key, value
            in fold_brier.items()
        },

        "deployed_late_brier": float(
            deployed_late_brier
        ),

        "forward_weighted_brier": float(
            objective
        ),

        "calibration_accepted": bool(
            calibrator["accepted"]
        ),

        "calibration_transfer_gain": float(
            calibrator[
                "transfer_gain"
            ]
        ),
    }

    return (
        weight_array.copy(),
        report,
        calibrator,
    )
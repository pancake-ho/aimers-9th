from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


TEMPORAL_VIEW_ORDER = (
    "base",
    "recent1",
    "recent2",
)


def recent_window_positions(
    seasons: Sequence[int],
    *,
    n_seasons: int,
) -> tuple[
    np.ndarray,
    dict[str, object],
]:
    """Select the latest N available training seasons.

    The function receives training-side seasons only.
    It never inspects validation/test rows.
    """

    values = np.asarray(
        seasons
    )

    if values.ndim != 1:
        raise ValueError(
            "seasons must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "Cannot build a recent window "
            "from an empty training fold."
        )

    numeric = values.astype(
        np.int64,
        copy=False,
    )

    unique = np.unique(
        numeric
    )

    unique.sort()

    n_seasons = int(
        n_seasons
    )

    if n_seasons <= 0:
        raise ValueError(
            "n_seasons must be positive."
        )

    selected = unique[
        -min(
            n_seasons,
            len(unique),
        ):
    ]

    mask = np.isin(
        numeric,
        selected,
    )

    positions = np.flatnonzero(
        mask
    )

    if len(positions) == 0:
        raise RuntimeError(
            "Recent-window selection "
            "produced zero rows."
        )

    return (
        positions,
        {
            "available_seasons": (
                unique.tolist()
            ),
            "selected_seasons": (
                selected.tolist()
            ),
            "n_rows": int(
                len(positions)
            ),
            "row_fraction": float(
                len(positions)
                / len(values)
            ),
        },
    )


def blend_temporal_views(
    views: Mapping[
        str,
        np.ndarray,
    ],
    weights: Sequence[float],
) -> np.ndarray:
    weight = np.asarray(
        weights,
        dtype=np.float64,
    )

    if weight.shape != (
        len(TEMPORAL_VIEW_ORDER),
    ):
        raise ValueError(
            "Temporal-view weight "
            "length mismatch."
        )

    if (
        not np.isfinite(
            weight
        ).all()
        or (
            weight < 0.0
        ).any()
        or weight.sum() <= 0.0
    ):
        raise ValueError(
            "Invalid temporal-view weights."
        )

    weight /= weight.sum()

    predictions = []

    expected_shape = None

    for name in TEMPORAL_VIEW_ORDER:
        if name not in views:
            raise KeyError(
                f"Missing temporal view: {name}"
            )

        prediction = np.asarray(
            views[name],
            dtype=np.float64,
        )

        if prediction.ndim != 1:
            raise ValueError(
                "Temporal-view prediction "
                "must be one-dimensional."
            )

        if expected_shape is None:
            expected_shape = (
                prediction.shape
            )
        elif prediction.shape != expected_shape:
            raise ValueError(
                "Temporal-view prediction "
                "shape mismatch."
            )

        predictions.append(
            prediction
        )

    matrix = np.column_stack(
        predictions
    )

    output = (
        matrix @ weight
    )

    return np.clip(
        output,
        1.0e-6,
        1.0 - 1.0e-6,
    )


def select_temporal_view_weights(
    folds: Sequence[
        Mapping[str, object]
    ],
    *,
    fold_importance: Sequence[float],
    grid_step: float,
    minimum_base_weight: float,
    maximum_aux_weight: float,
    protected_labels: Sequence[str],
    protected_tolerance: float,
) -> tuple[
    np.ndarray,
    dict[str, object],
]:
    """Select a constrained temporal-view blend by Brier."""

    importance = np.asarray(
        fold_importance,
        dtype=np.float64,
    )

    if importance.shape != (
        len(folds),
    ):
        raise ValueError(
            "fold_importance shape mismatch."
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

    importance /= (
        importance.sum()
    )

    labels = [
        str(
            fold[
                "validation_label"
            ]
        )
        for fold in folds
    ]

    protected_indices = []

    for label in protected_labels:
        if label not in labels:
            raise ValueError(
                "Protected temporal fold "
                f"is missing: {label}"
            )

        protected_indices.append(
            labels.index(
                label
            )
        )

    base_scores = []

    for fold in folds:
        y = np.asarray(
            fold["y_true"],
            dtype=np.float64,
        )

        base = np.asarray(
            fold[
                "xgb_temporal_views"
            ]["base"],
            dtype=np.float64,
        )

        base_scores.append(
            float(
                np.mean(
                    (
                        base - y
                    ) ** 2
                )
            )
        )

    best_weight = np.asarray(
        [
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    best_scores = list(
        base_scores
    )

    best_objective = float(
        np.dot(
            importance,
            np.asarray(
                base_scores
            ),
        )
    )

    step = float(
        grid_step
    )

    n_steps = int(
        round(
            1.0 / step
        )
    )

    if not np.isclose(
        n_steps * step,
        1.0,
    ):
        raise ValueError(
            "grid_step must divide one."
        )

    evaluated = 0
    feasible = 0

    for base_step in range(
        n_steps + 1
    ):
        for recent1_step in range(
            n_steps - base_step + 1
        ):
            recent2_step = (
                n_steps
                - base_step
                - recent1_step
            )

            weight = (
                np.asarray(
                    [
                        base_step,
                        recent1_step,
                        recent2_step,
                    ],
                    dtype=np.float64,
                )
                / n_steps
            )

            evaluated += 1

            if (
                weight[0]
                < minimum_base_weight
                - 1.0e-15
            ):
                continue

            if (
                weight[1]
                > maximum_aux_weight
                + 1.0e-15
                or weight[2]
                > maximum_aux_weight
                + 1.0e-15
            ):
                continue

            fold_scores = []

            for fold in folds:
                prediction = (
                    blend_temporal_views(
                        fold[
                            "xgb_temporal_views"
                        ],
                        weight,
                    )
                )

                y = np.asarray(
                    fold["y_true"],
                    dtype=np.float64,
                )

                fold_scores.append(
                    float(
                        np.mean(
                            (
                                prediction - y
                            ) ** 2
                        )
                    )
                )

            violates = any(
                fold_scores[index]
                > (
                    base_scores[index]
                    + protected_tolerance
                    + 1.0e-15
                )
                for index
                in protected_indices
            )

            if violates:
                continue

            feasible += 1

            objective = float(
                np.dot(
                    importance,
                    np.asarray(
                        fold_scores
                    ),
                )
            )

            if (
                objective
                < best_objective
                - 1.0e-15
            ):
                best_objective = (
                    objective
                )

                best_weight = (
                    weight.copy()
                )

                best_scores = list(
                    fold_scores
                )

            elif np.isclose(
                objective,
                best_objective,
                atol=1.0e-15,
            ):
                # Prefer the stable global model
                # on exact ties.
                if weight[0] > best_weight[0]:
                    best_weight = (
                        weight.copy()
                    )

                    best_scores = list(
                        fold_scores
                    )

    return (
        best_weight,
        {
            "view_order": list(
                TEMPORAL_VIEW_ORDER
            ),

            "weights": (
                best_weight.tolist()
            ),

            "fold_brier": {
                labels[index]: float(
                    best_scores[index]
                )
                for index
                in range(
                    len(labels)
                )
            },

            "base_fold_brier": {
                labels[index]: float(
                    base_scores[index]
                )
                for index
                in range(
                    len(labels)
                )
            },

            "gain_vs_base": {
                labels[index]: float(
                    base_scores[index]
                    - best_scores[index]
                )
                for index
                in range(
                    len(labels)
                )
            },

            "forward_weighted_brier": float(
                best_objective
            ),

            "evaluated_candidates": int(
                evaluated
            ),

            "feasible_candidates": int(
                feasible
            ),

            "grid_step": float(
                grid_step
            ),

            "minimum_base_weight": float(
                minimum_base_weight
            ),

            "maximum_aux_weight": float(
                maximum_aux_weight
            ),

            "protected_tolerance": float(
                protected_tolerance
            ),
        },
    )
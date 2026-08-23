from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def _brier(
    y_true,
    prediction,
) -> float:
    y = np.asarray(
        y_true,
        dtype=np.float64,
    )

    pred = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if y.shape != pred.shape:
        raise ValueError(
            "Brier target/prediction "
            "shape mismatch."
        )

    if (
        not np.isfinite(y).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError(
            "Brier input contains "
            "NaN or infinity."
        )

    pred = np.clip(
        pred,
        1.0e-6,
        1.0 - 1.0e-6,
    )

    return float(
        np.mean(
            (
                pred
                - y
            )
            ** 2
        )
    )


def blend_xgb_native_cat(
    base_prediction,
    native_prediction,
    native_weight: float,
) -> np.ndarray:
    weight = float(
        native_weight
    )

    if (
        not np.isfinite(weight)
        or weight < 0.0
        or weight > 1.0
    ):
        raise ValueError(
            "Native categorical XGB "
            "weight must lie in [0, 1]."
        )

    base = np.asarray(
        base_prediction,
        dtype=np.float64,
    )

    native = np.asarray(
        native_prediction,
        dtype=np.float64,
    )

    if base.shape != native.shape:
        raise ValueError(
            "Native/base XGB prediction "
            "shape mismatch."
        )

    output = (
        (
            1.0 - weight
        )
        * base
        + weight
        * native
    )

    if not np.isfinite(
        output
    ).all():
        raise ValueError(
            "Native categorical XGB "
            "blend is non-finite."
        )

    return np.clip(
        output,
        1.0e-6,
        1.0 - 1.0e-6,
    )


def select_xgb_native_cat_weight(
    folds: Sequence[
        Mapping[str, object]
    ],
    *,
    fold_importance: Sequence[float],
    grid_step: float,
    maximum_weight: float,
    minimum_material_protected_gain: float,
    minimum_forward_improvement: float,
    maximum_2023_regression: float,
) -> tuple[
    float,
    Dict[str, object],
]:
    """Select a conservative native-categorical XGB blend.

    Each fold must already contain the final V13 logical-XGB
    prediction in fold["predictions"]["xgb"] and the new expert
    prediction in fold["xgb_native_cat_prediction"].

    No test prediction or test statistic is used.
    """

    labels = [
        str(
            fold[
                "validation_label"
            ]
        )
        for fold in folds
    ]

    required = {
        "2023",
        "2024",
        "2024_late_abs",
    }

    if set(labels) != required:
        raise ValueError(
            "Native categorical XGB "
            "requires the three temporal "
            f"folds, got {labels}"
        )

    importance = np.asarray(
        fold_importance,
        dtype=np.float64,
    )

    if importance.shape != (
        len(folds),
    ):
        raise ValueError(
            "fold_importance shape "
            "mismatch."
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

    step = float(
        grid_step
    )

    maximum = float(
        maximum_weight
    )

    if (
        step <= 0.0
        or maximum < 0.0
        or maximum > 1.0
    ):
        raise ValueError(
            "Invalid native categorical "
            "XGB weight grid."
        )

    n_steps = int(
        round(
            maximum
            / step
        )
    )

    if not np.isclose(
        n_steps * step,
        maximum,
        atol=1.0e-12,
    ):
        raise ValueError(
            "maximum_weight must be "
            "divisible by grid_step."
        )

    baseline_scores: Dict[
        str,
        float,
    ] = {}

    for fold in folds:
        label = str(
            fold[
                "validation_label"
            ]
        )

        baseline_scores[
            label
        ] = _brier(
            fold[
                "y_true"
            ],
            fold[
                "predictions"
            ][
                "xgb"
            ],
        )

    baseline_forward = float(
        sum(
            importance[index]
            * baseline_scores[
                labels[index]
            ]
            for index
            in range(
                len(folds)
            )
        )
    )

    candidate_records = []

    for step_index in range(
        n_steps + 1
    ):
        weight = float(
            step_index
            * step
        )

        fold_scores: Dict[
            str,
            float,
        ] = {}

        gain_vs_base: Dict[
            str,
            float,
        ] = {}

        for fold in folds:
            label = str(
                fold[
                    "validation_label"
                ]
            )

            prediction = (
                blend_xgb_native_cat(
                    fold[
                        "predictions"
                    ][
                        "xgb"
                    ],
                    fold[
                        "xgb_native_cat_prediction"
                    ],
                    weight,
                )
            )

            score = _brier(
                fold[
                    "y_true"
                ],
                prediction,
            )

            fold_scores[
                label
            ] = score

            gain_vs_base[
                label
            ] = float(
                baseline_scores[
                    label
                ]
                - score
            )

        forward_score = float(
            sum(
                importance[index]
                * fold_scores[
                    labels[index]
                ]
                for index
                in range(
                    len(folds)
                )
            )
        )

        forward_gain = float(
            baseline_forward
            - forward_score
        )

        protected_gains = (
            gain_vs_base[
                "2024"
            ],
            gain_vs_base[
                "2024_late_abs"
            ],
        )

        if weight == 0.0:
            feasible = True
        else:
            protected_non_degradation = (
                min(
                    protected_gains
                )
                >= -1.0e-15
            )

            material = (
                max(
                    protected_gains
                )
                + 1.0e-15
                >= float(
                    minimum_material_protected_gain
                )
            )

            forward_material = (
                forward_gain
                + 1.0e-15
                >= float(
                    minimum_forward_improvement
                )
            )

            guard_2023 = (
                fold_scores[
                    "2023"
                ]
                <= baseline_scores[
                    "2023"
                ]
                + float(
                    maximum_2023_regression
                )
                + 1.0e-15
            )

            feasible = bool(
                protected_non_degradation
                and material
                and forward_material
                and guard_2023
            )

        candidate_records.append(
            {
                "weight": (
                    weight
                ),
                "fold_brier": {
                    key: float(value)
                    for key, value
                    in fold_scores.items()
                },
                "gain_vs_base": {
                    key: float(value)
                    for key, value
                    in gain_vs_base.items()
                },
                "forward_weighted_brier": (
                    forward_score
                ),
                "forward_improvement": (
                    forward_gain
                ),
                "feasible": bool(
                    feasible
                ),
            }
        )

    feasible_records = [
        record
        for record
        in candidate_records
        if record[
            "feasible"
        ]
    ]

    if not feasible_records:
        raise RuntimeError(
            "Native categorical XGB "
            "selection has no feasible "
            "baseline candidate."
        )

    selected = min(
        feasible_records,
        key=lambda record: (
            record[
                "forward_weighted_brier"
            ],
            record[
                "weight"
            ],
        ),
    )

    selected_weight = float(
        selected[
            "weight"
        ]
    )

    report = {
        "method": (
            "protected_native_"
            "categorical_xgb_blend"
        ),
        "selected_weight": (
            selected_weight
        ),
        "accepted_nonzero": bool(
            selected_weight > 0.0
        ),
        "baseline_forward_brier": (
            baseline_forward
        ),
        "selected_forward_brier": float(
            selected[
                "forward_weighted_brier"
            ]
        ),
        "forward_improvement": float(
            selected[
                "forward_improvement"
            ]
        ),
        "gain_vs_base": dict(
            selected[
                "gain_vs_base"
            ]
        ),
        "fold_brier": dict(
            selected[
                "fold_brier"
            ]
        ),
        "grid_step": step,
        "maximum_weight": maximum,
        "minimum_material_protected_gain": float(
            minimum_material_protected_gain
        ),
        "minimum_forward_improvement": float(
            minimum_forward_improvement
        ),
        "maximum_2023_regression": float(
            maximum_2023_regression
        ),
        "candidates": (
            candidate_records
        ),
    }

    return (
        selected_weight,
        report,
    )
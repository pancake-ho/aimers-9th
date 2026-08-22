from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


XGB_VIEW_ORDER = (
    "base",
    "representation",
    "representative",
)


def fit_numeric_pca_state(
    X: pd.DataFrame,
    numerical_cols: Sequence[str],
    *,
    n_components: int = 8,
    chunk_size: int = 65536,
) -> dict[str, object]:
    """Fit deterministic train-only PCA on numerical features.

    Target labels are never used.

    We diagonalize the standardized-feature covariance matrix instead
    of materializing a second full standardized training matrix.
    """

    columns = [
        str(column)
        for column in numerical_cols
    ]

    if not columns:
        raise ValueError(
            "PCA requires at least one "
            "numerical feature."
        )

    if len(set(columns)) != len(columns):
        raise ValueError(
            "Duplicate PCA input columns."
        )

    n_components = int(n_components)

    if (
        n_components <= 0
        or n_components > len(columns)
    ):
        raise ValueError(
            "Invalid PCA component count: "
            f"{n_components}"
        )

    values = X[columns]

    mean = (
        values
        .mean(axis=0)
        .to_numpy(
            dtype=np.float64
        )
    )

    std = (
        values
        .std(
            axis=0,
            ddof=0,
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    mean[
        ~np.isfinite(mean)
    ] = 0.0

    std[
        ~np.isfinite(std)
        | (std < 1.0e-6)
    ] = 1.0

    dimension = len(columns)

    covariance = np.zeros(
        (
            dimension,
            dimension,
        ),
        dtype=np.float64,
    )

    n_rows = int(
        len(X)
    )

    if n_rows <= 0:
        raise ValueError(
            "PCA training frame is empty."
        )

    for start in range(
        0,
        n_rows,
        int(chunk_size),
    ):
        stop = min(
            n_rows,
            start + int(chunk_size),
        )

        chunk = (
            X.iloc[
                start:stop
            ][columns]
            .to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

        chunk -= mean
        chunk /= std

        np.clip(
            chunk,
            -8.0,
            8.0,
            out=chunk,
        )

        covariance += (
            chunk.T @ chunk
        )

    covariance /= float(
        n_rows
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    order = order[
        :n_components
    ]

    selected_values = (
        eigenvalues[
            order
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    components = (
        eigenvectors[
            :,
            order
        ]
        .T
        .astype(
            np.float64,
            copy=True,
        )
    )

    # Deterministic sign convention.
    for index in range(
        components.shape[0]
    ):
        row = components[
            index
        ]

        pivot = int(
            np.argmax(
                np.abs(row)
            )
        )

        if row[pivot] < 0.0:
            components[
                index
            ] *= -1.0

    total_variance = float(
        np.maximum(
            eigenvalues,
            0.0,
        ).sum()
    )

    if total_variance > 0.0:
        explained_ratio = (
            np.maximum(
                selected_values,
                0.0,
            )
            / total_variance
        )
    else:
        explained_ratio = np.zeros(
            n_components,
            dtype=np.float64,
        )

    return {
        "state_version": 1,
        "input_columns": columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "components": (
            components.tolist()
        ),
        "explained_variance_ratio": (
            explained_ratio.tolist()
        ),
        "clip": 8.0,
        "output_prefix": (
            "mv_pca"
        ),
    }


def apply_numeric_pca_state(
    X: pd.DataFrame,
    state: Mapping[
        str,
        object,
    ],
    *,
    chunk_size: int = 65536,
) -> pd.DataFrame:
    columns = list(
        state[
            "input_columns"
        ]
    )

    mean = np.asarray(
        state["mean"],
        dtype=np.float64,
    )

    std = np.asarray(
        state["std"],
        dtype=np.float64,
    )

    components = np.asarray(
        state["components"],
        dtype=np.float64,
    )

    clip = float(
        state.get(
            "clip",
            8.0,
        )
    )

    if components.ndim != 2:
        raise ValueError(
            "Invalid PCA component matrix."
        )

    if (
        components.shape[1]
        != len(columns)
    ):
        raise ValueError(
            "PCA component/input mismatch."
        )

    n_rows = int(
        len(X)
    )

    output = np.empty(
        (
            n_rows,
            components.shape[0],
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        n_rows,
        int(chunk_size),
    ):
        stop = min(
            n_rows,
            start + int(chunk_size),
        )

        chunk = (
            X.iloc[
                start:stop
            ][columns]
            .to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

        chunk -= mean
        chunk /= std

        np.clip(
            chunk,
            -clip,
            clip,
            out=chunk,
        )

        projection = (
            chunk
            @ components.T
        )

        output[
            start:stop
        ] = projection.astype(
            np.float32,
            copy=False,
        )

    prefix = str(
        state.get(
            "output_prefix",
            "mv_pca",
        )
    )

    names = [
        f"{prefix}_{index:02d}"
        for index in range(
            components.shape[0]
        )
    ]

    result = pd.DataFrame(
        output,
        index=X.index,
        columns=names,
    )

    if not np.isfinite(
        output
    ).all():
        raise ValueError(
            "PCA representation contains "
            "NaN or infinity."
        )

    return result


def build_representation_view(
    X: pd.DataFrame,
    raw_feature_names: Sequence[str],
    pca_state: Mapping[
        str,
        object,
    ],
) -> pd.DataFrame:
    raw_names = [
        str(name)
        for name in raw_feature_names
    ]

    if not raw_names:
        raise ValueError(
            "Representation view has "
            "no raw features."
        )

    missing = [
        name
        for name in raw_names
        if name not in X.columns
    ]

    if missing:
        raise ValueError(
            "Representation raw features "
            f"are missing: {missing[:10]}"
        )

    pca = apply_numeric_pca_state(
        X,
        pca_state,
    )

    raw = (
        X[
            raw_names
        ]
        .copy()
    )

    overlap = set(
        raw.columns
    ) & set(
        pca.columns
    )

    if overlap:
        raise ValueError(
            "Representation feature "
            f"name collision: {overlap}"
        )

    output = pd.concat(
        [
            raw,
            pca,
        ],
        axis=1,
    )

    return output


def _frequency_balance_factor(
    values: pd.Series,
) -> np.ndarray:
    counts = (
        values
        .value_counts(
            dropna=False
        )
    )

    if counts.empty:
        return np.ones(
            len(values),
            dtype=np.float64,
        )

    anchor = float(
        np.median(
            counts.to_numpy(
                dtype=np.float64
            )
        )
    )

    anchor = max(
        anchor,
        1.0,
    )

    row_counts = (
        values
        .map(counts)
        .to_numpy(
            dtype=np.float64
        )
    )

    row_counts = np.maximum(
        row_counts,
        1.0,
    )

    return np.sqrt(
        anchor
        / row_counts
    )


def make_representative_sample_weight(
    X: pd.DataFrame,
    base_weight: np.ndarray,
    *,
    group_columns: Sequence[str],
    leverage_column: str | None,
    leverage_bins: int,
    clip_low: float,
    clip_high: float,
) -> tuple[
    np.ndarray,
    dict[str, object],
]:
    """Feature-only representative weighting.

    No labels or validation/test statistics are used.
    """

    weight = np.asarray(
        base_weight,
        dtype=np.float64,
    )

    if weight.shape != (
        len(X),
    ):
        raise ValueError(
            "Representative weight "
            "length mismatch."
        )

    factors = []

    used_groups = []

    for column in group_columns:
        if column not in X.columns:
            continue

        factor = (
            _frequency_balance_factor(
                X[column]
            )
        )

        factors.append(
            factor
        )

        used_groups.append(
            str(column)
        )

    leverage_used = False

    if (
        leverage_column
        and leverage_column
        in X.columns
    ):
        leverage = pd.to_numeric(
            X[
                leverage_column
            ],
            errors="coerce",
        )

        leverage = leverage.fillna(
            float(
                leverage.median()
            )
            if leverage.notna().any()
            else 0.0
        )

        try:
            bins = pd.qcut(
                leverage,
                q=int(
                    leverage_bins
                ),
                labels=False,
                duplicates="drop",
            )

            factors.append(
                _frequency_balance_factor(
                    bins
                )
            )

            leverage_used = True

        except ValueError:
            pass

    if factors:
        stacked = np.vstack(
            factors
        )

        log_factor = (
            np.log(
                np.clip(
                    stacked,
                    1.0e-8,
                    None,
                )
            )
            .mean(
                axis=0
            )
        )

        factor = np.exp(
            log_factor
        )

    else:
        factor = np.ones(
            len(X),
            dtype=np.float64,
        )

    factor = np.clip(
        factor,
        float(clip_low),
        float(clip_high),
    )

    # Preserve overall loss scale.
    mean_factor = float(
        factor.mean()
    )

    if (
        not np.isfinite(
            mean_factor
        )
        or mean_factor <= 0.0
    ):
        raise RuntimeError(
            "Invalid representative "
            "weight normalization."
        )

    factor /= mean_factor

    output = (
        weight
        * factor
    )

    original_sum = float(
        weight.sum()
    )

    output_sum = float(
        output.sum()
    )

    if (
        output_sum <= 0.0
        or not np.isfinite(
            output_sum
        )
    ):
        raise RuntimeError(
            "Invalid representative "
            "sample weights."
        )

    output *= (
        original_sum
        / output_sum
    )

    diagnostics = {
        "group_columns": (
            used_groups
        ),
        "leverage_column": (
            leverage_column
            if leverage_used
            else None
        ),
        "factor_min": float(
            factor.min()
        ),
        "factor_max": float(
            factor.max()
        ),
        "factor_mean": float(
            factor.mean()
        ),
        "factor_std": float(
            factor.std()
        ),
    }

    return (
        output.astype(
            np.float32,
            copy=False,
        ),
        diagnostics,
    )


def blend_xgb_views(
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
        len(XGB_VIEW_ORDER),
    ):
        raise ValueError(
            "XGB multi-view weight "
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
            "Invalid XGB multi-view "
            "weights."
        )

    weight /= weight.sum()

    predictions = []

    for name in XGB_VIEW_ORDER:
        if name not in views:
            raise KeyError(
                f"Missing XGB view: {name}"
            )

        prediction = np.asarray(
            views[name],
            dtype=np.float64,
        )

        predictions.append(
            prediction
        )

    stacked = np.column_stack(
        predictions
    )

    if (
        stacked.shape[1]
        != len(
            XGB_VIEW_ORDER
        )
    ):
        raise RuntimeError(
            "Invalid XGB view matrix."
        )

    output = (
        stacked
        @ weight
    )

    return np.clip(
        output,
        1.0e-6,
        1.0 - 1.0e-6,
    )


def select_xgb_multiview_weights(
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
    """Choose a small, constrained internal XGB ensemble."""

    importance = np.asarray(
        fold_importance,
        dtype=np.float64,
    )

    if importance.shape != (
        len(folds),
    ):
        raise ValueError(
            "XGB view fold importance "
            "shape mismatch."
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
            "Invalid XGB view fold "
            "importance."
        )

    importance /= (
        importance.sum()
    )

    n_steps = int(
        round(
            1.0
            / float(
                grid_step
            )
        )
    )

    if (
        n_steps <= 0
        or not np.isclose(
            n_steps
            * float(
                grid_step
            ),
            1.0,
        )
    ):
        raise ValueError(
            "XGB multi-view grid step "
            "must divide one."
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
                "Protected XGB view fold "
                f"is absent: {label}"
            )

        protected_indices.append(
            labels.index(
                label
            )
        )

    base_scores = []

    for fold in folds:
        target = np.asarray(
            fold["y_true"],
            dtype=np.float64,
        )

        base = np.asarray(
            fold[
                "xgb_views"
            ]["base"],
            dtype=np.float64,
        )

        base_scores.append(
            float(
                np.mean(
                    (
                        base
                        - target
                    )
                    ** 2
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
                base_scores,
                dtype=np.float64,
            ),
        )
    )

    candidates = 0
    feasible = 1

    for base_step in range(
        n_steps + 1
    ):
        for repr_step in range(
            n_steps
            - base_step
            + 1
        ):
            representative_step = (
                n_steps
                - base_step
                - repr_step
            )

            weights = (
                np.asarray(
                    [
                        base_step,
                        repr_step,
                        representative_step,
                    ],
                    dtype=np.float64,
                )
                / float(
                    n_steps
                )
            )

            candidates += 1

            if (
                weights[0]
                < float(
                    minimum_base_weight
                )
                - 1.0e-15
            ):
                continue

            if (
                weights[1]
                > float(
                    maximum_aux_weight
                )
                + 1.0e-15
                or weights[2]
                > float(
                    maximum_aux_weight
                )
                + 1.0e-15
            ):
                continue

            fold_scores = []

            for fold in folds:
                prediction = (
                    blend_xgb_views(
                        fold[
                            "xgb_views"
                        ],
                        weights,
                    )
                )

                target = np.asarray(
                    fold["y_true"],
                    dtype=np.float64,
                )

                score = float(
                    np.mean(
                        (
                            prediction
                            - target
                        )
                        ** 2
                    )
                )

                fold_scores.append(
                    score
                )

            violates = any(
                fold_scores[index]
                > (
                    base_scores[index]
                    + float(
                        protected_tolerance
                    )
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
                        fold_scores,
                        dtype=np.float64,
                    ),
                )
            )

            # Prefer more base-model weight on exact ties.
            if (
                objective
                < best_objective
                - 1.0e-15
                or (
                    np.isclose(
                        objective,
                        best_objective,
                        atol=1.0e-15,
                    )
                    and weights[0]
                    > best_weight[0]
                )
            ):
                best_objective = (
                    objective
                )

                best_weight = (
                    weights.copy()
                )

                best_scores = list(
                    fold_scores
                )

    return (
        best_weight,
        {
            "view_order": list(
                XGB_VIEW_ORDER
            ),
            "weights": (
                best_weight.tolist()
            ),
            "forward_weighted_brier": float(
                best_objective
            ),
            "fold_brier": {
                labels[index]: float(
                    best_scores[index]
                )
                for index in range(
                    len(labels)
                )
            },
            "base_fold_brier": {
                labels[index]: float(
                    base_scores[index]
                )
                for index in range(
                    len(labels)
                )
            },
            "gain_vs_base": {
                labels[index]: float(
                    base_scores[index]
                    - best_scores[index]
                )
                for index in range(
                    len(labels)
                )
            },
            "evaluated_candidates": int(
                candidates
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
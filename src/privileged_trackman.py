from __future__ import annotations

import gc
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import ExperimentConfig
from src.metrics import evaluate_probabilities
from src.models import (
    train_xgboost_temporal_full,
)
from src.preprocessing import TabularPreprocessor
from src.splits import uniform_weights


PRIVILEGED_NUMERIC_COLUMNS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)

PRIVILEGED_PITCH_TYPE_COLUMN = (
    "priv_pitch_type_group"
)

TEACHER_PRIVILEGED_COLUMNS = (
    PRIVILEGED_PITCH_TYPE_COLUMN,
    *(
        f"priv_{name}"
        for name
        in PRIVILEGED_NUMERIC_COLUMNS
    ),
)

TRACKMAN_USECOLS = (
    "trackman_id",
    "season",
    "game_date",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_trackman_id",
    "pitcher_hand",
    "batter_hand",
    "pitch_type_group",
    *PRIVILEGED_NUMERIC_COLUMNS,
)


def _hand_code(value) -> int:
    if pd.isna(value):
        return -1

    token = str(value).strip().lower()

    if token in {
        "left",
        "l",
        "1",
        "1.0",
    }:
        return 1

    if token in {
        "right",
        "r",
        "2",
        "2.0",
    }:
        return 2

    return -1


def _half_code(value) -> int:
    if pd.isna(value):
        return -1

    token = str(value).strip().lower()

    if token in {
        "t",
        "top",
    }:
        return 0

    if token in {
        "b",
        "bottom",
    }:
        return 1

    return -1


def _numeric_int(
    value,
    default: int = -1,
) -> int:
    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return int(default)

    if not np.isfinite(result):
        return int(default)

    return int(result)


def _pitch_token(row) -> tuple:
    """
    Alignment token uses ONLY structural/pre-pitch state.

    Current-pitch type, speed, spin, movement, target, etc.
    are intentionally absent.
    """

    return (
        _numeric_int(
            row.game_month
        ),
        _numeric_int(
            row.game_dayofweek
        ),
        _numeric_int(
            row.inning
        ),
        _half_code(
            row.top_bottom
        ),
        _numeric_int(
            row.balls_before
        ),
        _numeric_int(
            row.strikes_before
        ),
        _numeric_int(
            row.outs_before
        ),
        _hand_code(
            row.pitcher_hand
        ),
        _hand_code(
            row.batter_hand
        ),
    )


def load_privileged_trackman(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(
            TRACKMAN_USECOLS
        ),
        encoding="utf-8-sig",
        low_memory=False,
    )

    frame["season"] = (
        pd.to_numeric(
            frame["season"],
            errors="raise",
        )
        .astype(np.int16)
    )

    frame[
        "pitcher_trackman_id"
    ] = (
        pd.to_numeric(
            frame[
                "pitcher_trackman_id"
            ],
            errors="coerce",
        )
    )

    frame = frame.dropna(
        subset=[
            "pitcher_trackman_id",
        ]
    ).copy()

    frame[
        "pitcher_trackman_id"
    ] = (
        frame[
            "pitcher_trackman_id"
        ]
        .astype(np.int64)
    )

    frame[
        "_parsed_game_date"
    ] = pd.to_datetime(
        frame["game_date"],
        errors="coerce",
    )

    if (
        frame[
            "_parsed_game_date"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Trackman game_date parsing "
            "failed."
        )

    return frame


def align_pitcher_season(
    main_rows: pd.DataFrame,
    trackman_rows: pd.DataFrame,
    *,
    entity_confidence: float,
    min_matching_block: int,
    min_pair_matched_rows: int,
    min_pair_match_ratio: float,
) -> pd.DataFrame:
    if (
        main_rows.empty
        or trackman_rows.empty
    ):
        return pd.DataFrame()

    main = (
        main_rows
        .copy()
    )

    main[
        "_asof_n"
    ] = pd.to_numeric(
        main[
            "asof_pitcher_n"
        ],
        errors="raise",
    )

    main = (
        main
        .sort_values(
            "_asof_n",
            kind="mergesort",
        )
    )

    if (
        main["_asof_n"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate asof_pitcher_n "
            "inside pitcher sequence."
        )

    trackman = (
        trackman_rows
        .sort_values(
            [
                "_parsed_game_date",
                "trackman_game_id",
                "pitch_no",
                "trackman_id",
            ],
            kind="mergesort",
        )
        .copy()
    )

    main_tokens = [
        _pitch_token(row)
        for row
        in main.itertuples()
    ]

    trackman_tokens = [
        _pitch_token(row)
        for row
        in trackman.itertuples()
    ]

    matcher = SequenceMatcher(
        a=main_tokens,
        b=trackman_tokens,
        autojunk=False,
    )

    records = []

    for block in (
        matcher.get_matching_blocks()
    ):
        if (
            int(block.size)
            < int(
                min_matching_block
            )
        ):
            continue

        for offset in range(
            int(block.size)
        ):
            main_position = (
                int(block.a)
                + offset
            )

            trackman_position = (
                int(block.b)
                + offset
            )

            main_index = (
                main.index[
                    main_position
                ]
            )

            tm_row = (
                trackman.iloc[
                    trackman_position
                ]
            )

            record = {
                "main_index": (
                    main_index
                ),
                "trackman_id": int(
                    tm_row[
                        "trackman_id"
                    ]
                ),
                "priv_entity_confidence": (
                    float(
                        entity_confidence
                    )
                ),
                "priv_alignment_block_size": (
                    float(
                        block.size
                    )
                ),
                PRIVILEGED_PITCH_TYPE_COLUMN: (
                    str(
                        tm_row[
                            "pitch_type_group"
                        ]
                    )
                    if not pd.isna(
                        tm_row[
                            "pitch_type_group"
                        ]
                    )
                    else "__MISSING__"
                ),
            }

            for name in (
                PRIVILEGED_NUMERIC_COLUMNS
            ):
                value = pd.to_numeric(
                    pd.Series(
                        [
                            tm_row[
                                name
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                record[
                    f"priv_{name}"
                ] = (
                    float(value)
                    if np.isfinite(value)
                    else np.nan
                )

            records.append(
                record
            )

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(
        records
    )

    matched_rows = int(
        len(result)
    )

    denominator = max(
        1,
        min(
            len(main),
            len(trackman),
        ),
    )

    match_ratio = float(
        matched_rows
        / denominator
    )

    if (
        matched_rows
        < int(
            min_pair_matched_rows
        )
        or match_ratio
        < float(
            min_pair_match_ratio
        )
    ):
        return pd.DataFrame()

    result[
        "priv_alignment_pair_ratio"
    ] = np.float32(
        match_ratio
    )

    if (
        result[
            "main_index"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "One main row received "
            "multiple Trackman rows."
        )

    if (
        result[
            "trackman_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "One Trackman pitch was "
            "used multiple times."
        )

    return result


def build_privileged_alignment(
    train: pd.DataFrame,
    trackman: pd.DataFrame,
    feature_state: Mapping[
        str,
        object,
    ],
    config: ExperimentConfig,
) -> tuple[
    pd.DataFrame,
    Dict[str, object],
]:
    """
    Recover high-precision TRAIN-ONLY pitch alignment.

    Pitcher identity is taken from the existing official-data-only
    Trackman entity state.

    For season S, the (S+1, pitcher_id) mapping is used because
    that mapping was fitted with Trackman/main rows through S.

    No control_success value participates in matching.
    """

    entity_state = (
        feature_state
        .get(
            "trackman",
            {},
        )
        .get(
            "entity"
        )
    )

    if not entity_state:
        raise RuntimeError(
            "Privileged distillation "
            "requires Trackman entity state."
        )

    mapping_profile = (
        entity_state[
            "profiles"
        ][
            "mapping"
        ]
    )

    mapping_lookup = (
        mapping_profile[
            "lookup"
        ]
    )

    seasons = sorted(
        int(value)
        for value
        in pd.unique(
            train["season"]
        )
    )

    aligned_parts = []

    audit_by_season = {}

    for season in seasons:
        main_season = (
            train.loc[
                pd.to_numeric(
                    train[
                        "season"
                    ],
                    errors="raise",
                )
                == season
            ]
        )

        tm_season = (
            trackman.loc[
                trackman[
                    "season"
                ]
                == season
            ]
        )

        attempted_pairs = 0
        accepted_pairs = 0
        matched_rows = 0

        for pitcher_id in pd.unique(
            main_season[
                "pitcher_id"
            ]
        ):
            try:
                main_pitcher_id = int(
                    pitcher_id
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            mapping = (
                mapping_lookup.get(
                    (
                        season + 1,
                        main_pitcher_id,
                    )
                )
            )

            if mapping is None:
                continue

            confidence = float(
                mapping.get(
                    "tm_entity_map_confidence",
                    0.0,
                )
            )

            if (
                confidence
                < float(
                    config
                    .privileged
                    .min_entity_confidence
                )
            ):
                continue

            trackman_pitcher_id = int(
                mapping[
                    "trackman_id"
                ]
            )

            main_pair = (
                main_season.loc[
                    pd.to_numeric(
                        main_season[
                            "pitcher_id"
                        ],
                        errors="coerce",
                    )
                    == main_pitcher_id
                ]
            )

            tm_pair = (
                tm_season.loc[
                    tm_season[
                        "pitcher_trackman_id"
                    ]
                    == trackman_pitcher_id
                ]
            )

            attempted_pairs += 1

            pair = (
                align_pitcher_season(
                    main_pair,
                    tm_pair,
                    entity_confidence=(
                        confidence
                    ),
                    min_matching_block=(
                        config
                        .privileged
                        .min_matching_block
                    ),
                    min_pair_matched_rows=(
                        config
                        .privileged
                        .min_pair_matched_rows
                    ),
                    min_pair_match_ratio=(
                        config
                        .privileged
                        .min_pair_match_ratio
                    ),
                )
            )

            if pair.empty:
                continue

            accepted_pairs += 1
            matched_rows += len(
                pair
            )

            pair[
                "season"
            ] = np.int16(
                season
            )

            pair[
                "pitcher_id"
            ] = np.int64(
                main_pitcher_id
            )

            pair[
                "pitcher_trackman_id"
            ] = np.int64(
                trackman_pitcher_id
            )

            aligned_parts.append(
                pair
            )

        season_audit = {
            "attempted_pairs": int(
                attempted_pairs
            ),
            "accepted_pairs": int(
                accepted_pairs
            ),
            "matched_rows": int(
                matched_rows
            ),
            "season_rows": int(
                len(main_season)
            ),
            "row_coverage": float(
                matched_rows
                / max(
                    len(main_season),
                    1,
                )
            ),
        }

        audit_by_season[
            str(season)
        ] = season_audit

        print(
            "[LUPI-ALIGN] "
            f"season={season} "
            f"pairs="
            f"{accepted_pairs}/"
            f"{attempted_pairs} "
            f"rows={matched_rows:,} "
            f"coverage="
            f"{season_audit['row_coverage']:.4f}"
        )

    if aligned_parts:
        aligned = pd.concat(
            aligned_parts,
            ignore_index=True,
        )
    else:
        aligned = pd.DataFrame()

    if aligned.empty:
        raise RuntimeError(
            "Privileged Trackman "
            "alignment produced zero rows."
        )

    if (
        aligned[
            "main_index"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Privileged alignment "
            "contains duplicate main rows."
        )

    if (
        aligned[
            "trackman_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Privileged alignment "
            "contains duplicate Trackman rows."
        )

    aligned = (
        aligned
        .set_index(
            "main_index",
            drop=True,
        )
        .sort_index()
    )

    total_coverage = float(
        len(aligned)
        / max(
            len(train),
            1,
        )
    )

    audit = {
        "matched_rows": int(
            len(aligned)
        ),
        "total_rows": int(
            len(train)
        ),
        "row_coverage": (
            total_coverage
        ),
        "by_season": (
            audit_by_season
        ),
        "one_to_one_main": True,
        "one_to_one_trackman": True,
    }

    return (
        aligned,
        audit,
    )


def _teacher_frame(
    features: pd.DataFrame,
    alignment: pd.DataFrame,
    indices,
) -> pd.DataFrame:
    frame = (
        features.loc[
            indices
        ]
        .copy()
    )

    privileged = (
        alignment
        .reindex(
            indices
        )
    )

    for column in (
        TEACHER_PRIVILEGED_COLUMNS
    ):
        frame[
            column
        ] = (
            privileged[
                column
            ]
            .to_numpy(
                copy=True
            )
        )

    return frame


def build_prequential_teacher_state(
    train: pd.DataFrame,
    features: pd.DataFrame,
    feature_state: Mapping[
        str,
        object,
    ],
    config: ExperimentConfig,
) -> Dict[str, object]:
    if not config.privileged.enabled:
        raise ValueError(
            "Privileged teacher requested "
            "while feature is disabled."
        )

    print(
        "[LUPI] Loading training-only "
        "Trackman privileged view..."
    )

    trackman = (
        load_privileged_trackman(
            config.paths.trackman_path
        )
    )

    (
        alignment,
        alignment_audit,
    ) = build_privileged_alignment(
        train,
        trackman,
        feature_state,
        config,
    )

    del trackman
    gc.collect()

    n_rows = len(train)

    teacher_probability = np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )

    teacher_available = np.zeros(
        n_rows,
        dtype=bool,
    )

    target = (
        config
        .features
        .target_col
    )

    seasons = sorted(
        int(value)
        for value
        in pd.unique(
            train["season"]
        )
    )

    teacher_metrics = {}

    aligned_indices = (
        alignment.index
    )

    aligned_season = (
        pd.to_numeric(
            train.loc[
                aligned_indices,
                "season",
            ],
            errors="raise",
        )
    )

    teacher_categorical = (
        *config
        .features
        .categorical_cols,
        PRIVILEGED_PITCH_TYPE_COLUMN,
    )

    for season in seasons[1:]:
        train_indices = (
            aligned_indices[
                (
                    aligned_season
                    < season
                )
                .to_numpy()
            ]
        )

        valid_indices = (
            aligned_indices[
                (
                    aligned_season
                    == season
                )
                .to_numpy()
            ]
        )

        if (
            len(train_indices)
            < int(
                config
                .privileged
                .teacher_min_train_rows
            )
            or len(
                valid_indices
            )
            == 0
        ):
            print(
                "[LUPI-TEACHER] "
                f"season={season} "
                "skipped "
                f"train={len(train_indices):,} "
                f"valid={len(valid_indices):,}"
            )
            continue

        teacher_train = (
            _teacher_frame(
                features,
                alignment,
                train_indices,
            )
        )

        teacher_valid = (
            _teacher_frame(
                features,
                alignment,
                valid_indices,
            )
        )

        preprocessor = (
            TabularPreprocessor(
                categorical_cols=(
                    teacher_categorical
                ),
                excluded_cols=(
                    config
                    .features
                    .excluded_cols
                ),
            )
        )

        X_train = (
            preprocessor
            .fit_transform(
                teacher_train
            )
        )

        X_valid = (
            preprocessor
            .transform(
                teacher_valid
            )
        )

        y_train = (
            train.loc[
                train_indices,
                target,
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        # No season-S labels are supplied to teacher fitting.
        model = (
            train_xgboost_temporal_full(
                X_train,
                y_train,
                uniform_weights(
                    len(y_train)
                ),
                config.models,
                seed=(
                    config
                    .privileged
                    .teacher_seed
                    + int(season)
                ),
                num_boost_round=(
                    config
                    .privileged
                    .teacher_num_boost_round
                ),
            )
        )

        dvalid = xgb.DMatrix(
            X_valid,
            feature_names=list(
                X_valid.columns
            ),
        )

        prediction = np.asarray(
            model.predict(
                dvalid
            ),
            dtype=np.float64,
        )

        if not np.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                "Teacher produced "
                "non-finite prediction."
            )

        positions = (
            train.index
            .get_indexer(
                valid_indices
            )
        )

        if (
            positions < 0
        ).any():
            raise RuntimeError(
                "Could not map teacher "
                "indices to train positions."
            )

        teacher_probability[
            positions
        ] = prediction.astype(
            np.float32
        )

        teacher_available[
            positions
        ] = True

        y_valid = (
            train.loc[
                valid_indices,
                target,
            ]
            .to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

        metrics = (
            evaluate_probabilities(
                y_valid,
                prediction,
            )
        )

        teacher_metrics[
            str(season)
        ] = metrics

        print(
            "[LUPI-TEACHER] "
            f"season={season} "
            f"train={len(train_indices):,} "
            f"predict={len(valid_indices):,} "
            f"brier={metrics['brier']:.8f}"
        )

        del (
            teacher_train,
            teacher_valid,
            preprocessor,
            X_train,
            X_valid,
            y_train,
            y_valid,
            model,
            dvalid,
            prediction,
        )

        gc.collect()

    available_rows = int(
        teacher_available.sum()
    )

    print(
        "[LUPI] "
        f"teacher_oof_rows={available_rows:,} "
        f"coverage="
        f"{available_rows / max(n_rows, 1):.4f}"
    )

    # IMPORTANT:
    # alignment/current-pitch Trackman columns are intentionally
    # NOT returned to the deployment pipeline.
    del alignment
    gc.collect()

    return {
        "teacher_probability": (
            teacher_probability
        ),
        "teacher_available": (
            teacher_available
        ),
        "alignment_audit": (
            alignment_audit
        ),
        "teacher_metrics": (
            teacher_metrics
        ),
    }


def build_distillation_target(
    y_true,
    teacher_probability,
    teacher_available,
    *,
    strength: float,
) -> np.ndarray:
    if not (
        0.0
        <= float(strength)
        <= 1.0
    ):
        raise ValueError(
            "Distillation strength "
            "must be in [0, 1]."
        )

    y = np.asarray(
        y_true,
        dtype=np.float32,
    )

    teacher = np.asarray(
        teacher_probability,
        dtype=np.float32,
    )

    available = np.asarray(
        teacher_available,
        dtype=bool,
    )

    if (
        y.shape
        != teacher.shape
        or y.shape
        != available.shape
    ):
        raise ValueError(
            "Distillation arrays "
            "must have equal shape."
        )

    result = y.copy()

    usable = (
        available
        & np.isfinite(
            teacher
        )
    )

    result[
        usable
    ] = (
        (
            1.0
            - float(strength)
        )
        * y[
            usable
        ]
        + float(strength)
        * teacher[
            usable
        ]
    )

    return np.clip(
        result,
        1.0e-5,
        1.0 - 1.0e-5,
    )


def select_lupi_weight(
    folds,
    *,
    candidate_weights,
    fold_importance,
    minimum_forward_gain,
    maximum_2023_regression,
):
    importance = np.asarray(
        fold_importance,
        dtype=np.float64,
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
        for fold
        in folds
    ]

    if labels != [
        "2023",
        "2024",
        "2024_late_abs",
    ]:
        raise ValueError(
            "Unexpected folds for "
            "LUPI selection."
        )

    base_scores = {}

    for fold in folds:
        label = (
            fold[
                "validation_label"
            ]
        )

        y = np.asarray(
            fold[
                "y_true"
            ],
            dtype=np.float64,
        )

        base = np.asarray(
            fold[
                "predictions"
            ][
                "xgb"
            ],
            dtype=np.float64,
        )

        base_scores[
            label
        ] = float(
            np.mean(
                (
                    base
                    - y
                )
                ** 2
            )
        )

    base_objective = float(
        sum(
            importance[index]
            * base_scores[
                labels[index]
            ]
            for index
            in range(
                len(folds)
            )
        )
    )

    candidates = []

    for raw_weight in (
        candidate_weights
    ):
        weight = float(
            raw_weight
        )

        fold_scores = {}

        feasible = True

        for fold in folds:
            label = (
                fold[
                    "validation_label"
                ]
            )

            y = np.asarray(
                fold[
                    "y_true"
                ],
                dtype=np.float64,
            )

            base = np.asarray(
                fold[
                    "predictions"
                ][
                    "xgb"
                ],
                dtype=np.float64,
            )

            student = np.asarray(
                fold[
                    "xgb_lupi_prediction"
                ],
                dtype=np.float64,
            )

            prediction = (
                (
                    1.0
                    - weight
                )
                * base
                + weight
                * student
            )

            score = float(
                np.mean(
                    (
                        prediction
                        - y
                    )
                    ** 2
                )
            )

            fold_scores[
                label
            ] = score

            if (
                label
                in {
                    "2024",
                    "2024_late_abs",
                }
                and score
                > base_scores[
                    label
                ]
                + 1.0e-15
            ):
                feasible = False

            if (
                label
                == "2023"
                and score
                > base_scores[
                    label
                ]
                + float(
                    maximum_2023_regression
                )
                + 1.0e-15
            ):
                feasible = False

        objective = float(
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

        candidates.append(
            {
                "weight": (
                    weight
                ),
                "fold_brier": (
                    fold_scores
                ),
                "forward_brier": (
                    objective
                ),
                "forward_gain": (
                    base_objective
                    - objective
                ),
                "feasible": (
                    bool(feasible)
                ),
            }
        )

    feasible_candidates = [
        item
        for item
        in candidates
        if item[
            "feasible"
        ]
    ]

    if not feasible_candidates:
        raise RuntimeError(
            "No feasible LUPI "
            "blend candidate."
        )

    best = min(
        feasible_candidates,
        key=lambda item: (
            item[
                "forward_brier"
            ],
            item[
                "weight"
            ],
        ),
    )

    accepted = bool(
        best["weight"]
        > 0.0
        and best[
            "forward_gain"
        ]
        >= float(
            minimum_forward_gain
        )
    )

    if not accepted:
        best = next(
            item
            for item
            in candidates
            if np.isclose(
                item[
                    "weight"
                ],
                0.0,
            )
        )

    selected_weight = float(
        best[
            "weight"
        ]
    )

    gain_vs_base = {
        label: float(
            base_scores[label]
            - best[
                "fold_brier"
            ][label]
        )
        for label
        in labels
    }

    return (
        selected_weight,
        {
            "method": (
                "prequential_privileged_"
                "teacher_distillation"
            ),
            "selected_weight": (
                selected_weight
            ),
            "accepted_nonzero": (
                bool(
                    accepted
                )
            ),
            "base_forward_brier": (
                base_objective
            ),
            "forward_brier": float(
                best[
                    "forward_brier"
                ]
            ),
            "forward_gain": float(
                base_objective
                - best[
                    "forward_brier"
                ]
            ),
            "gain_vs_base": (
                gain_vs_base
            ),
            "candidates": (
                candidates
            ),
        },
    )
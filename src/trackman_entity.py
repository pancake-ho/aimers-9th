from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


PITCH_GROUPS = ("fastball", "breaking", "offspeed")
HAND_ALIASES = {
    "Left": 1,
    "Right": 2,
    "L": 1,
    "R": 2,
    "1": 1,
    "2": 2,
}
MISSING_TEAM_TOKEN = "__MISSING_TEAM__"


def _official_team_token(value: object) -> str:
    """Return the exact team token contained in the official Trackman file.

    No semantic aliasing, organization-history lookup, or external baseball
    knowledge is applied. Team evidence remains a soft resolver signal.
    """
    if pd.isna(value):
        return MISSING_TEAM_TOKEN

    token = str(value).strip()
    return token if token else MISSING_TEAM_TOKEN


@dataclass(frozen=True)
class ResolutionThresholds:
    min_pitches: int
    max_distance: float
    min_margin_ratio: float
    strong_margin_ratio: float
    min_count_ratio: float
    max_count_ratio: float
    team_penalty: float
    team_min_support: int
    team_min_dominance: float


@dataclass
class ResolutionResult:
    mapping: Dict[int, Dict[str, float | int]]
    top1: Dict[int, int]
    team_map: Dict[int, str]
    audit: Dict[str, object]


def _mode_as_int(values: pd.Series) -> int:
    modes = values.dropna().mode()
    return int(modes.iloc[0]) if not modes.empty else -1


def _main_fingerprint(main: pd.DataFrame, seasons: Sequence[int]) -> pd.DataFrame:
    required = {
        "season",
        "pitcher_id",
        "pitcher_hand",
        "pitcher_team_id",
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    }
    missing = required - set(main.columns)
    if missing:
        raise ValueError(f"Missing main entity-resolution columns: {sorted(missing)}")

    frame = main[list(required)].copy()
    frame["pitcher_id"] = pd.to_numeric(frame["pitcher_id"], errors="coerce")
    mapped_hand = frame["pitcher_hand"].map(HAND_ALIASES)
    frame["pitcher_hand"] = mapped_hand.fillna(
        pd.to_numeric(frame["pitcher_hand"], errors="coerce")
    )
    frame["pitcher_team_id"] = pd.to_numeric(
        frame["pitcher_team_id"], errors="coerce"
    )
    frame["asof_pitcher_pitchmix_n"] = pd.to_numeric(
        frame["asof_pitcher_pitchmix_n"], errors="coerce"
    ).fillna(0.0)
    frame = frame.dropna(subset=["pitcher_id", "pitcher_hand", "pitcher_team_id"])
    frame["pitcher_id"] = frame["pitcher_id"].astype(np.int64)

    grouped = frame.groupby("pitcher_id", observed=True, sort=False)
    # The maximum official as-of count is the latest cumulative repertoire
    # snapshot.  This is robust even if CSV rows are not physically sorted.
    latest_index = grouped["asof_pitcher_pitchmix_n"].idxmax()
    latest = frame.loc[latest_index].set_index("pitcher_id")
    result = pd.DataFrame(
        {
            "n": grouped.size().astype(np.int64),
            "hand": grouped["pitcher_hand"].agg(_mode_as_int),
            "team": latest["pitcher_team_id"].astype(np.int64),
        }
    )
    for group in PITCH_GROUPS:
        result[group] = pd.to_numeric(
            latest[f"asof_pitcher_{group}_rate"], errors="coerce"
        )

    seasonal = (
        frame.groupby(["pitcher_id", "season"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(seasons), fill_value=0)
    )
    for season in seasons:
        result[f"n_{season}"] = seasonal[season].reindex(result.index, fill_value=0)
    return result.dropna(subset=[*PITCH_GROUPS])


def _trackman_fingerprint(
    trackman: pd.DataFrame,
    seasons: Sequence[int],
) -> pd.DataFrame:
    required = {
        "season",
        "pitcher_trackman_id",
        "pitcher_hand",
        "pitcher_team",
        "pitch_type_group",
    }
    missing = required - set(trackman.columns)
    if missing:
        raise ValueError(
            f"Missing Trackman entity-resolution columns: {sorted(missing)}"
        )

    frame = trackman[list(required)].copy()

    frame["pitcher_trackman_id"] = pd.to_numeric(
        frame["pitcher_trackman_id"],
        errors="coerce",
    )

    mapped_hand = frame["pitcher_hand"].map(HAND_ALIASES)
    frame["pitcher_hand"] = mapped_hand.fillna(
        pd.to_numeric(
            frame["pitcher_hand"],
            errors="coerce",
        )
    )

    frame = frame.dropna(
        subset=[
            "pitcher_trackman_id",
            "pitcher_hand",
        ]
    )

    frame["pitcher_trackman_id"] = (
        frame["pitcher_trackman_id"]
        .astype(np.int64)
    )

    # Exact raw token from the official Trackman file.
    frame["team_token"] = (
        frame["pitcher_team"]
        .map(_official_team_token)
    )

    grouped = frame.groupby(
        "pitcher_trackman_id",
        observed=True,
        sort=False,
    )

    result = pd.DataFrame(
        {
            "n": grouped.size().astype(np.int64),
            "hand": grouped["pitcher_hand"].agg(_mode_as_int),
            "team_tokens": grouped["team_token"].agg(
                lambda values: tuple(
                    sorted(
                        set(values)
                        - {MISSING_TEAM_TOKEN}
                    )
                )
            ),
        }
    )

    counts = pd.crosstab(
        frame["pitcher_trackman_id"],
        frame["pitch_type_group"],
    )
    counts = counts.reindex(
        columns=list(PITCH_GROUPS),
        fill_value=0,
    )

    for group in PITCH_GROUPS:
        result[group] = (
            counts[group]
            .reindex(
                result.index,
                fill_value=0,
            )
            / result["n"]
        )

    seasonal = (
        frame.groupby(
            [
                "pitcher_trackman_id",
                "season",
            ],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=list(seasons),
            fill_value=0,
        )
    )

    for season in seasons:
        result[f"n_{season}"] = (
            seasonal[season]
            .reindex(
                result.index,
                fill_value=0,
            )
        )

    return result


def _distance_matrix(
    main_fp: pd.DataFrame,
    trackman_fp: pd.DataFrame,
    seasons: Sequence[int],
    team_map: Mapping[int, str] | None,
    team_penalty: float,
) -> np.ndarray:
    # Fingerprint scales remain label-free and fixed.
    result = np.full(
        (len(main_fp), len(trackman_fp)),
        1.0e6,
        dtype=np.float64,
    )

    trackman_hand = (
        trackman_fp["hand"]
        .to_numpy(
            dtype=np.int64,
            copy=False,
        )
    )
    season_cols = [
        f"n_{season}"
        for season in seasons
    ]

    for row_pos, (_, row) in enumerate(
        main_fp.iterrows()
    ):
        candidate_pos = np.flatnonzero(
            trackman_hand == int(row["hand"])
        )
        if candidate_pos.size == 0:
            continue

        candidates = trackman_fp.iloc[
            candidate_pos
        ]

        count_distance = np.square(
            (
                np.log1p(
                    candidates["n"]
                    .to_numpy(dtype=np.float64)
                )
                - np.log1p(float(row["n"]))
            )
            / 0.18
        )

        mix_distance = np.square(
            (
                candidates[
                    list(PITCH_GROUPS)
                ].to_numpy(dtype=np.float64)
                - row[
                    list(PITCH_GROUPS)
                ].to_numpy(dtype=np.float64)
            )
            / 0.06
        ).sum(axis=1)

        main_season = row[
            season_cols
        ].to_numpy(dtype=np.float64)
        main_season /= (
            main_season.sum()
            + 1.0e-12
        )

        trackman_season = (
            candidates[
                season_cols
            ].to_numpy(dtype=np.float64)
        )
        trackman_season /= (
            trackman_season.sum(
                axis=1,
                keepdims=True,
            )
            + 1.0e-12
        )

        season_distance = np.square(
            (
                trackman_season
                - main_season
            )
            / 0.10
        ).sum(axis=1)

        candidate_distance = (
            count_distance
            + mix_distance
            + season_distance
        )

        # Team can change ranking slightly but is never a hard condition.
        expected_team = (
            None
            if team_map is None
            else team_map.get(
                int(row["team"])
            )
        )

        if (
            expected_team is not None
            and team_penalty > 0.0
        ):
            candidate_distance += np.asarray(
                [
                    0.0
                    if expected_team in tokens
                    else float(team_penalty)
                    for tokens
                    in candidates[
                        "team_tokens"
                    ]
                ],
                dtype=np.float64,
            )

        result[
            row_pos,
            candidate_pos,
        ] = candidate_distance

    return result


def _infer_team_map(
    main_fp: pd.DataFrame,
    trackman_fp: pd.DataFrame,
    base_distance: np.ndarray,
    thresholds: ResolutionThresholds,
) -> Dict[int, str]:
    """Learn an exact-official-token soft correspondence.

    No KBO organization alias or external team-history information is used.
    """
    if len(trackman_fp) < 2:
        return {}

    order = np.argsort(
        base_distance,
        axis=1,
    )[:, :2]

    first = base_distance[
        np.arange(len(main_fp)),
        order[:, 0],
    ]
    second = base_distance[
        np.arange(len(main_fp)),
        order[:, 1],
    ]

    margins = second / (
        first + 1.0e-9
    )

    support: Dict[
        tuple[int, str],
        int,
    ] = {}

    for row_pos, (_, row) in enumerate(
        main_fp.iterrows()
    ):
        candidate = trackman_fp.iloc[
            order[row_pos, 0]
        ]

        count_ratio = (
            float(candidate["n"])
            / max(
                float(row["n"]),
                1.0,
            )
        )

        team_tokens = tuple(
            candidate["team_tokens"]
        )

        # Use only very clear identity candidates to infer a raw-token hint.
        if not (
            float(row["n"]) >= 1000
            and float(candidate["n"]) >= 1000
            and margins[row_pos] >= 1.5
            and first[row_pos] <= 8.0
            and 0.60 <= count_ratio <= 1.70
            and len(team_tokens) == 1
        ):
            continue

        key = (
            int(row["team"]),
            str(team_tokens[0]),
        )
        support[key] = (
            support.get(key, 0)
            + 1
        )

    team_map: Dict[int, str] = {}

    for main_team in sorted(
        {
            key[0]
            for key in support
        }
    ):
        candidates = sorted(
            (
                (count, token)
                for (
                    team,
                    token,
                ), count
                in support.items()
                if team == main_team
            ),
            reverse=True,
        )

        total = sum(
            count
            for count, _
            in candidates
        )

        if (
            candidates
            and candidates[0][0]
            >= thresholds.team_min_support
            and candidates[0][0]
            / max(total, 1)
            >= thresholds.team_min_dominance
        ):
            team_map[
                main_team
            ] = candidates[0][1]

    return team_map


def resolve_pitcher_entities(
    main_past: pd.DataFrame,
    trackman_past: pd.DataFrame,
    seasons: Sequence[int],
    thresholds: ResolutionThresholds,
    previous_top1: Mapping[int, int] | None = None,
) -> ResolutionResult:
    """Recover high-confidence, strictly-past, one-to-one pitcher links."""
    main_fp = _main_fingerprint(
        main_past,
        seasons,
    )
    trackman_fp = _trackman_fingerprint(
        trackman_past,
        seasons,
    )

    if (
        main_fp.empty
        or len(trackman_fp) < 2
    ):
        return ResolutionResult(
            mapping={},
            top1={},
            team_map={},
            audit={
                "main_pitchers": int(
                    len(main_fp)
                ),
                "trackman_pitchers": int(
                    len(trackman_fp)
                ),
                "accepted_candidate_count": 0,
                "unique_trackman_candidate_count": 0,
                "collided_trackman_ids": [],
                "collision_excess": 0,
                "collision_count": 0,
                "collision_rate": 0.0,
                "matched_pitchers": 0,
                "matched_trackman_pitchers": 0,
                "final_mapping_unique_trackman_count": 0,
                "final_one_to_one": True,
                "row_coverage": 0.0,
                "team_map": {},
            },
        )

    base_distance = _distance_matrix(
        main_fp,
        trackman_fp,
        seasons,
        team_map=None,
        team_penalty=0.0,
    )

    team_map = _infer_team_map(
        main_fp,
        trackman_fp,
        base_distance,
        thresholds,
    )

    distance = _distance_matrix(
        main_fp,
        trackman_fp,
        seasons,
        team_map=team_map,
        team_penalty=thresholds.team_penalty,
    )

    order = np.argsort(
        distance,
        axis=1,
    )[:, :2]

    first = distance[
        np.arange(len(main_fp)),
        order[:, 0],
    ]
    second = distance[
        np.arange(len(main_fp)),
        order[:, 1],
    ]

    margins = second / (
        first + 1.0e-9
    )

    top1 = {
        int(main_id): int(
            trackman_fp.index[
                order[row_pos, 0]
            ]
        )
        for row_pos, main_id
        in enumerate(main_fp.index)
    }

    candidates = []
    previous_top1 = (
        previous_top1
        or {}
    )

    for row_pos, (
        main_id,
        row,
    ) in enumerate(
        main_fp.iterrows()
    ):
        candidate = trackman_fp.iloc[
            order[row_pos, 0]
        ]

        trackman_id = int(
            candidate.name
        )

        count_ratio = (
            float(candidate["n"])
            / max(
                float(row["n"]),
                1.0,
            )
        )

        expected_team = team_map.get(
            int(row["team"])
        )
        team_available = (
            expected_team is not None
        )
        team_agreement = bool(
            team_available
            and expected_team
            in tuple(
                candidate[
                    "team_tokens"
                ]
            )
        )

        stable = (
            int(main_id)
            not in previous_top1
            or int(
                previous_top1[
                    int(main_id)
                ]
            )
            == trackman_id
        )

        strong = (
            margins[row_pos]
            >= thresholds.strong_margin_ratio
            and first[row_pos]
            <= (
                0.60
                * thresholds.max_distance
            )
        )

        # Team is intentionally NOT a hard acceptance condition.
        accepted = (
            float(row["n"])
            >= thresholds.min_pitches
            and float(candidate["n"])
            >= thresholds.min_pitches
            and (
                thresholds.min_count_ratio
                <= count_ratio
                <= thresholds.max_count_ratio
            )
            and first[row_pos]
            <= thresholds.max_distance
            and margins[row_pos]
            >= thresholds.min_margin_ratio
            and (
                stable
                or strong
            )
        )

        if not accepted:
            continue

        confidence = float(
            np.clip(
                1.0
                - first[row_pos]
                / thresholds.max_distance,
                0.0,
                1.0,
            )
            * np.clip(
                1.0
                - thresholds.min_margin_ratio
                / max(
                    margins[row_pos],
                    1.0e-9,
                ),
                0.0,
                1.0,
            )
        )

        rank_score = float(
            first[row_pos]
            / max(
                margins[row_pos],
                1.0e-9,
            )
        )

        candidates.append(
            (
                rank_score,
                int(main_id),
                trackman_id,
                {
                    "trackman_id": (
                        trackman_id
                    ),
                    "distance": float(
                        first[row_pos]
                    ),
                    "margin_ratio": float(
                        margins[row_pos]
                    ),
                    "confidence": confidence,
                    "main_log_n": float(
                        np.log1p(
                            row["n"]
                        )
                    ),
                    "trackman_log_n": float(
                        np.log1p(
                            candidate["n"]
                        )
                    ),
                    "count_ratio": (
                        count_ratio
                    ),
                    "team_available": float(
                        team_available
                    ),
                    "team_agreement": float(
                        team_agreement
                    ),
                    "stable": float(
                        stable
                    ),
                },
            )
        )

    # --------------------------------------------------------
    # PRE-DEDUP collision audit.
    # --------------------------------------------------------
    candidate_counts = Counter(
        trackman_id
        for (
            _,
            _,
            trackman_id,
            _,
        )
        in candidates
    )

    collided_trackman_ids = sorted(
        int(trackman_id)
        for (
            trackman_id,
            count,
        )
        in candidate_counts.items()
        if count > 1
    )

    collision_excess = int(
        sum(
            max(
                int(count) - 1,
                0,
            )
            for count
            in candidate_counts.values()
        )
    )

    accepted_candidate_count = int(
        len(candidates)
    )

    # --------------------------------------------------------
    # Deterministic one-to-one dedup.
    # --------------------------------------------------------
    mapping: Dict[
        int,
        Dict[str, float | int],
    ] = {}

    used_trackman_ids: set[int] = set()

    for (
        _,
        main_id,
        trackman_id,
        values,
    ) in sorted(candidates):
        if (
            trackman_id
            in used_trackman_ids
        ):
            continue

        mapping[
            main_id
        ] = values
        used_trackman_ids.add(
            trackman_id
        )

    final_trackman_ids = [
        int(
            values["trackman_id"]
        )
        for values
        in mapping.values()
    ]

    final_one_to_one = (
        len(final_trackman_ids)
        == len(
            set(final_trackman_ids)
        )
    )

    matched_rows = (
        float(
            main_fp.loc[
                list(mapping),
                "n",
            ].sum()
        )
        if mapping
        else 0.0
    )

    total_rows = float(
        main_fp["n"].sum()
    )

    audit = {
        "main_pitchers": int(
            len(main_fp)
        ),
        "trackman_pitchers": int(
            len(trackman_fp)
        ),

        # Before dedup.
        "accepted_candidate_count": (
            accepted_candidate_count
        ),
        "unique_trackman_candidate_count": int(
            len(candidate_counts)
        ),
        "collided_trackman_ids": (
            collided_trackman_ids
        ),
        "collision_excess": (
            collision_excess
        ),

        # Backward-compatible name, now meaningful.
        "collision_count": (
            collision_excess
        ),
        "collision_rate": (
            float(
                collision_excess
            )
            / max(
                float(
                    accepted_candidate_count
                ),
                1.0,
            )
        ),

        # After dedup.
        "matched_pitchers": int(
            len(mapping)
        ),
        "matched_trackman_pitchers": int(
            len(used_trackman_ids)
        ),
        "final_mapping_unique_trackman_count": int(
            len(
                set(
                    final_trackman_ids
                )
            )
        ),
        "final_one_to_one": bool(
            final_one_to_one
        ),

        "row_coverage": (
            matched_rows
            / max(
                total_rows,
                1.0,
            )
        ),
        "team_map": {
            int(key): value
            for key, value
            in sorted(
                team_map.items()
            )
        },
    }

    return ResolutionResult(
        mapping=mapping,
        top1=top1,
        team_map=team_map,
        audit=audit,
    )
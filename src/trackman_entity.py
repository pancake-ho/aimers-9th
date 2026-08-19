from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


PITCH_GROUPS = ("fastball", "breaking", "offspeed")
HAND_ALIASES = {"Left": 1, "Right": 2, "L": 1, "R": 2, "1": 1, "2": 2}

# Trackman contains first- and minor-team aliases.  These aliases are not an
# external player lookup: they are normalized versions of official values in
# trackman_history.csv and are used only to learn the numeric main-team
# correspondence from the competition files themselves.
TRACKMAN_ORG_ALIASES = {
    "DOO_BEA": "DOO",
    "MIN_DOO": "DOO",
    "HAN_EAG": "HAN",
    "MIN_HAN": "HAN",
    "KIA_TIG": "KIA",
    "MIN_KIA": "KIA",
    "KIW_HER": "KIW",
    "MIN_HER": "KIW",
    "KT_WIZ": "KT",
    "MIN_KTW": "KT",
    "LG_TWI": "LG",
    "MIN_LGT": "LG",
    "LOT_GIA": "LOT",
    "MIN_LOT": "LOT",
    "NC_DIN": "NC",
    "MIN_NCD": "NC",
    "SAM_LIO": "SAM",
    "MIN_SAM": "SAM",
    "SK_WYV": "SSG",
    "SSG_LAN": "SSG",
    "MIN_SKW": "SSG",
    "MIN_SSG": "SSG",
}
PRIMARY_ORGS = frozenset(TRACKMAN_ORG_ALIASES.values())


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


def canonical_trackman_org(value: object) -> str:
    return TRACKMAN_ORG_ALIASES.get(str(value), "OTHER")


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
        frame["pitcher_trackman_id"], errors="coerce"
    )
    mapped_hand = frame["pitcher_hand"].map(HAND_ALIASES)
    frame["pitcher_hand"] = mapped_hand.fillna(
        pd.to_numeric(frame["pitcher_hand"], errors="coerce")
    )
    frame = frame.dropna(subset=["pitcher_trackman_id", "pitcher_hand"])
    frame["pitcher_trackman_id"] = frame["pitcher_trackman_id"].astype(np.int64)
    frame["org"] = frame["pitcher_team"].map(canonical_trackman_org)

    grouped = frame.groupby("pitcher_trackman_id", observed=True, sort=False)
    result = pd.DataFrame(
        {
            "n": grouped.size().astype(np.int64),
            "hand": grouped["pitcher_hand"].agg(_mode_as_int),
            "orgs": grouped["org"].agg(
                lambda values: tuple(sorted(set(values) - {"OTHER"}))
            ),
        }
    )
    counts = pd.crosstab(frame["pitcher_trackman_id"], frame["pitch_type_group"])
    counts = counts.reindex(columns=list(PITCH_GROUPS), fill_value=0)
    for group in PITCH_GROUPS:
        result[group] = counts[group].reindex(result.index, fill_value=0) / result["n"]

    seasonal = (
        frame.groupby(["pitcher_trackman_id", "season"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(seasons), fill_value=0)
    )
    for season in seasons:
        result[f"n_{season}"] = seasonal[season].reindex(result.index, fill_value=0)
    return result


def _distance_matrix(
    main_fp: pd.DataFrame,
    trackman_fp: pd.DataFrame,
    seasons: Sequence[int],
    team_map: Mapping[int, str] | None,
    team_penalty: float,
) -> np.ndarray:
    # Scales were fixed from the semantics of each fingerprint component, not
    # fitted against control_success.  A distance of roughly 1 corresponds to
    # an 18% log-count, six-point repertoire, or ten-point seasonal-usage gap.
    result = np.full((len(main_fp), len(trackman_fp)), 1.0e6, dtype=np.float64)
    trackman_hand = trackman_fp["hand"].to_numpy(dtype=np.int64, copy=False)
    season_cols = [f"n_{season}" for season in seasons]

    for row_pos, (_, row) in enumerate(main_fp.iterrows()):
        valid = trackman_hand == int(row["hand"])
        candidate_pos = np.flatnonzero(valid)
        if candidate_pos.size == 0:
            continue
        candidates = trackman_fp.iloc[candidate_pos]

        count_distance = np.square(
            (
                np.log1p(candidates["n"].to_numpy(dtype=np.float64))
                - np.log1p(float(row["n"]))
            )
            / 0.18
        )
        mix_distance = np.square(
            (
                candidates[list(PITCH_GROUPS)].to_numpy(dtype=np.float64)
                - row[list(PITCH_GROUPS)].to_numpy(dtype=np.float64)
            )
            / 0.06
        ).sum(axis=1)

        main_season = row[season_cols].to_numpy(dtype=np.float64)
        main_season /= main_season.sum() + 1.0e-12
        trackman_season = candidates[season_cols].to_numpy(dtype=np.float64)
        trackman_season /= trackman_season.sum(axis=1, keepdims=True) + 1.0e-12
        season_distance = np.square(
            (trackman_season - main_season) / 0.10
        ).sum(axis=1)
        distance = count_distance + mix_distance + season_distance

        expected_org = None if team_map is None else team_map.get(int(row["team"]))
        if expected_org is not None:
            distance += np.asarray(
                [0.0 if expected_org in orgs else float(team_penalty) for orgs in candidates["orgs"]],
                dtype=np.float64,
            )
        result[row_pos, candidate_pos] = distance
    return result


def _infer_team_map(
    main_fp: pd.DataFrame,
    trackman_fp: pd.DataFrame,
    base_distance: np.ndarray,
    thresholds: ResolutionThresholds,
) -> Dict[int, str]:
    if len(trackman_fp) < 2:
        return {}
    order = np.argsort(base_distance, axis=1)[:, :2]
    first = base_distance[np.arange(len(main_fp)), order[:, 0]]
    second = base_distance[np.arange(len(main_fp)), order[:, 1]]
    margins = second / (first + 1.0e-9)
    support: Dict[tuple[int, str], int] = {}

    for row_pos, (_, row) in enumerate(main_fp.iterrows()):
        candidate = trackman_fp.iloc[order[row_pos, 0]]
        count_ratio = float(candidate["n"]) / max(float(row["n"]), 1.0)
        orgs = tuple(candidate["orgs"])
        if not (
            float(row["n"]) >= 1000
            and float(candidate["n"]) >= 1000
            and margins[row_pos] >= 1.5
            and first[row_pos] <= 8.0
            and 0.60 <= count_ratio <= 1.70
            and len(orgs) == 1
        ):
            continue
        key = (int(row["team"]), str(orgs[0]))
        support[key] = support.get(key, 0) + 1

    team_map: Dict[int, str] = {}
    for main_team in sorted({key[0] for key in support}):
        candidates = sorted(
            (
                (count, org)
                for (team, org), count in support.items()
                if team == main_team
            ),
            reverse=True,
        )
        total = sum(count for count, _ in candidates)
        if (
            candidates
            and candidates[0][0] >= thresholds.team_min_support
            and candidates[0][0] / max(total, 1) >= thresholds.team_min_dominance
        ):
            team_map[main_team] = candidates[0][1]

    # The ten primary organizations occur in both official files.  When nine
    # mappings are unambiguous, recover the last one by one-to-one elimination
    # instead of hard-coding a numeric main-team dictionary.
    team_activity = main_fp.groupby("team", observed=True)["n"].sum().sort_values(ascending=False)
    primary_main_teams = {int(team) for team in team_activity.head(10).index}
    missing_main = primary_main_teams - set(team_map)
    missing_org = set(PRIMARY_ORGS) - set(team_map.values())
    if len(missing_main) == 1 and len(missing_org) == 1:
        team_map[next(iter(missing_main))] = next(iter(missing_org))
    return team_map


def resolve_pitcher_entities(
    main_past: pd.DataFrame,
    trackman_past: pd.DataFrame,
    seasons: Sequence[int],
    thresholds: ResolutionThresholds,
    previous_top1: Mapping[int, int] | None = None,
) -> ResolutionResult:
    """Recover only high-confidence, strictly-past one-to-one pitcher links."""
    main_fp = _main_fingerprint(main_past, seasons)
    trackman_fp = _trackman_fingerprint(trackman_past, seasons)
    if main_fp.empty or len(trackman_fp) < 2:
        return ResolutionResult(
            mapping={},
            top1={},
            team_map={},
            audit={"matched_pitchers": 0, "row_coverage": 0.0},
        )

    base_distance = _distance_matrix(
        main_fp, trackman_fp, seasons, team_map=None, team_penalty=0.0
    )
    team_map = _infer_team_map(main_fp, trackman_fp, base_distance, thresholds)
    distance = _distance_matrix(
        main_fp,
        trackman_fp,
        seasons,
        team_map=team_map,
        team_penalty=thresholds.team_penalty,
    )
    order = np.argsort(distance, axis=1)[:, :2]
    first = distance[np.arange(len(main_fp)), order[:, 0]]
    second = distance[np.arange(len(main_fp)), order[:, 1]]
    margins = second / (first + 1.0e-9)
    top1 = {
        int(main_id): int(trackman_fp.index[order[row_pos, 0]])
        for row_pos, main_id in enumerate(main_fp.index)
    }

    candidates = []
    previous_top1 = previous_top1 or {}
    for row_pos, (main_id, row) in enumerate(main_fp.iterrows()):
        candidate = trackman_fp.iloc[order[row_pos, 0]]
        trackman_id = int(candidate.name)
        count_ratio = float(candidate["n"]) / max(float(row["n"]), 1.0)
        expected_org = team_map.get(int(row["team"]))
        team_agreement = expected_org is None or expected_org in candidate["orgs"]
        stable = (
            int(main_id) not in previous_top1
            or int(previous_top1[int(main_id)]) == trackman_id
        )
        strong = (
            margins[row_pos] >= thresholds.strong_margin_ratio
            and first[row_pos] <= 0.60 * thresholds.max_distance
        )
        accepted = (
            float(row["n"]) >= thresholds.min_pitches
            and float(candidate["n"]) >= thresholds.min_pitches
            and thresholds.min_count_ratio <= count_ratio <= thresholds.max_count_ratio
            and first[row_pos] <= thresholds.max_distance
            and margins[row_pos] >= thresholds.min_margin_ratio
            and team_agreement
            and (stable or strong)
        )
        if not accepted:
            continue
        confidence = float(
            np.clip(1.0 - first[row_pos] / thresholds.max_distance, 0.0, 1.0)
            * np.clip(
                1.0
                - thresholds.min_margin_ratio / max(margins[row_pos], 1.0e-9),
                0.0,
                1.0,
            )
        )
        # Lower is better.  Sorting before collision removal makes the
        # one-to-one decision deterministic and keeps the clearer identity.
        rank_score = float(first[row_pos] / max(margins[row_pos], 1.0e-9))
        candidates.append(
            (
                rank_score,
                int(main_id),
                trackman_id,
                {
                    "trackman_id": trackman_id,
                    "distance": float(first[row_pos]),
                    "margin_ratio": float(margins[row_pos]),
                    "confidence": confidence,
                    "main_log_n": float(np.log1p(row["n"])),
                    "trackman_log_n": float(np.log1p(candidate["n"])),
                    "count_ratio": count_ratio,
                    "team_agreement": float(team_agreement),
                    "stable": float(stable),
                },
            )
        )

    mapping: Dict[int, Dict[str, float | int]] = {}
    used_trackman_ids: set[int] = set()
    for _, main_id, trackman_id, values in sorted(candidates):
        if trackman_id in used_trackman_ids:
            continue
        mapping[main_id] = values
        used_trackman_ids.add(trackman_id)

    matched_rows = float(main_fp.loc[list(mapping), "n"].sum()) if mapping else 0.0
    total_rows = float(main_fp["n"].sum())
    audit = {
        "main_pitchers": int(len(main_fp)),
        "trackman_pitchers": int(len(trackman_fp)),
        "matched_pitchers": int(len(mapping)),
        "matched_trackman_pitchers": int(len(used_trackman_ids)),
        "collision_count": int(len(mapping) - len(used_trackman_ids)),
        "row_coverage": matched_rows / max(total_rows, 1.0),
        "team_map": {int(key): value for key, value in sorted(team_map.items())},
    }
    return ResolutionResult(
        mapping=mapping,
        top1=top1,
        team_map=team_map,
        audit=audit,
    )

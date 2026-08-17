from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FeatureConfig

PITCHER_RATE_COLS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)

BATTER_RATE_COLS = (
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
)

TRACKMAN_MEAN_METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
)

TRACKMAN_STD_METRICS = (
    "rel_speed",
    "spin_rate",
)


class StrictPastTrackmanFeatures:
    """Build season-level, strictly-past Trackman summaries.

    A row from season S receives Trackman statistics computed only from
    seasons < S. The main-table and Trackman player IDs are not assumed to
    correspond, so the first submission uses only pitcher-hand x batter-hand
    profiles. Test rows never participate in fitting.
    """

    HAND_MAP = {
        "Left": 1,
        "Right": 2,
        "L": 1,
        "R": 2,
        1: 1,
        2: 2,
        "1": 1,
        "2": 2,
    }

    def __init__(self) -> None:
        self.lookup_: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self.feature_names_: list[str] = [
            *(f"tm_past_{m}_mean" for m in TRACKMAN_MEAN_METRICS),
            *(f"tm_past_{m}_std" for m in TRACKMAN_STD_METRICS),
            "tm_past_n",
            "tm_past_available",
        ]

    def fit_from_csv(self, trackman_path) -> "StrictPastTrackmanFeatures":
        print("[FEATURE] Building strictly-past Trackman features...")

        usecols = [
            "season",
            "pitcher_hand",
            "batter_hand",
            *TRACKMAN_MEAN_METRICS,
        ]
        tm = pd.read_csv(trackman_path, usecols=usecols, low_memory=False)
        tm["pitcher_hand"] = tm["pitcher_hand"].map(self.HAND_MAP)
        tm["batter_hand"] = tm["batter_hand"].map(self.HAND_MAP)
        tm = tm.dropna(subset=["season", "pitcher_hand", "batter_hand"])
        tm["season"] = tm["season"].astype(np.int16)
        tm["pitcher_hand"] = tm["pitcher_hand"].astype(np.int8)
        tm["batter_hand"] = tm["batter_hand"].astype(np.int8)

        for metric in TRACKMAN_MEAN_METRICS:
            tm[metric] = pd.to_numeric(tm[metric], errors="coerce").astype(np.float32)

        min_season = int(tm["season"].min())
        max_season = int(tm["season"].max())
        self.lookup_.clear()

        for target_season in range(min_season, max_season + 2):
            past = tm.loc[tm["season"] < target_season]
            if past.empty:
                continue

            grouped = past.groupby(["pitcher_hand", "batter_hand"], observed=True)
            means = grouped[list(TRACKMAN_MEAN_METRICS)].mean()
            stds = grouped[list(TRACKMAN_STD_METRICS)].std()
            counts = grouped.size()

            for ph, bh in means.index:
                key = (int(target_season), int(ph), int(bh))
                values: Dict[str, float] = {}
                for metric in TRACKMAN_MEAN_METRICS:
                    value = means.loc[(ph, bh), metric]
                    values[f"tm_past_{metric}_mean"] = (
                        float(value) if np.isfinite(value) else np.nan
                    )
                for metric in TRACKMAN_STD_METRICS:
                    value = stds.loc[(ph, bh), metric]
                    values[f"tm_past_{metric}_std"] = (
                        float(value) if np.isfinite(value) else np.nan
                    )
                values["tm_past_n"] = float(counts.loc[(ph, bh)])
                values["tm_past_available"] = 1.0
                self.lookup_[key] = values

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        result = {
            col: np.full(n, np.nan, dtype=np.float32)
            for col in self.feature_names_
        }
        result["tm_past_n"][:] = 0.0
        result["tm_past_available"][:] = 0.0

        if not self.lookup_ or n == 0:
            return pd.DataFrame(result, index=df.index)

        seasons = pd.to_numeric(df["season"], errors="coerce").fillna(-1).astype(int).to_numpy()
        ph = (
            df["pitcher_hand"]
            .map(self.HAND_MAP)
            .fillna(pd.to_numeric(df["pitcher_hand"], errors="coerce"))
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )
        bh = (
            df["batter_hand"]
            .map(self.HAND_MAP)
            .fillna(pd.to_numeric(df["batter_hand"], errors="coerce"))
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )

        for i, key in enumerate(zip(seasons, ph, bh)):
            stats = self.lookup_.get((int(key[0]), int(key[1]), int(key[2])))
            if stats is None:
                continue
            for col, value in stats.items():
                result[col][i] = value

        return pd.DataFrame(result, index=df.index)

    def export_state(self) -> Dict[str, object]:
        return {
            "lookup": self.lookup_,
            "feature_names": list(self.feature_names_),
        }


class LeakageSafeFeatureEngineer:
    """Row-time-safe features for validation and final inference.

    There is intentionally no target-dependent fitting step. In particular:
    - no KMeans cluster made from a pitcher's future rows;
    - no target-derived global mean used as a smoothing prior;
    - no validation/test-distribution normalization;
    - no rolling or aggregation across test rows.
    """

    def __init__(
        self,
        config: FeatureConfig,
        trackman_features: Optional[StrictPastTrackmanFeatures] = None,
    ) -> None:
        self.config = config
        self.trackman_features = trackman_features

    def fit(self, df: pd.DataFrame, y=None) -> "LeakageSafeFeatureEngineer":
        # Compatibility with the existing temporal runner.
        # Intentionally no target-dependent or future-row fitting occurs.
        return self

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        prior = float(self.config.smoothing_prior)

        # A. Count state
        out["count_state"] = (
            out["balls_before"].fillna(-1).astype(str)
            + "-"
            + out["strikes_before"].fillna(-1).astype(str)
        )
        out["is_hitter_count"] = (
            (out["balls_before"] >= 2) & (out["strikes_before"] <= 1)
        ).astype(np.int8)
        out["is_pitcher_count"] = (
            (out["strikes_before"] == 2) & (out["balls_before"] <= 1)
        ).astype(np.int8)
        out["is_full_count"] = (
            (out["balls_before"] == 3) & (out["strikes_before"] == 2)
        ).astype(np.int8)
        out["is_two_strike"] = (out["strikes_before"] == 2).astype(np.int8)
        out["is_three_ball"] = (out["balls_before"] == 3).astype(np.int8)
        out["count_leverage"] = (
            out["balls_before"].fillna(0) - out["strikes_before"].fillna(0)
        ).astype(np.int8)

        # B. Runner / game state
        out["is_scoring_position"] = (
            (out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)
        ).astype(np.int8)
        out["is_bases_loaded"] = (out["num_runners_on"] == 3).astype(np.int8)
        out["abs_score_diff"] = out["score_diff_pitcher_team"].abs()
        out["is_close_game"] = (out["abs_score_diff"] <= 2).astype(np.int8)
        out["is_late_game"] = (out["inning"] >= 7).astype(np.int8)
        out["li_log"] = np.log1p(out["li"].clip(lower=0))
        out["is_high_leverage"] = (out["li"] >= 2.0).astype(np.int8)
        out["is_clutch"] = (
            (out["inning"] >= 7)
            & (out["abs_score_diff"] <= 2)
            & (out["li"] > 1.2)
        ).astype(np.int8)
        out["win_expectancy_diff"] = (
            out["home_win_expectancy"] - out["away_win_expectancy"]
        )
        out["stadium_owner_team"] = np.where(
            out["top_bottom"].eq("T"),
            out["pitcher_team_id"],
            out["batter_team_id"],
        )
        out["same_hand"] = (
            out["pitcher_hand"].astype(str) == out["batter_hand"].astype(str)
        ).astype(np.int8)

        # C. Fixed-prior smoothing; no target labels are used here.
        pitcher_n = pd.to_numeric(out["asof_pitcher_n"], errors="coerce").fillna(0).astype(np.float32)
        for col in PITCHER_RATE_COLS:
            raw = pd.to_numeric(out[col], errors="coerce").fillna(prior).astype(np.float32)
            strength = float(self.config.pitcher_prior_strength)
            out[f"{col}_smoothed"] = (
                (pitcher_n * raw + strength * prior) / (pitcher_n + strength)
            ).astype(np.float32)

        batter_n = pd.to_numeric(out["asof_batter_n"], errors="coerce").fillna(0).astype(np.float32)
        for col in BATTER_RATE_COLS:
            raw = pd.to_numeric(out[col], errors="coerce").fillna(prior).astype(np.float32)
            strength = float(self.config.batter_prior_strength)
            out[f"{col}_smoothed"] = (
                (batter_n * raw + strength * prior) / (batter_n + strength)
            ).astype(np.float32)

        out["pitcher_history_log_n"] = np.log1p(out["asof_pitcher_n"].fillna(0))
        out["batter_history_log_n"] = np.log1p(out["asof_batter_n"].fillna(0))
        out["pitchmix_history_log_n"] = np.log1p(out["asof_pitcher_pitchmix_n"].fillna(0))
        out["pitcher_cold_start"] = (out["asof_pitcher_n"].fillna(0) < 50).astype(np.int8)
        out["batter_cold_start"] = (out["asof_batter_n"].fillna(0) < 50).astype(np.int8)

        # D. Recent form vs long-term baseline
        baseline_success = out["asof_pitcher_success_rate_smoothed"]
        baseline_middle = out["asof_pitcher_middle_rate_smoothed"]
        for k in (1, 3, 5):
            out[f"diff_prev{k}_success"] = (
                out[f"asof_pitcher_prev{k}_game_success_rate"] - baseline_success
            )
            out[f"diff_prev{k}_middle"] = (
                out[f"asof_pitcher_prev{k}_game_middle_rate"] - baseline_middle
            )

        out["pitcher_volatility"] = (
            out["diff_prev1_success"].abs() + out["diff_prev3_success"].abs()
        )
        out["pressure_interaction"] = out["pitcher_volatility"] * out["li_log"]
        out["recent_form_leverage"] = out["diff_prev3_success"] * out["li_log"]

        # E. Control profile ratios
        eps = 1e-5
        out["strike_to_ball_ratio"] = (
            out["asof_pitcher_strike_rate_smoothed"]
            / (out["asof_pitcher_ball_rate_smoothed"] + eps)
        )
        out["reverse_to_success_ratio"] = (
            out["asof_pitcher_reverse_rate_smoothed"]
            / (out["asof_pitcher_success_rate_smoothed"] + eps)
        )
        out["middle_to_success_ratio"] = (
            out["asof_pitcher_middle_rate_smoothed"]
            / (out["asof_pitcher_success_rate_smoothed"] + eps)
        )

        # F. Pitch mix
        pitchmix_cols = [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
        p = out[pitchmix_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        p_entropy = p.clip(lower=eps, upper=1.0)
        out["pitchmix_entropy"] = -(p_entropy * np.log(p_entropy)).sum(axis=1)
        out["pitchmix_max_share"] = p.max(axis=1)

        # G. Strict-past Trackman
        if self.trackman_features is not None:
            tm = self.trackman_features.transform(out)
            for col in tm.columns:
                out[col] = tm[col].to_numpy(copy=False)

        return out

    def export_runtime_state(self) -> Dict[str, object]:
        return {
            "smoothing_prior": float(self.config.smoothing_prior),
            "pitcher_prior_strength": float(self.config.pitcher_prior_strength),
            "batter_prior_strength": float(self.config.batter_prior_strength),
            "trackman": (
                None
                if self.trackman_features is None
                else self.trackman_features.export_state()
            ),
        }

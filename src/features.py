# src/features.py

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

PITCHER_CLUSTER_COLS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_strike_rate",
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


def _safe_std(x: float) -> float:
    if not np.isfinite(x) or x < 1e-8:
        return 1.0
    return float(x)


class StrictPastTrackmanFeatures:
    """
    Strictly-past Trackman aggregation.

    For a main-table row from season S,
    only Trackman seasons < S are used.

    Therefore:
        2023 validation -> Trackman <= 2022
        2024 validation -> Trackman <= 2023
        2025 test       -> Trackman <= 2024

    No test rows are used.
    No current-season Trackman information is used.

    Main/Trackman player IDs are not assumed to map 1:1,
    so this first implementation only uses
    pitcher_hand x batter_hand historical profiles.
    """

    HAND_MAP = {
        "Left": 1,
        "Right": 2,
        "L": 1,
        "R": 2,
        1: 1,
        2: 2,
    }

    def __init__(self):
        self.lookup_: Dict[
            Tuple[int, int, int],
            Dict[str, float],
        ] = {}

        self.feature_names_: list[str] = []

    def fit_from_csv(
        self,
        trackman_path,
    ) -> "StrictPastTrackmanFeatures":

        print(
            "[FEATURE] Building strictly-past "
            "Trackman features..."
        )

        usecols = [
            "season",
            "pitcher_hand",
            "batter_hand",
            *TRACKMAN_MEAN_METRICS,
        ]

        tm = pd.read_csv(
            trackman_path,
            usecols=usecols,
            low_memory=False,
        )

        tm["pitcher_hand"] = (
            tm["pitcher_hand"]
            .map(self.HAND_MAP)
        )

        tm["batter_hand"] = (
            tm["batter_hand"]
            .map(self.HAND_MAP)
        )

        tm = tm.dropna(
            subset=[
                "season",
                "pitcher_hand",
                "batter_hand",
            ]
        )

        tm["season"] = (
            tm["season"]
            .astype(np.int16)
        )

        tm["pitcher_hand"] = (
            tm["pitcher_hand"]
            .astype(np.int8)
        )

        tm["batter_hand"] = (
            tm["batter_hand"]
            .astype(np.int8)
        )

        for metric in TRACKMAN_MEAN_METRICS:
            tm[metric] = pd.to_numeric(
                tm[metric],
                errors="coerce",
            )

        target_seasons = range(
            int(tm["season"].min()),
            int(tm["season"].max()) + 2,
        )

        self.feature_names_ = []

        for metric in TRACKMAN_MEAN_METRICS:
            self.feature_names_.append(
                f"tm_past_{metric}_mean"
            )

        for metric in TRACKMAN_STD_METRICS:
            self.feature_names_.append(
                f"tm_past_{metric}_std"
            )

        self.feature_names_.append(
            "tm_past_n"
        )

        self.feature_names_.append(
            "tm_past_available"
        )

        # Only 7 target seasons exist (2019~2025),
        # so looping over cutoff seasons is cheap and clear.
        for target_season in target_seasons:

            past = tm[
                tm["season"] < target_season
            ]

            if past.empty:
                continue

            grouped = past.groupby(
                [
                    "pitcher_hand",
                    "batter_hand",
                ],
                observed=True,
            )

            mean_df = grouped[
                list(TRACKMAN_MEAN_METRICS)
            ].mean()

            std_df = grouped[
                list(TRACKMAN_STD_METRICS)
            ].std()

            count_series = grouped.size()

            for key in mean_df.index:
                ph, bh = key

                values: Dict[str, float] = {}

                for metric in TRACKMAN_MEAN_METRICS:
                    value = mean_df.loc[
                        key,
                        metric,
                    ]

                    values[
                        f"tm_past_{metric}_mean"
                    ] = (
                        float(value)
                        if np.isfinite(value)
                        else np.nan
                    )

                for metric in TRACKMAN_STD_METRICS:
                    value = std_df.loc[
                        key,
                        metric,
                    ]

                    values[
                        f"tm_past_{metric}_std"
                    ] = (
                        float(value)
                        if np.isfinite(value)
                        else np.nan
                    )

                values["tm_past_n"] = float(
                    count_series.loc[key]
                )

                values[
                    "tm_past_available"
                ] = 1.0

                self.lookup_[
                    (
                        int(target_season),
                        int(ph),
                        int(bh),
                    )
                ] = values

        del tm

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = pd.DataFrame(
            index=df.index
        )

        if not self.lookup_:
            for col in self.feature_names_:
                out[col] = 0.0
            return out

        seasons = (
            df["season"]
            .astype(int)
            .to_numpy()
        )

        pitcher_hands = (
            df["pitcher_hand"]
            .map(self.HAND_MAP)
            .fillna(df["pitcher_hand"])
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )

        batter_hands = (
            df["batter_hand"]
            .map(self.HAND_MAP)
            .fillna(df["batter_hand"])
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )

        values_by_col = {
            c: np.full(
                len(df),
                np.nan,
                dtype=np.float32,
            )
            for c in self.feature_names_
        }

        values_by_col[
            "tm_past_available"
        ][:] = 0.0

        values_by_col[
            "tm_past_n"
        ][:] = 0.0

        for i, (season, ph, bh) in enumerate(
            zip(
                seasons,
                pitcher_hands,
                batter_hands,
            )
        ):
            stats = self.lookup_.get(
                (
                    int(season),
                    int(ph),
                    int(bh),
                )
            )

            if stats is None:
                continue

            for col, value in stats.items():
                values_by_col[col][i] = value

        for col, values in values_by_col.items():
            out[col] = values

        return out


class LeakageSafeFeatureEngineer:
    """
    Any statistic that needs fitting is learned ONLY
    from the training side of the temporal fold.
    """

    def __init__(
        self,
        config: FeatureConfig,
        trackman_features: Optional[
            StrictPastTrackmanFeatures
        ] = None,
    ):
        self.config = config
        self.trackman_features = (
            trackman_features
        )

        self.global_mean_: Optional[
            float
        ] = None

        self.kmeans_: Optional[
            KMeans
        ] = None

        self.pitcher_cluster_map_: Dict[
            int,
            int,
        ] = {}

        self.zscore_stats_: Dict[
            Tuple[str, str],
            Tuple[float, float],
        ] = {}

        self.latest_train_season_: Optional[
            int
        ] = None

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
    ) -> "LeakageSafeFeatureEngineer":

        self.global_mean_ = float(
            np.asarray(y).mean()
        )

        self.latest_train_season_ = int(
            df["season"].max()
        )

        self._fit_pitcher_clusters(df)

        base = self._build_base_features(
            df,
            fit_mode=True,
        )

        self._fit_zscore_stats(base)

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.global_mean_ is None:
            raise RuntimeError(
                "FeatureEngineer must be fitted "
                "before transform()."
            )

        out = self._build_base_features(
            df,
            fit_mode=False,
        )

        out = self._apply_zscores(out)

        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:

        self.fit(df, y)
        return self.transform(df)

    def _fit_pitcher_clusters(
        self,
        df: pd.DataFrame,
    ) -> None:

        sort_cols = [
            "pitcher_id",
            "season",
            "game_month",
            "inning",
        ]

        latest = (
            df.sort_values(
                sort_cols,
                kind="mergesort",
            )
            .drop_duplicates(
                "pitcher_id",
                keep="last",
            )
            .copy()
        )

        cluster_x = (
            latest[
                list(
                    PITCHER_CLUSTER_COLS
                )
            ]
            .fillna(0.5)
            .astype(np.float32)
        )

        n_pitchers = len(latest)

        if n_pitchers < 2:
            self.pitcher_cluster_map_ = {}
            return

        n_clusters = min(
            self.config.pitcher_cluster_k,
            n_pitchers,
        )

        self.kmeans_ = KMeans(
            n_clusters=n_clusters,
            random_state=(
                self.config.random_seed
            ),
            n_init=10,
        )

        labels = self.kmeans_.fit_predict(
            cluster_x
        )

        self.pitcher_cluster_map_ = dict(
            zip(
                latest[
                    "pitcher_id"
                ].astype(int),
                labels.astype(int),
            )
        )

    def _build_base_features(
        self,
        df: pd.DataFrame,
        fit_mode: bool,
    ) -> pd.DataFrame:

        out = df.copy()

        global_mean = float(
            self.global_mean_
        )

        # -----------------------------
        # A. Basic game-state features
        # -----------------------------

        out["count_state"] = (
            out["balls_before"].astype(str)
            + "-"
            + out[
                "strikes_before"
            ].astype(str)
        )

        out["is_hitter_count"] = (
            (
                out["balls_before"] >= 2
            )
            & (
                out["strikes_before"] <= 1
            )
        ).astype(np.int8)

        out["is_pitcher_count"] = (
            (
                out["strikes_before"] == 2
            )
            & (
                out["balls_before"] <= 1
            )
        ).astype(np.int8)

        out["is_full_count"] = (
            (
                out["balls_before"] == 3
            )
            & (
                out["strikes_before"] == 2
            )
        ).astype(np.int8)

        out["is_two_strike"] = (
            out["strikes_before"] == 2
        ).astype(np.int8)

        out["is_three_ball"] = (
            out["balls_before"] == 3
        ).astype(np.int8)

        out["count_leverage"] = (
            out["balls_before"]
            - out["strikes_before"]
        ).astype(np.int8)

        # -----------------------------
        # B. Runner / game situation
        # -----------------------------

        out["is_scoring_position"] = (
            (
                out["runner_on_2b"] == 1
            )
            | (
                out["runner_on_3b"] == 1
            )
        ).astype(np.int8)

        out["is_bases_loaded"] = (
            out["num_runners_on"] == 3
        ).astype(np.int8)

        out["abs_score_diff"] = (
            out[
                "score_diff_pitcher_team"
            ]
            .abs()
        )

        out["is_close_game"] = (
            out["abs_score_diff"] <= 2
        ).astype(np.int8)

        out["is_late_game"] = (
            out["inning"] >= 7
        ).astype(np.int8)

        out["li_log"] = np.log1p(
            out["li"].clip(lower=0)
        )

        out["is_high_leverage"] = (
            out["li"] >= 2.0
        ).astype(np.int8)

        out["is_clutch"] = (
            (
                out["inning"] >= 7
            )
            & (
                out["abs_score_diff"] <= 2
            )
            & (
                out["li"] > 1.2
            )
        ).astype(np.int8)

        out[
            "win_expectancy_diff"
        ] = (
            out[
                "home_win_expectancy"
            ]
            - out[
                "away_win_expectancy"
            ]
        )

        # top of inning:
        # batter = away team
        # pitcher = home team
        #
        # bottom:
        # batter = home team
        #
        # Therefore home/stadium team:
        out[
            "stadium_owner_team"
        ] = np.where(
            out["top_bottom"].eq("T"),
            out["pitcher_team_id"],
            out["batter_team_id"],
        )

        out["same_hand"] = (
            out["pitcher_hand"]
            == out["batter_hand"]
        ).astype(np.int8)

        # -----------------------------
        # C. Safe empirical-Bayes
        # -----------------------------

        pitcher_n = (
            out["asof_pitcher_n"]
            .fillna(0)
            .astype(np.float32)
        )

        for col in PITCHER_RATE_COLS:

            raw = (
                out[col]
                .fillna(global_mean)
                .astype(np.float32)
            )

            strength = (
                self.config
                .pitcher_prior_strength
            )

            out[
                f"{col}_smoothed"
            ] = (
                pitcher_n * raw
                + strength * global_mean
            ) / (
                pitcher_n + strength
            )

        batter_n = (
            out["asof_batter_n"]
            .fillna(0)
            .astype(np.float32)
        )

        for col in BATTER_RATE_COLS:

            raw = (
                out[col]
                .fillna(global_mean)
                .astype(np.float32)
            )

            strength = (
                self.config
                .batter_prior_strength
            )

            out[
                f"{col}_smoothed"
            ] = (
                batter_n * raw
                + strength * global_mean
            ) / (
                batter_n + strength
            )

        # Cold-start confidence
        out[
            "pitcher_history_log_n"
        ] = np.log1p(
            out[
                "asof_pitcher_n"
            ].fillna(0)
        )

        out[
            "batter_history_log_n"
        ] = np.log1p(
            out[
                "asof_batter_n"
            ].fillna(0)
        )

        out[
            "pitchmix_history_log_n"
        ] = np.log1p(
            out[
                "asof_pitcher_pitchmix_n"
            ].fillna(0)
        )

        out[
            "pitcher_cold_start"
        ] = (
            out[
                "asof_pitcher_n"
            ].fillna(0)
            < 50
        ).astype(np.int8)

        out[
            "batter_cold_start"
        ] = (
            out[
                "asof_batter_n"
            ].fillna(0)
            < 50
        ).astype(np.int8)

        # -----------------------------
        # D. Recent form vs baseline
        # -----------------------------

        baseline_success = out[
            "asof_pitcher_success_rate_smoothed"
        ]

        baseline_middle = out[
            "asof_pitcher_middle_rate_smoothed"
        ]

        for n in (1, 3, 5):

            out[
                f"diff_prev{n}_success"
            ] = (
                out[
                    f"asof_pitcher_prev{n}_game_success_rate"
                ]
                - baseline_success
            )

            out[
                f"diff_prev{n}_middle"
            ] = (
                out[
                    f"asof_pitcher_prev{n}_game_middle_rate"
                ]
                - baseline_middle
            )

        out["pitcher_volatility"] = (
            out[
                "diff_prev1_success"
            ].abs()
            + out[
                "diff_prev3_success"
            ].abs()
        )

        out[
            "pressure_interaction"
        ] = (
            out["pitcher_volatility"]
            * out["li_log"]
        )

        out[
            "recent_form_leverage"
        ] = (
            out[
                "diff_prev3_success"
            ]
            * out["li_log"]
        )

        # -----------------------------
        # E. Control profile
        # -----------------------------

        eps = 1e-5

        out[
            "strike_to_ball_ratio"
        ] = (
            out[
                "asof_pitcher_strike_rate_smoothed"
            ]
            / (
                out[
                    "asof_pitcher_ball_rate_smoothed"
                ]
                + eps
            )
        )

        out[
            "reverse_to_success_ratio"
        ] = (
            out[
                "asof_pitcher_reverse_rate_smoothed"
            ]
            / (
                out[
                    "asof_pitcher_success_rate_smoothed"
                ]
                + eps
            )
        )

        out[
            "middle_to_success_ratio"
        ] = (
            out[
                "asof_pitcher_middle_rate_smoothed"
            ]
            / (
                out[
                    "asof_pitcher_success_rate_smoothed"
                ]
                + eps
            )
        )

        # -----------------------------
        # F. Pitch mix
        # -----------------------------

        pitchmix_cols = [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]

        p = (
            out[pitchmix_cols]
            .fillna(0.0)
            .clip(lower=eps, upper=1.0)
        )

        out["pitchmix_entropy"] = -(
            p * np.log(p)
        ).sum(axis=1)

        out[
            "pitchmix_max_share"
        ] = p.max(axis=1)

        # -----------------------------
        # G. Fold-local pitcher cluster
        # -----------------------------

        out[
            "pitcher_style_cluster"
        ] = (
            out["pitcher_id"]
            .map(
                self.pitcher_cluster_map_
            )
            .fillna(-1)
            .astype(np.int16)
        )

        out[
            "pitcher_cluster_unseen"
        ] = (
            out[
                "pitcher_style_cluster"
            ] < 0
        ).astype(np.int8)

        # -----------------------------
        # H. Strict-past Trackman
        # -----------------------------

        if (
            self.trackman_features
            is not None
        ):
            tm_features = (
                self.trackman_features
                .transform(out)
            )

            for col in tm_features.columns:
                out[col] = (
                    tm_features[col]
                    .to_numpy()
                )

        return out

    def _fit_zscore_stats(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validation distribution is NEVER used.

        When training 2019~2023 and validating 2024,
        z-score reference statistics are fitted from
        the latest TRAIN season (2023), not 2024.
        """

        latest = df[
            df["season"]
            == self.latest_train_season_
        ]

        rate_cols = [
            f"{c}_smoothed"
            for c in (
                *PITCHER_RATE_COLS,
                *BATTER_RATE_COLS,
            )
        ]

        self.zscore_stats_.clear()

        for col in rate_cols:

            if col not in latest.columns:
                continue

            for game_type in (
                latest[
                    "game_type"
                ]
                .dropna()
                .unique()
            ):

                values = latest.loc[
                    latest[
                        "game_type"
                    ] == game_type,
                    col,
                ].dropna()

                if len(values) == 0:
                    continue

                mean = float(values.mean())
                std = _safe_std(
                    float(values.std())
                )

                self.zscore_stats_[
                    (
                        col,
                        str(game_type),
                    )
                ] = (
                    mean,
                    std,
                )

            # fallback independent
            # of game_type
            values = latest[
                col
            ].dropna()

            if len(values) > 0:
                self.zscore_stats_[
                    (
                        col,
                        "__GLOBAL__",
                    )
                ] = (
                    float(values.mean()),
                    _safe_std(
                        float(values.std())
                    ),
                )

    def _apply_zscores(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = df.copy()

        rate_cols = [
            f"{c}_smoothed"
            for c in (
                *PITCHER_RATE_COLS,
                *BATTER_RATE_COLS,
            )
        ]

        game_types = (
            out["game_type"]
            .astype(str)
            .to_numpy()
        )

        for col in rate_cols:

            if col not in out.columns:
                continue

            values = (
                out[col]
                .astype(np.float32)
                .to_numpy()
            )

            z = np.zeros(
                len(out),
                dtype=np.float32,
            )

            for game_type in np.unique(
                game_types
            ):

                mask = (
                    game_types
                    == game_type
                )

                stats = (
                    self.zscore_stats_
                    .get(
                        (
                            col,
                            game_type,
                        )
                    )
                )

                if stats is None:
                    stats = (
                        self.zscore_stats_
                        .get(
                            (
                                col,
                                "__GLOBAL__",
                            ),
                            (0.0, 1.0),
                        )
                    )

                mean, std = stats

                z[mask] = (
                    values[mask]
                    - mean
                ) / (
                    std + 1e-6
                )

            out[
                f"{col}_zscore"
            ] = z

        return out
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import (
    ExperimentConfig,
)

from src.data import (
    load_train,
    validate_train_schema,
)

from src.features import (
    LeakageSafeFeatureEngineer,
    StrictPastTrackmanFeatures,
)

from src.metrics import (
    evaluate_constant_baselines,
    evaluate_probabilities,
)

from src.models import (
    train_catboost,
    train_lightgbm,
    train_xgboost,
)

from src.preprocessing import (
    TabularPreprocessor,
)

from src.splits import (
    make_recency_weights,
    make_temporal_folds,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recency-lambda",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "xgb",
            "lgb",
            "cat",
        ],
        choices=[
            "xgb",
            "lgb",
            "cat",
        ],
    )

    parser.add_argument(
        "--valid-season",
        type=int,
        default=None,
        choices=[
            2023,
            2024,
        ],
        help=(
            "Run only one temporal fold. "
            "Recommended for memory safety."
        ),
    )

    parser.add_argument(
        "--no-trackman",
        action="store_true",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="v7_tabred",
    )

    return parser.parse_args()


def print_memory(
    label: str,
):
    try:
        import resource

        rss_kb = (
            resource
            .getrusage(
                resource.RUSAGE_SELF
            )
            .ru_maxrss
        )

        rss_gb = (
            rss_kb
            / 1024.0
            / 1024.0
        )

        print(
            f"[MEM] {label}: "
            f"peak RSS≈{rss_gb:.2f} GB"
        )

    except Exception:
        pass


def print_season_target_rates(
    df: pd.DataFrame,
    target_col: str,
):

    stats = (
        df
        .groupby(
            "season"
        )[target_col]
        .agg(
            [
                "count",
                "mean",
            ]
        )
    )

    print(
        "\n[TARGET DRIFT]"
    )

    print(
        stats.to_string()
    )

    print()


def print_baselines(
    baseline_metrics,
):

    print(
        "\n[BASELINES]"
    )

    print(
        "  train mean probability = "
        f"{baseline_metrics['train_mean']:.6f}"
    )

    print(
        "  train-mean Brier = "
        f"{baseline_metrics['train_mean_brier']:.8f}"
    )

    if (
        "latest_season_mean"
        in baseline_metrics
    ):
        print(
            "  latest train season = "
            f"{baseline_metrics['latest_train_season']}"
        )

        print(
            "  latest-season mean probability = "
            f"{baseline_metrics['latest_season_mean']:.6f}"
        )

        print(
            "  latest-season mean Brier = "
            f"{baseline_metrics['latest_season_mean_brier']:.8f}"
        )

    print()


def print_metrics(
    fold_name,
    model_name,
    metrics,
):

    print(
        f"\n[{fold_name}] "
        f"{model_name}"
    )

    print(
        f"  Brier = "
        f"{metrics['brier']:.8f}"
    )

    print(
        f"  LogLoss = "
        f"{metrics['logloss']:.8f}"
    )

    print(
        f"  AUC = "
        f"{metrics['auc']:.6f}"
    )

    print(
        f"  target_mean = "
        f"{metrics['target_mean']:.6f}"
    )

    print(
        f"  pred_mean = "
        f"{metrics['pred_mean']:.6f}"
    )

    print(
        f"  pred_std = "
        f"{metrics['pred_std']:.6f}"
    )

    print(
        f"  calibration_bias = "
        f"{metrics['calibration_bias']:.6f}"
    )


def select_fold_specs(
    config,
    valid_season,
):

    if valid_season is None:
        return (
            config.temporal_folds
        )

    selected = tuple(
        spec
        for spec
        in config.temporal_folds
        if spec[1]
        == valid_season
    )

    if not selected:
        raise ValueError(
            f"No fold found for "
            f"valid season "
            f"{valid_season}"
        )

    return selected


def main():

    args = parse_args()

    config = ExperimentConfig(
        recency_lambda=(
            args.recency_lambda
        ),
        use_trackman=(
            not args.no_trackman
        ),
    )

    fold_suffix = (
        f"valid_{args.valid_season}"
        if args.valid_season
        is not None
        else "all_folds"
    )

    output_dir = (
        config
        .paths
        .output_dir
        / args.experiment_name
        / (
            f"lambda_"
            f"{args.recency_lambda}"
        )
        / fold_suffix
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = load_train(
        config.paths.train_path
    )

    validate_train_schema(
        train_df,
        target_col=(
            config
            .features
            .target_col
        ),
        id_col=(
            config
            .features
            .id_col
        ),
    )

    target_col = (
        config
        .features
        .target_col
    )

    print_season_target_rates(
        train_df,
        target_col,
    )

    print_memory(
        "after train load"
    )

    trackman = None

    if config.use_trackman:

        if not (
            config
            .paths
            .trackman_path
            .exists()
        ):
            raise FileNotFoundError(
                config
                .paths
                .trackman_path
            )

        trackman = (
            StrictPastTrackmanFeatures()
            .fit_from_csv(
                config
                .paths
                .trackman_path
            )
        )

        gc.collect()

        print_memory(
            "after Trackman build"
        )

    fold_specs = (
        select_fold_specs(
            config,
            args.valid_season,
        )
    )

    experiment_results = []

    for fold in make_temporal_folds(
        train_df,
        fold_specs,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"[FOLD] {fold.name}"
        )

        print(
            f"[FOLD] train seasons="
            f"{fold.train_seasons}"
        )

        print(
            f"[FOLD] valid season="
            f"{fold.valid_season}"
        )

        print(
            f"[FOLD] n_train="
            f"{len(fold.train_idx):,}, "
            f"n_valid="
            f"{len(fold.valid_idx):,}"
        )

        print(
            "=" * 80
        )

        # No explicit .copy().
        # FeatureEngineer owns its transformed copy.
        raw_train = (
            train_df
            .iloc[
                fold.train_idx
            ]
        )

        raw_valid = (
            train_df
            .iloc[
                fold.valid_idx
            ]
        )

        y_train = (
            raw_train[
                target_col
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        y_valid = (
            raw_valid[
                target_col
            ]
            .to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

        train_seasons_array = (
            raw_train[
                "season"
            ]
            .to_numpy(
                dtype=np.int16,
                copy=True,
            )
        )

        baseline_metrics = (
            evaluate_constant_baselines(
                y_train=y_train,
                y_valid=y_valid,
                train_seasons=(
                    train_seasons_array
                ),
            )
        )

        print_baselines(
            baseline_metrics
        )

        print_memory(
            "before feature engineering"
        )

        # -------------------------------------------------
        # 1. Fold-local feature engineering
        # -------------------------------------------------

        feature_engineer = (
            LeakageSafeFeatureEngineer(
                config=(
                    config.features
                ),
                trackman_features=(
                    trackman
                ),
            )
        )

        feature_engineer.fit(
            raw_train,
            raw_train[
                target_col
            ],
        )

        train_features = (
            feature_engineer
            .transform(
                raw_train
            )
        )

        valid_features = (
            feature_engineer
            .transform(
                raw_valid
            )
        )

        print_memory(
            "after feature engineering"
        )

        # -------------------------------------------------
        # 2. Fold-local preprocessing
        # -------------------------------------------------

        preprocessor = (
            TabularPreprocessor(
                categorical_cols=(
                    config
                    .features
                    .categorical_cols
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
                train_features
            )
        )

        X_valid = (
            preprocessor
            .transform(
                valid_features
            )
        )

        del train_features
        del valid_features

        gc.collect()

        print_memory(
            "after preprocessing"
        )

        if list(
            X_train.columns
        ) != list(
            X_valid.columns
        ):
            raise RuntimeError(
                "Train/valid feature "
                "columns do not match."
            )

        # -------------------------------------------------
        # 3. Recency weight
        # -------------------------------------------------

        sample_weight = (
            make_recency_weights(
                train_seasons_array,
                max_train_season=max(
                    fold.train_seasons
                ),
                decay_lambda=(
                    config
                    .recency_lambda
                ),
            )
        )

        print(
            "[WEIGHT] "
            f"lambda="
            f"{config.recency_lambda}, "
            f"min="
            f"{sample_weight.min():.4f}, "
            f"max="
            f"{sample_weight.max():.4f}, "
            f"mean="
            f"{sample_weight.mean():.4f}"
        )

        fold_pred_frame = (
            pd.DataFrame(
                {
                    "row_id": (
                        raw_valid[
                            config
                            .features
                            .id_col
                        ].to_numpy()
                    ),
                    "season": (
                        raw_valid[
                            "season"
                        ].to_numpy()
                    ),
                    "y_true": (
                        y_valid
                    ),
                }
            )
        )

        # -------------------------------------------------
        # 4. XGBoost
        # -------------------------------------------------

        if "xgb" in args.models:

            print_memory(
                "before XGBoost"
            )

            start = (
                time.perf_counter()
            )

            model, pred = (
                train_xgboost(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    sample_weight=(
                        sample_weight
                    ),
                    config=(
                        config.models
                    ),
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            metrics = (
                evaluate_probabilities(
                    y_valid,
                    pred,
                )
            )

            metrics[
                "train_seconds"
            ] = float(
                elapsed
            )

            print_metrics(
                fold.name,
                "xgb",
                metrics,
            )

            experiment_results.append(
                {
                    "fold": (
                        fold.name
                    ),
                    "valid_season": (
                        fold.valid_season
                    ),
                    "model": "xgb",
                    **baseline_metrics,
                    **metrics,
                }
            )

            fold_pred_frame[
                "pred_xgb"
            ] = pred

            model.save_model(
                str(
                    output_dir
                    / (
                        f"{fold.name}"
                        "_xgb.json"
                    )
                )
            )

            del model
            del pred

            gc.collect()

            print_memory(
                "after XGBoost"
            )

        # -------------------------------------------------
        # 5. LightGBM
        # -------------------------------------------------

        if "lgb" in args.models:

            print_memory(
                "before LightGBM"
            )

            start = (
                time.perf_counter()
            )

            model, pred = (
                train_lightgbm(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    sample_weight=(
                        sample_weight
                    ),
                    categorical_cols=(
                        preprocessor
                        .cat_cols_
                    ),
                    config=(
                        config.models
                    ),
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            metrics = (
                evaluate_probabilities(
                    y_valid,
                    pred,
                )
            )

            metrics[
                "train_seconds"
            ] = float(
                elapsed
            )

            print_metrics(
                fold.name,
                "lgb",
                metrics,
            )

            experiment_results.append(
                {
                    "fold": (
                        fold.name
                    ),
                    "valid_season": (
                        fold.valid_season
                    ),
                    "model": "lgb",
                    **baseline_metrics,
                    **metrics,
                }
            )

            fold_pred_frame[
                "pred_lgb"
            ] = pred

            model.save_model(
                str(
                    output_dir
                    / (
                        f"{fold.name}"
                        "_lgb.txt"
                    )
                )
            )

            del model
            del pred

            gc.collect()

            print_memory(
                "after LightGBM"
            )

        # -------------------------------------------------
        # 6. CatBoost
        # -------------------------------------------------

        if "cat" in args.models:

            print_memory(
                "before CatBoost"
            )

            start = (
                time.perf_counter()
            )

            model, pred = (
                train_catboost(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    sample_weight=(
                        sample_weight
                    ),
                    categorical_indices=(
                        preprocessor
                        .categorical_indices
                    ),
                    config=(
                        config.models
                    ),
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            metrics = (
                evaluate_probabilities(
                    y_valid,
                    pred,
                )
            )

            metrics[
                "train_seconds"
            ] = float(
                elapsed
            )

            print_metrics(
                fold.name,
                "cat",
                metrics,
            )

            experiment_results.append(
                {
                    "fold": (
                        fold.name
                    ),
                    "valid_season": (
                        fold.valid_season
                    ),
                    "model": "cat",
                    **baseline_metrics,
                    **metrics,
                }
            )

            fold_pred_frame[
                "pred_cat"
            ] = pred

            model.save_model(
                str(
                    output_dir
                    / (
                        f"{fold.name}"
                        "_cat.cbm"
                    )
                )
            )

            del model
            del pred

            gc.collect()

            print_memory(
                "after CatBoost"
            )

        # -------------------------------------------------
        # 7. Save fold state
        # -------------------------------------------------

        joblib.dump(
            {
                "feature_engineer": (
                    feature_engineer
                ),
                "preprocessor": (
                    preprocessor
                ),
                "train_seasons": (
                    fold.train_seasons
                ),
                "valid_season": (
                    fold.valid_season
                ),
            },
            output_dir
            / (
                f"{fold.name}"
                "_pipeline.pkl"
            ),
        )

        fold_pred_frame.to_csv(
            output_dir
            / (
                f"{fold.name}"
                "_predictions.csv"
            ),
            index=False,
        )

        del raw_train
        del raw_valid
        del X_train
        del X_valid
        del y_train
        del y_valid
        del train_seasons_array
        del sample_weight
        del feature_engineer
        del preprocessor
        del fold_pred_frame

        gc.collect()

        print_memory(
            "fold completed"
        )

    # -------------------------------------------------
    # 8. Result summary
    # -------------------------------------------------

    result_df = (
        pd.DataFrame(
            experiment_results
        )
    )

    result_df.to_csv(
        output_dir
        / "metrics.csv",
        index=False,
    )

    summary = {
        "experiment_name": (
            args.experiment_name
        ),
        "recency_lambda": (
            args.recency_lambda
        ),
        "use_trackman": (
            config.use_trackman
        ),
        "models": args.models,
        "valid_season": (
            args.valid_season
        ),
    }

    with open(
        output_dir
        / "experiment.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "[DONE] Temporal validation"
    )

    if not result_df.empty:

        cols = [
            "fold",
            "model",
            "brier",
            "train_mean_brier",
            "latest_season_mean_brier",
            "logloss",
            "auc",
            "pred_mean",
            "target_mean",
            "train_seconds",
        ]

        cols = [
            c
            for c in cols
            if c
            in result_df.columns
        ]

        print(
            result_df[
                cols
            ].to_string(
                index=False
            )
        )

    print(
        f"\nSaved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
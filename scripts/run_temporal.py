# scripts/run_temporal.py

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


from src.config import ExperimentConfig
from src.data import (
    load_train,
    validate_train_schema,
)
from src.features import (
    LeakageSafeFeatureEngineer,
    StrictPastTrackmanFeatures,
)
from src.metrics import (
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
        "--no-trackman",
        action="store_true",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="v7_tabred",
    )

    return parser.parse_args()


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
        f"  BSS(empirical) = "
        f"{metrics['brier_skill_empirical']:.6f}"
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

    output_dir = (
        config.paths.output_dir
        / args.experiment_name
        / (
            f"lambda_"
            f"{args.recency_lambda}"
        )
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

    # Trackman is built from official history only.
    # It NEVER reads test rows.
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

    experiment_results = []

    prediction_frames = []

    for fold in make_temporal_folds(
        train_df,
        config.temporal_folds,
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

        raw_train = (
            train_df
            .iloc[
                fold.train_idx
            ]
            .copy()
        )

        raw_valid = (
            train_df
            .iloc[
                fold.valid_idx
            ]
            .copy()
        )

        target_col = (
            config
            .features
            .target_col
        )

        y_train = (
            raw_train[
                target_col
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        y_valid = (
            raw_valid[
                target_col
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        # -----------------------------------------
        # 1. FIT feature engineering on train only.
        # -----------------------------------------

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

        # -----------------------------------------
        # 2. FIT preprocessing on train only.
        # -----------------------------------------

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

        if list(
            X_train.columns
        ) != list(
            X_valid.columns
        ):
            raise RuntimeError(
                "Train/valid feature "
                "columns do not match."
            )

        # -----------------------------------------
        # 3. Train-side recency weights only.
        # -----------------------------------------

        sample_weight = (
            make_recency_weights(
                raw_train[
                    "season"
                ].to_numpy(),
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

        fold_pred_frame = pd.DataFrame(
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
                "y_true": y_valid,
            }
        )

        # -----------------------------------------
        # 4. Models
        # -----------------------------------------

        if "xgb" in args.models:

            start = time.perf_counter()

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
            ] = float(elapsed)

            print_metrics(
                fold.name,
                "xgb",
                metrics,
            )

            experiment_results.append(
                {
                    "fold": fold.name,
                    "valid_season": (
                        fold.valid_season
                    ),
                    "model": "xgb",
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

        if "lgb" in args.models:

            start = time.perf_counter()

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
            ] = float(elapsed)

            print_metrics(
                fold.name,
                "lgb",
                metrics,
            )

            experiment_results.append(
                {
                    "fold": fold.name,
                    "valid_season": (
                        fold.valid_season
                    ),
                    "model": "lgb",
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

        if "cat" in args.models:

            start = time.perf_counter()

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
            ] = float(elapsed)

            print_metrics(
                fold.name,
                "cat",
                metrics,
            )

            experiment_results.append(
                {
                    "fold": fold.name,
                    "valid_season": (
                        fold.valid_season
                    ),
                    "model": "cat",
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

        # -----------------------------------------
        # 5. Save fold-local transformation states.
        # -----------------------------------------

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

        prediction_frames.append(
            fold_pred_frame
        )

        del (
            raw_train,
            raw_valid,
            train_features,
            valid_features,
            X_train,
            X_valid,
            y_train,
            y_valid,
        )

        gc.collect()

    # ---------------------------------------------
    # 6. Save all results
    # ---------------------------------------------

    result_df = pd.DataFrame(
        experiment_results
    )

    result_df.to_csv(
        output_dir
        / "metrics.csv",
        index=False,
    )

    all_predictions = pd.concat(
        prediction_frames,
        axis=0,
        ignore_index=True,
    )

    all_predictions.to_csv(
        output_dir
        / "temporal_predictions.csv",
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

    print(
        result_df[
            [
                "fold",
                "model",
                "brier",
                "logloss",
                "auc",
                "pred_mean",
                "target_mean",
            ]
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
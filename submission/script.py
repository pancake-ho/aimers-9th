from __future__ import annotations

import gc
import importlib.util
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier


ROOT = Path(".")
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"

ID_COL = "row_id"
TARGET_COL = "control_success"


def _load_module(
    filename: str,
    module_name: str,
):
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load runtime module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def _load_csv(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


def _validate_inputs(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    bundle,
) -> None:
    if bundle.get("bundle_version") != 7:
        raise ValueError(
            "Unsupported bundle version: "
            f"{bundle.get('bundle_version')}"
        )

    if (
        bundle.get("id_col") != ID_COL
        or bundle.get("target_col")
        != TARGET_COL
    ):
        raise ValueError(
            "Bundle column contract does not "
            "match competition contract."
        )

    if (
        ID_COL not in test
        or TARGET_COL in test
    ):
        raise ValueError(
            "test.csv schema is invalid."
        )

    if (
        test[ID_COL].isna().any()
        or test[ID_COL].duplicated().any()
    ):
        raise ValueError(
            "test row_id must be "
            "non-null and unique."
        )

    if list(sample.columns) != [
        ID_COL,
        TARGET_COL,
    ]:
        raise ValueError(
            "Unexpected sample_submission "
            f"columns: {list(sample.columns)}"
        )

    if (
        sample[ID_COL].isna().any()
        or sample[ID_COL].duplicated().any()
    ):
        raise ValueError(
            "sample_submission row_id "
            "must be non-null and unique."
        )

    if (
        len(test) != len(sample)
        or set(test[ID_COL])
        != set(sample[ID_COL])
    ):
        raise ValueError(
            "test/sample row_id sets "
            "do not match."
        )

    expected = set(
        bundle["expected_raw_columns"]
    )

    missing = (
        expected - set(test.columns)
    )

    if missing:
        raise ValueError(
            "test.csv is missing trained "
            f"columns: {sorted(missing)}"
        )


def _load_gbdt_models(
    bundle,
    active_outer_models,
):
    bagging_state = (
        bundle.get(
            "xgb_bagging",
            {},
        )
    )

    xgb_files = list(
        bagging_state.get(
            "model_files",
            [
                "xgb_model.json",
            ],
        )
    )

    xgb_seeds = list(
        bagging_state.get(
            "seeds",
            [],
        )
    )

    if not xgb_files:
        raise ValueError(
            "XGBoost model file list "
            "must not be empty."
        )

    if (
        xgb_files[0]
        != "xgb_model.json"
    ):
        raise ValueError(
            "First XGBoost model must "
            "be xgb_model.json."
        )

    if (
        xgb_seeds
        and len(xgb_seeds)
        != len(xgb_files)
    ):
        raise ValueError(
            "XGBoost seed/model count "
            "mismatch."
        )

    xgb_models = []

    for filename in xgb_files:
        path = (
            MODEL_DIR
            / filename
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        model = xgb.Booster()

        model.load_model(
            str(path)
        )

        model.set_param(
            {
                "nthread": 6,
                "device": "cpu",
            }
        )

        xgb_models.append(
            model
        )

    lgb_path = (
        MODEL_DIR
        / "lgb_model.txt"
    )

    cat_path = (
        MODEL_DIR
        / "cat_model.cbm"
    )

    for path in (
        lgb_path,
        cat_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    lgb_model = lgb.Booster(
        model_file=str(
            lgb_path
        )
    )

    cat_model = (
        CatBoostClassifier()
    )

    cat_model.load_model(
        str(cat_path)
    )

    print(
        "[XGB-BAG] loaded_models="
        f"{len(xgb_models)} "
        f"seeds={xgb_seeds}"
    )

    return (
        xgb_models,
        lgb_model,
        cat_model,
    )

def _predict_xgb_bag(
    models,
    dmatrix,
) -> np.ndarray:
    if not models:
        raise ValueError(
            "XGBoost bag must contain "
            "at least one model."
        )

    predictions = []

    expected_shape = None

    for index, model in enumerate(
        models
    ):
        prediction = np.asarray(
            model.predict(
                dmatrix
            ),
            dtype=np.float64,
        )

        if prediction.ndim != 1:
            raise ValueError(
                "XGBoost bag prediction "
                "must be one-dimensional: "
                f"index={index} "
                f"shape={prediction.shape}"
            )

        if expected_shape is None:
            expected_shape = (
                prediction.shape
            )
        elif (
            prediction.shape
            != expected_shape
        ):
            raise ValueError(
                "XGBoost bag prediction "
                "shape mismatch: "
                f"index={index} "
                f"expected={expected_shape} "
                f"actual={prediction.shape}"
            )

        if not np.isfinite(
            prediction
        ).all():
            raise ValueError(
                "XGBoost bag prediction "
                "contains NaN or infinity: "
                f"index={index}"
            )

        predictions.append(
            prediction
        )

    stacked = np.vstack(
        predictions
    )

    output = stacked.mean(
        axis=0,
        dtype=np.float64,
    )

    if not np.isfinite(
        output
    ).all():
        raise ValueError(
            "Averaged XGBoost prediction "
            "contains NaN or infinity."
        )

    return output


def main() -> None:
    started = time.perf_counter()

    print(
        "[1/8] Load immutable model bundle"
    )

    bundle = joblib.load(
        MODEL_DIR / "bundle.pkl"
    )

    runtime = _load_module(
        "runtime.py",
        "aimers_runtime",
    )

    model_order = list(
        bundle["ensemble"]["model_order"]
    )

    ensemble = bundle[
        "ensemble"
    ]

    outer_weight = {
        name: float(weight)
        for name, weight
        in zip(
            model_order,
            ensemble[
                "weights"
            ],
        )
    }

    WEIGHT_EPS = 1.0e-12

    active_outer_models = {
        name
        for name, weight
        in outer_weight.items()
        if abs(weight)
        > WEIGHT_EPS
    }

    print(
        "[MODEL] active_outer_models="
        f"{sorted(active_outer_models)}"
    )

    supported_orders = [
        ["xgb", "lgb", "cat"],
        [
            "xgb",
            "lgb",
            "cat",
            "resnet",
        ],
        [
            "xgb",
            "lgb",
            "cat",
            "resnet",
            "ft_transformer",
        ],
    ]

    if model_order not in supported_orders:
        raise ValueError(
            "Unsupported model order: "
            f"{model_order}"
        )

    neural = None
    neural_device = None

    if (
        "resnet" in model_order
        or "ft_transformer"
        in model_order
    ):
        neural = _load_module(
            "neural_runtime.py",
            "aimers_neural_runtime",
        )

        neural.torch.set_num_threads(6)

        neural_device = (
            neural.resolve_device("auto")
        )

    print(
        "[2/8] Load and validate inputs"
    )

    test = _load_csv(
        DATA_DIR / "test.csv"
    )

    sample = _load_csv(
        DATA_DIR / "sample_submission.csv"
    )

    _validate_inputs(
        test,
        sample,
        bundle,
    )

    row_ids = test[ID_COL].copy()

    print(
        "[3/8] Build row-independent "
        f"features: rows={len(test):,}"
    )

    features = runtime.build_features(
        test,
        bundle["feature_state"],
    )

    del test
    gc.collect()

    print(
        "[4/8] Apply train-fitted "
        "preprocessing"
    )

    X = runtime.preprocess_frame(
        features,
        bundle[
            "preprocessor_state"
        ],
    )

    native_state = bundle.get(
        "xgb_native_categorical"
    )

    X_native = None

    if (
        native_state
        and native_state.get(
            "enabled",
            False,
        )
        and float(
            native_state.get(
                "weight",
                0.0,
            )
        )
        > 0.0
    ):
        X_native = (
            runtime
            .preprocess_xgb_native_frame(
                features,
                bundle[
                    "preprocessor_state"
                ],
            )
        )

        print(
            "[XGB-NATIVE-CAT] "
            f"prepared_features="
            f"{X_native.shape[1]} "
            f"weight="
            f"{float(native_state['weight']):.3f}"
        )

    del features
    gc.collect()

    print(
        "[5/8] Predict "
        "XGBoost(full entity) + "
        "LightGBM/CatBoost(aux-base) "
        f"xgb_features={X.shape[1]} "
    )

    (
        xgb_models,
        lgb_model,
        cat_model,
    ) = _load_gbdt_models(
        bundle
    )

    dtest = xgb.DMatrix(
        X,
        feature_names=list(X.columns),
    )

    xgb_prediction = (
        _predict_xgb_bag(
            xgb_models,
            dtest,
        )
    )
    multiview = (
        bundle.get(
            "xgb_multiview"
        )
    )

    if multiview:
        representation_model = (
            xgb.Booster()
        )

        representation_model.load_model(
            str(
                MODEL_DIR
                / multiview[
                    "representation_model_file"
                ]
            )
        )

        representation_model.set_param(
            {
                "nthread": 6,
                "device": "cpu",
            }
        )

        representative_model = (
            xgb.Booster()
        )

        representative_model.load_model(
            str(
                MODEL_DIR
                / multiview[
                    "representative_model_file"
                ]
            )
        )

        representative_model.set_param(
            {
                "nthread": 6,
                "device": "cpu",
            }
        )

        pca = (
            runtime
            .apply_numeric_pca_state(
                X,
                multiview[
                    "pca_state"
                ],
            )
        )

        representation_X = (
            pd.concat(
                [
                    X[
                        multiview[
                            "representation_raw_features"
                        ]
                    ],
                    pca,
                ],
                axis=1,
            )
        )

        drepresentation = (
            xgb.DMatrix(
                representation_X,
                feature_names=list(
                    representation_X.columns
                ),
            )
        )

        representation_prediction = (
            np.asarray(
                representation_model.predict(
                    drepresentation
                ),
                dtype=np.float64,
            )
        )

        representative_prediction = (
            np.asarray(
                representative_model.predict(
                    dtest
                ),
                dtype=np.float64,
            )
        )

        xgb_prediction = (
            runtime
            .blend_predictions(
                [
                    xgb_prediction,
                    representation_prediction,
                    representative_prediction,
                ],
                multiview[
                    "weights"
                ],
            )
        )

        print(
            "[XGB-MULTIVIEW] "
            f"weights="
            f"{multiview['weights']} "
            f"repr_features="
            f"{representation_X.shape[1]}"
        )

        del (
            representation_model,
            representative_model,
            representation_prediction,
            representative_prediction,
            representation_X,
            drepresentation,
            pca,
        )
        
    temporal_state = (
        bundle.get(
            "xgb_temporal_views"
        )
    )

    if temporal_state:
        temporal_files = dict(
            temporal_state[
                "model_files"
            ]
        )

        temporal_predictions = [
            xgb_prediction
        ]

        for name in (
            "recent1",
            "recent2",
        ):
            filename = str(
                temporal_files[
                    name
                ]
            )

            path = (
                MODEL_DIR
                / filename
            )

            if not path.exists():
                raise FileNotFoundError(
                    path
                )

            temporal_model = (
                xgb.Booster()
            )

            temporal_model.load_model(
                str(
                    path
                )
            )

            temporal_model.set_param(
                {
                    "nthread": 6,
                    "device": "cpu",
                }
            )

            temporal_prediction = (
                np.asarray(
                    temporal_model.predict(
                        dtest
                    ),
                    dtype=np.float64,
                )
            )

            temporal_predictions.append(
                temporal_prediction
            )

            del temporal_model

        xgb_prediction = (
            runtime.blend_predictions(
                temporal_predictions,
                temporal_state[
                    "weights"
                ],
            )
        )

        print(
            "[XGB-TEMPORAL] "
            f"weights="
            f"{temporal_state['weights']}"
        )

        del temporal_predictions    

    if (
        native_state
        and X_native is not None
    ):
        native_weight = float(
            native_state[
                "weight"
            ]
        )

        native_filename = str(
            native_state[
                "model_file"
            ]
        )

        native_path = (
            MODEL_DIR
            / native_filename
        )

        if not native_path.exists():
            raise FileNotFoundError(
                native_path
            )

        native_model = xgb.Booster()

        native_model.load_model(
            str(
                native_path
            )
        )

        native_model.set_param(
            {
                "nthread": 6,
                "device": "cpu",
            }
        )

        d_native = xgb.DMatrix(
            X_native,
            feature_names=list(
                X_native.columns
            ),
            enable_categorical=True,
        )

        native_prediction = (
            np.asarray(
                native_model.predict(
                    d_native
                ),
                dtype=np.float64,
            )
        )

        if (
            native_prediction.shape
            != xgb_prediction.shape
        ):
            raise ValueError(
                "Native categorical XGB "
                "prediction shape mismatch."
            )

        if not np.isfinite(
            native_prediction
        ).all():
            raise ValueError(
                "Native categorical XGB "
                "prediction is non-finite."
            )

        xgb_prediction = (
            (
                1.0
                - native_weight
            )
            * xgb_prediction
            + native_weight
            * native_prediction
        )

        xgb_prediction = np.clip(
            xgb_prediction,
            1.0e-6,
            1.0 - 1.0e-6,
        )

        print(
            "[XGB-NATIVE-CAT] "
            f"weight={native_weight:.3f}"
        )

        del (
            native_model,
            native_prediction,
            d_native,
            X_native,
        )

    lgb_prediction = np.asarray(
        lgb_model.predict(
            X,
            num_threads=6,
        ),
        dtype=np.float64,
    )

    cat_prediction = np.asarray(
        cat_model.predict_proba(
            X,
            thread_count=6,
        )[:, 1],
        dtype=np.float64,
    )

    predictions = {
        "xgb": xgb_prediction,
        "lgb": lgb_prediction,
        "cat": cat_prediction,
    }

    del (
        dtest,
        xgb_models,
        lgb_model,
        cat_model,
    )

    gc.collect()

    if neural is not None:
        neural_names = model_order[3:]

        print(
            "[6/8] Predict neural models: "
            f"{neural_names} "
            f"device={neural_device}"
        )

        neural_arrays = (
            neural.prepare_neural_arrays(
                X,
                bundle[
                    "neural_preprocessor_state"
                ],
            )
        )

        if "resnet" in neural_names:
            resnet_model = (
                neural.load_checkpoint(
                    MODEL_DIR / "resnet.pt",
                    device=neural_device,
                )
            )

            predictions["resnet"] = (
                neural.predict_model(
                    resnet_model,
                    neural_arrays,
                    device=neural_device,
                    batch_size=8192,
                )
            )

            del resnet_model

        if (
            "ft_transformer"
            in neural_names
        ):
            ft_model = (
                neural.load_checkpoint(
                    MODEL_DIR
                    / "ft_transformer.pt",
                    device=neural_device,
                )
            )

            predictions[
                "ft_transformer"
            ] = neural.predict_model(
                ft_model,
                neural_arrays,
                device=neural_device,
                batch_size=2048,
            )

            del ft_model

        del neural_arrays

    else:
        print(
            "[6/8] Neural models absent"
        )

    del X
    gc.collect()

    print(
        "[7/8] Blend and apply "
        "train-only calibration"
    )

    ensemble = bundle["ensemble"]

    prediction = (
        runtime.blend_predictions(
            [
                predictions[name]
                for name in model_order
            ],
            ensemble["weights"],
        )
    )

    calibration = (
        ensemble["calibration"]
    )

    if calibration.get(
        "accepted",
        False,
    ):
        allowed_methods = {
            "logit_intercept",
            "abs_regime_logit_intercept",
        }

        if (
            calibration.get("method")
            not in allowed_methods
        ):
            raise ValueError(
                "Unsupported calibration: "
                f"{calibration.get('method')}"
            )

        prediction = (
            runtime.apply_logit_intercept(
                prediction,
                float(
                    calibration[
                        "intercept"
                    ]
                ),
            )
        )

    if prediction.shape != (
        len(row_ids),
    ):
        raise ValueError(
            "Prediction shape mismatch: "
            f"{prediction.shape}"
        )

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            "Prediction contains "
            "NaN or infinity."
        )

    prediction = np.clip(
        prediction,
        1e-6,
        1.0 - 1e-6,
    )

    print(
        "[8/8] Restore sample order "
        "and write submission.csv"
    )

    prediction_by_id = pd.DataFrame(
        {
            ID_COL: row_ids,
            TARGET_COL: prediction,
        }
    )

    submission = (
        sample[[ID_COL]]
        .merge(
            prediction_by_id,
            on=ID_COL,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    if submission[
        TARGET_COL
    ].isna().any():
        raise ValueError(
            "Submission merge produced "
            "missing predictions."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / "submission.csv"
    )

    submission.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        f"[DONE] {output_path} "
        f"rows={len(submission):,} "
        f"pred_mean="
        f"{submission[TARGET_COL].mean():.6f} "
        f"pred_std="
        f"{submission[TARGET_COL].std(ddof=0):.6f} "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
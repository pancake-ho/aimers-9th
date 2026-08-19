from __future__ import annotations

import gc
import importlib.util
import sys
import time
from pathlib import Path

import joblib
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


def _load_module(filename: str, module_name: str):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _validate_inputs(test: pd.DataFrame, sample: pd.DataFrame, bundle) -> None:
    if bundle.get("bundle_version") != 5:
        raise ValueError(f"Unsupported bundle version: {bundle.get('bundle_version')}")
    if bundle.get("id_col") != ID_COL or bundle.get("target_col") != TARGET_COL:
        raise ValueError("Bundle column contract does not match the competition contract.")
    if ID_COL not in test or TARGET_COL in test:
        raise ValueError("test.csv must contain row_id and must not contain the target.")
    if test[ID_COL].isna().any() or test[ID_COL].duplicated().any():
        raise ValueError("test.csv row_id values must be non-null and unique.")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Unexpected sample_submission columns: {list(sample.columns)}")
    if sample[ID_COL].isna().any() or sample[ID_COL].duplicated().any():
        raise ValueError("sample_submission row_id values must be non-null and unique.")
    if len(test) != len(sample) or set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("test.csv and sample_submission.csv row_id sets must match exactly.")

    expected = set(bundle["expected_raw_columns"])
    missing = expected - set(test.columns)
    if missing:
        raise ValueError(f"test.csv is missing trained columns: {sorted(missing)}")


def _load_gbdt_models():
    xgb_path = MODEL_DIR / "xgb_model.json"
    cat_path = MODEL_DIR / "cat_model.cbm"
    if not xgb_path.exists() or not cat_path.exists():
        raise FileNotFoundError("XGBoost and CatBoost model files are required.")

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(xgb_path))
    xgb_model.set_param({"nthread": 6, "device": "cpu"})
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(cat_path))
    return xgb_model, cat_model


def main() -> None:
    started = time.perf_counter()
    print("[1/8] Load immutable model bundle and runtimes")
    bundle = joblib.load(MODEL_DIR / "bundle.pkl")
    runtime = _load_module("runtime.py", "aimers_runtime")
    model_order = list(bundle["ensemble"]["model_order"])
    supported_orders = [
        ["xgb", "cat"],
        ["xgb", "cat", "resnet"],
        ["xgb", "cat", "resnet", "ft_transformer"],
    ]
    if model_order not in supported_orders:
        raise ValueError(f"Unsupported model order: {model_order}")
    neural = None
    neural_device = None
    if "resnet" in model_order or "ft_transformer" in model_order:
        neural = _load_module("neural_runtime.py", "aimers_neural_runtime")
        neural.torch.set_num_threads(6)
        neural_device = neural.resolve_device("auto")

    print("[2/8] Load and validate official inputs")
    test = _load_csv(DATA_DIR / "test.csv")
    sample = _load_csv(DATA_DIR / "sample_submission.csv")
    _validate_inputs(test, sample, bundle)
    row_ids = test[ID_COL].copy()

    print(f"[3/8] Build row-independent features: rows={len(test):,}")
    features = runtime.build_features(test, bundle["feature_state"])
    del test
    gc.collect()

    print("[4/8] Apply train-fitted encoding and imputation")
    X = runtime.preprocess_frame(features, bundle["preprocessor_state"])
    del features
    gc.collect()

    print(f"[5/8] Predict XGBoost + CatBoost: features={X.shape[1]}")
    xgb_model, cat_model = _load_gbdt_models()
    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
    xgb_prediction = np.asarray(xgb_model.predict(dtest), dtype=np.float64)
    cat_prediction = np.asarray(cat_model.predict_proba(X)[:, 1], dtype=np.float64)
    predictions = {
        "xgb": xgb_prediction,
        "cat": cat_prediction,
    }
    del dtest, xgb_model, cat_model
    gc.collect()

    if neural is not None:
        neural_names = model_order[2:]
        print(
            f"[6/8] Predict neural models: names={neural_names} "
            f"device={neural_device}"
        )
        neural_arrays = neural.prepare_neural_arrays(
            X, bundle["neural_preprocessor_state"]
        )
        if "resnet" in neural_names:
            resnet_model = neural.load_checkpoint(
                MODEL_DIR / "resnet.pt", device=neural_device
            )
            predictions["resnet"] = neural.predict_model(
                resnet_model,
                neural_arrays,
                device=neural_device,
                batch_size=8192,
            )
            del resnet_model
        if "ft_transformer" in neural_names:
            ft_model = neural.load_checkpoint(
                MODEL_DIR / "ft_transformer.pt", device=neural_device
            )
            predictions["ft_transformer"] = neural.predict_model(
                ft_model,
                neural_arrays,
                device=neural_device,
                batch_size=2048,
            )
            del ft_model
        del neural_arrays
    else:
        print("[6/8] Neural models absent: use validated GBDT-only fallback")
    del X
    gc.collect()

    print("[7/8] Blend and apply prequential calibration")
    ensemble = bundle["ensemble"]
    prediction = runtime.blend_predictions(
        [predictions[name] for name in model_order],
        ensemble["weights"],
    )
    calibration = ensemble["calibration"]
    if calibration.get("accepted", False):
        if calibration.get("method") != "logit_intercept":
            raise ValueError(f"Unsupported calibration: {calibration.get('method')}")
        prediction = runtime.apply_logit_intercept(
            prediction, float(calibration["intercept"])
        )

    if prediction.shape != (len(row_ids),):
        raise ValueError(f"Prediction shape mismatch: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinity.")
    if ((prediction <= 0.0) | (prediction >= 1.0)).any():
        raise ValueError("Prediction is outside the open probability interval (0, 1).")

    print("[8/8] Restore official sample order and write submission.csv")
    prediction_by_id = pd.DataFrame({ID_COL: row_ids, TARGET_COL: prediction})
    submission = sample[[ID_COL]].merge(
        prediction_by_id,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if submission[TARGET_COL].isna().any():
        raise ValueError("Submission merge produced missing predictions.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - started
    print(
        f"[DONE] {output_path} rows={len(submission):,} "
        f"pred_mean={submission[TARGET_COL].mean():.6f} "
        f"pred_std={submission[TARGET_COL].std(ddof=0):.6f} "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()

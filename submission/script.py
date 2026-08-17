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


def _load_runtime_module():
    path = MODEL_DIR / "runtime.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("aimers_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _validate_inputs(test: pd.DataFrame, sample: pd.DataFrame, bundle) -> None:
    if bundle.get("bundle_version") != 2:
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


def _load_models():
    xgb_path = MODEL_DIR / "xgb_model.json"
    cat_path = MODEL_DIR / "cat_model.cbm"
    if not xgb_path.exists() or not cat_path.exists():
        raise FileNotFoundError("Both XGBoost and CatBoost model files are required.")

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(xgb_path))
    xgb_model.set_param({"nthread": 6, "device": "cpu"})

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(cat_path))
    return xgb_model, cat_model


def main() -> None:
    started = time.perf_counter()
    print("[1/7] Load immutable model bundle and runtime")
    bundle = joblib.load(MODEL_DIR / "bundle.pkl")
    runtime = _load_runtime_module()

    print("[2/7] Load and validate official inputs")
    test = _load_csv(DATA_DIR / "test.csv")
    sample = _load_csv(DATA_DIR / "sample_submission.csv")
    _validate_inputs(test, sample, bundle)
    row_ids = test[ID_COL].copy()

    print(f"[3/7] Build row-independent features: rows={len(test):,}")
    features = runtime.build_features(test, bundle["feature_state"])
    del test
    gc.collect()

    print("[4/7] Apply train-fitted encoding and imputation")
    X = runtime.preprocess_frame(features, bundle["preprocessor_state"])
    del features
    gc.collect()

    print(f"[5/7] Predict XGBoost + CatBoost: features={X.shape[1]}")
    xgb_model, cat_model = _load_models()
    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
    xgb_prediction = np.asarray(xgb_model.predict(dtest), dtype=np.float64)
    cat_prediction = np.asarray(cat_model.predict_proba(X)[:, 1], dtype=np.float64)
    del dtest, X, xgb_model, cat_model
    gc.collect()

    print("[6/7] Blend and apply prequential calibration")
    ensemble = bundle["ensemble"]
    if list(ensemble["model_order"]) != ["xgb", "cat"]:
        raise ValueError(f"Unsupported model order: {ensemble['model_order']}")
    prediction = runtime.blend_predictions(
        [xgb_prediction, cat_prediction], ensemble["weights"]
    )
    calibration = ensemble["calibration"]
    if calibration.get("accepted", False):
        prediction = runtime.apply_probability_bias(
            prediction, float(calibration["bias"])
        )

    if prediction.shape != (len(row_ids),):
        raise ValueError(f"Prediction shape mismatch: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinity.")
    if ((prediction <= 0.0) | (prediction >= 1.0)).any():
        raise ValueError("Prediction is outside the open probability interval (0, 1).")

    print("[7/7] Restore official sample order and write submission.csv")
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

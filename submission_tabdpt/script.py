from __future__ import annotations

import gc
import importlib.util
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from tabdpt import TabDPTClassifier


ROOT = Path(".")
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
ID_COL = "row_id"
TARGET_COL = "control_success"


def _data_dir() -> Path:
    # DACON's baseline uses ./data.  Keep explicit fallbacks for evaluation
    # runners that mount the immutable archive as ./open or ./open/data.
    for candidate in (ROOT / "data", ROOT / "open", ROOT / "open" / "data"):
        if (candidate / "test.csv").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate test.csv under data/, open/ or open/data/.")


def _load_module(filename: str, module_name: str):
    path = MODEL_DIR / filename
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
    if bundle.get("bundle_version") != 1:
        raise ValueError(f"Unsupported TabDPT bundle: {bundle.get('bundle_version')}")
    if bundle.get("strategy") != "recent_context_tabdpt_turbo_xgb_v1":
        raise ValueError(f"Unexpected strategy: {bundle.get('strategy')}")
    if bundle.get("id_col") != ID_COL or bundle.get("target_col") != TARGET_COL:
        raise ValueError("Bundle ID/target contract does not match DACON.")
    if ID_COL not in test or TARGET_COL in test:
        raise ValueError("test.csv schema is invalid.")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Unexpected sample_submission columns: {list(sample.columns)}")
    if test[ID_COL].isna().any() or test[ID_COL].duplicated().any():
        raise ValueError("test row_id must be non-null and unique.")
    if sample[ID_COL].isna().any() or sample[ID_COL].duplicated().any():
        raise ValueError("sample_submission row_id must be non-null and unique.")
    if len(test) != len(sample) or set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("test/sample row_id sets do not match.")
    missing = set(bundle["expected_raw_columns"]) - set(test.columns)
    if missing:
        raise ValueError(f"test.csv is missing trained columns: {sorted(missing)}")


def _predict_tabdpt(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    config: dict,
) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("TabDPT candidate requires DACON's allocated L4 GPU.")
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = TabDPTClassifier(
        normalizer="standard",
        missing_indicators=False,
        feature_reduction="subsample",
        context_reduction="subsample",
        device="cuda",
        use_flash=bool(config["use_flash_attention"]),
        compile=bool(config["compile_model"]),
        model_weight_path=str(MODEL_DIR / "tabdpt1_2.safetensors"),
        verbose=False,
    )
    model.fit(context_x, context_y)

    initial_batch_size = int(config["inference_batch_size"])
    batch_size = initial_batch_size
    while True:
        try:
            if int(config["n_ensembles"]) == 1:
                probability = model.predict_proba(
                    query_x,
                    context_size=None,
                    batch_size=batch_size,
                    seed=int(config["random_seed"]),
                )
            else:
                probability = model.ensemble_predict_proba(
                    query_x,
                    n_ensembles=int(config["n_ensembles"]),
                    temperature=1.0,
                    context_size=None,
                    batch_size=batch_size,
                    permute_classes=True,
                    seed=int(config["random_seed"]),
                )
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size //= 2
            if batch_size < 4_096:
                raise
            print(
                f"[TABDPT] CUDA OOM at batch={batch_size * 2:,}; "
                f"retry batch={batch_size:,}"
            )

    if batch_size != initial_batch_size:
        print(f"[TABDPT] completed with adaptive batch={batch_size:,}")
    prediction = np.asarray(probability[:, 1], dtype=np.float64)
    del probability, model
    torch.cuda.empty_cache()
    gc.collect()
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def main() -> None:
    started = time.perf_counter()
    print("[1/9] Load immutable TabDPT candidate bundle")
    bundle = joblib.load(MODEL_DIR / "bundle.pkl")
    runtime = _load_module("runtime.py", "aimers_tabdpt_runtime")

    print("[2/9] Load and validate official inputs")
    data_dir = _data_dir()
    test = _load_csv(data_dir / "test.csv")
    sample = _load_csv(data_dir / "sample_submission.csv")
    _validate_inputs(test, sample, bundle)
    row_ids = test[ID_COL].copy()

    print(f"[3/9] Build row-independent features: rows={len(test):,}")
    features = runtime.build_features(test, bundle["feature_state"])
    del test
    gc.collect()

    print("[4/9] Apply train-fitted encoding and imputation")
    X = runtime.preprocess_frame(features, bundle["preprocessor_state"])
    del features
    gc.collect()

    print(f"[5/9] Predict XGBoost: features={X.shape[1]}")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "xgb_model.json"))
    booster.set_param({"nthread": 6, "device": "cpu"})
    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
    xgb_prediction = np.asarray(booster.predict(dtest), dtype=np.float64)
    del booster, dtest
    gc.collect()

    print("[6/9] Load fixed 2024 TabDPT context")
    with np.load(MODEL_DIR / "tabdpt_context.npz", allow_pickle=False) as context:
        context_x = np.ascontiguousarray(context["X"], dtype=np.float32)
        context_y = np.asarray(context["y"], dtype=np.int64)
        context_feature_names = context["feature_names"].astype(str).tolist()
    expected_names = list(bundle["tabdpt_feature_names"])
    if context_feature_names != expected_names:
        raise ValueError("TabDPT context feature contract does not match bundle.pkl.")
    if context_x.shape != (len(context_y), len(expected_names)):
        raise ValueError(f"Invalid bundled TabDPT context shape: {context_x.shape}")
    query_x = np.ascontiguousarray(
        X.loc[:, expected_names].to_numpy(dtype=np.float32, copy=True)
    )
    del X
    gc.collect()

    print(
        f"[7/9] Predict TabDPT-Turbo: context={len(context_x):,} "
        f"features={context_x.shape[1]} queries={len(query_x):,}"
    )
    tabdpt_prediction = _predict_tabdpt(
        context_x,
        context_y,
        query_x,
        dict(bundle["tabdpt_config"]),
    )
    del context_x, context_y, query_x
    gc.collect()

    print("[8/9] Apply validated blend and prequential calibration")
    ensemble = bundle["ensemble"]
    if ensemble["model_order"] != ["xgb", "tabdpt"]:
        raise ValueError(f"Unexpected model order: {ensemble['model_order']}")
    prediction = runtime.blend_predictions(
        [xgb_prediction, tabdpt_prediction], ensemble["weights"]
    )
    calibration = ensemble["calibration"]
    if calibration.get("accepted", False):
        if calibration.get("method") != "logit_intercept":
            raise ValueError(f"Unsupported calibration: {calibration.get('method')}")
        prediction = runtime.apply_logit_intercept(
            prediction, float(calibration["intercept"])
        )
    if prediction.shape != (len(row_ids),) or not np.isfinite(prediction).all():
        raise ValueError("Final prediction is invalid.")

    print("[9/9] Restore sample order and write submission.csv")
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
        f"mean={submission[TARGET_COL].mean():.6f} "
        f"std={submission[TARGET_COL].std(ddof=0):.6f} elapsed={elapsed:.2f}s"
    )
    if elapsed > 540.0:
        print("[WARN] Runtime exceeded the 9-minute safety target.")


if __name__ == "__main__":
    main()

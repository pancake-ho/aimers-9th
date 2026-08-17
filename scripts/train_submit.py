from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.features import LeakageSafeFeatureEngineer, StrictPastTrackmanFeatures
from src.submission_model import train_lightgbm_full
from src.preprocessing import TabularPreprocessor
from src.splits import make_recency_weights


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recency-lambda", type=float, default=0.0)
    p.add_argument("--num-boost-round", type=int, default=180)
    p.add_argument("--no-trackman", action="store_true")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(
        recency_lambda=args.recency_lambda,
        use_trackman=not args.no_trackman,
    )
    build_dir = cfg.paths.submission_build_dir
    model_dir = build_dir / "model"
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] Loading {cfg.paths.train_path}")
    train = pd.read_csv(cfg.paths.train_path, encoding="utf-8-sig", low_memory=False)
    target = cfg.features.target_col
    if target not in train.columns:
        raise ValueError(f"Missing target column: {target}")

    y = train[target].to_numpy(dtype=np.float32, copy=True)
    seasons = train["season"].to_numpy(dtype=np.int16, copy=True)

    trackman = None
    if cfg.use_trackman:
        trackman = StrictPastTrackmanFeatures().fit_from_csv(cfg.paths.trackman_path)

    fe = LeakageSafeFeatureEngineer(cfg.features, trackman_features=trackman)
    print("[FEATURE] Transform full 2019~2024 train")
    features = fe.transform(train)

    pp = TabularPreprocessor(
        categorical_cols=cfg.features.categorical_cols,
        excluded_cols=cfg.features.excluded_cols,
    )
    X = pp.fit_transform(features)
    del features
    gc.collect()

    weights = make_recency_weights(
        seasons,
        max_train_season=int(seasons.max()),
        decay_lambda=args.recency_lambda,
    )
    print(
        f"[WEIGHT] lambda={args.recency_lambda}, "
        f"min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}"
    )

    start = time.perf_counter()
    model = train_lightgbm_full(
        X_train=X,
        y_train=y,
        sample_weight=weights,
        categorical_cols=pp.cat_cols_,
        config=cfg.models,
        num_boost_round=args.num_boost_round,
    )
    train_seconds = time.perf_counter() - start

    model_path = model_dir / "lgb_model.txt"
    model.save_model(str(model_path))

    raw_feature_cols = [c for c in train.columns if c != target]
    bundle = {
        "bundle_version": 1,
        "id_col": cfg.features.id_col,
        "target_col": target,
        "expected_raw_columns": raw_feature_cols,
        "feature_state": fe.export_runtime_state(),
        "preprocessor_state": pp.export_state(),
        "training": {
            "train_seasons": [int(seasons.min()), int(seasons.max())],
            "n_rows": int(len(train)),
            "recency_lambda": float(args.recency_lambda),
            "num_boost_round": int(args.num_boost_round),
            "target_mean": float(y.mean()),
            "random_seed": int(cfg.models.random_seed),
        },
    }
    bundle_path = model_dir / "bundle.pkl"
    joblib.dump(bundle, bundle_path, compress=3)

    manifest = {
        "model": "LightGBM",
        "feature_version": "v1_fixed_prior_no_cluster_no_zscore",
        "model_file": model_path.name,
        "bundle_file": bundle_path.name,
        "recency_lambda": float(args.recency_lambda),
        "num_boost_round": int(args.num_boost_round),
        "n_features": int(X.shape[1]),
        "train_seconds": float(train_seconds),
        "use_trackman": bool(cfg.use_trackman),
    }
    with open(model_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[DONE] model={model_path}")
    print(f"[DONE] bundle={bundle_path}")
    print(f"[DONE] n_features={X.shape[1]}, train_seconds={train_seconds:.2f}")


if __name__ == "__main__":
    main()

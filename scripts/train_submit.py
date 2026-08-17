from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, ModelConfig
from src.data import load_csv, validate_train_schema
from src.models import validate_xgboost_backend
from src.training import (
    build_feature_table,
    run_temporal_validation,
    train_and_save_final_models,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate, calibrate and train the next Aimers submit.zip models."
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-trackman", action="store_true")
    parser.add_argument(
        "--cat-task-type",
        choices=("CPU", "GPU"),
        default="CPU",
        help="GPU is faster for local training; CPU is the reproducible default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        use_trackman=not args.no_trackman,
        models=ModelConfig(cat_task_type=args.cat_task_type),
    )
    validate_xgboost_backend(config.models)
    build_dir = config.paths.submission_build_dir
    model_dir = build_dir / "model"
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train = load_csv(config.paths.train_path)
    validate_train_schema(
        train,
        target_col=config.features.target_col,
        id_col=config.features.id_col,
    )
    season_stats = (
        train.groupby("season")[config.features.target_col]
        .agg(["size", "mean"])
    )
    print("[DATA] target drift by season:\n" + season_stats.to_string())

    features, feature_state = build_feature_table(train, config)
    ensemble_state, validation_report = run_temporal_validation(
        train, features, config
    )
    with open(model_dir / "validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(validation_report, handle, indent=2, ensure_ascii=False)

    manifest = train_and_save_final_models(
        train=train,
        features=features,
        feature_state=feature_state,
        ensemble_state=ensemble_state,
        config=config,
        model_dir=model_dir,
    )
    del features, train
    gc.collect()

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 88)
    print(f"[DONE] model_dir={model_dir}")
    print(f"[DONE] weights={manifest['weights']}")
    print(f"[DONE] calibration={manifest['calibration']}")
    print(f"[DONE] total_train_seconds={elapsed:.2f}")
    print("[NEXT] python3 scripts/build_submit.py")
    print("=" * 88)


if __name__ == "__main__":
    main()

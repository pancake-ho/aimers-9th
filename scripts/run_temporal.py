from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, ModelConfig, NeuralConfig
from src.data import load_csv, validate_train_schema
from src.models import validate_xgboost_backend
from src.neural import validate_neural_backend
from src.training import build_feature_table, run_temporal_validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-trackman", action="store_true")
    parser.add_argument("--cat-task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nn-device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "next_submit_temporal_report.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        use_trackman=not args.no_trackman,
        models=ModelConfig(
            cat_task_type=args.cat_task_type,
            xgb_device=args.xgb_device,
        ),
        neural=NeuralConfig(device=args.nn_device),
    )
    validate_xgboost_backend(config.models)
    validate_neural_backend(config.neural)
    train = load_csv(config.paths.train_path)
    validate_train_schema(
        train,
        target_col=config.features.target_col,
        id_col=config.features.id_col,
    )
    features, _ = build_feature_table(train, config)
    _, report = run_temporal_validation(train, features, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()

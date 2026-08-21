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

from src.config import (
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    NeuralConfig,
)
from src.data import load_csv, validate_train_schema
from src.features import validate_feature_config_contract
from src.models import validate_xgboost_backend
from src.tabfm.config import TabDPTExperimentConfig
from src.tabfm.tabdpt_backend import validate_tabdpt_backend
from src.tabfm.training import (
    run_tabdpt_temporal_validation,
    train_and_save_tabdpt_candidate,
)
from src.training import build_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a recent-context TabDPT-Turbo + XGBoost candidate and "
            "train submission artifacts only when it improves forward folds."
        )
    )
    parser.add_argument("--model-weight-path", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-trackman", action="store_true")
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tabdpt-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--context-size", type=int, default=32_768)
    parser.add_argument("--n-ensembles", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=65_536)
    parser.add_argument(
        "--context-strategy",
        choices=(
            "recent_proportional",
            "representative_v1",
        ),
        default="representative_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = ExperimentConfig(
        features=FeatureConfig(
            trackman_entity_enabled=False,
        ),
        use_trackman=not args.no_trackman,
        use_main_history=False,
        models=ModelConfig(
            xgb_device=args.xgb_device,
        ),
        neural=NeuralConfig(
            enabled=False,
        ),
    )
    tabdpt_config = (
        TabDPTExperimentConfig(
            context_size=(
                args.context_size
            ),
            n_ensembles=(
                args.n_ensembles
            ),
            inference_batch_size=(
                args.inference_batch_size
            ),
            context_strategy=(
                args.context_strategy
            ),
        )
    )
    print(
        "[CONFIG] "
        f"context_strategy="
        f"{tabdpt_config.context_strategy} "
        f"context_size="
        f"{tabdpt_config.context_size} "
        f"fractions=("
        f"{tabdpt_config.representative_recent_fraction:.2f},"
        f"{tabdpt_config.representative_pitcher_fraction:.2f},"
        f"{tabdpt_config.representative_situation_fraction:.2f})"
    )
    tabdpt_config.validate()
    validate_feature_config_contract(experiment_config.features)
    validate_xgboost_backend(experiment_config.models)
    validate_tabdpt_backend(args.model_weight_path, device=args.tabdpt_device)

    build_dir = PROJECT_ROOT / "tabdpt_submission_build"
    model_dir = build_dir / "model"
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train = load_csv(experiment_config.paths.train_path)
    validate_train_schema(
        train,
        target_col=experiment_config.features.target_col,
        id_col=experiment_config.features.id_col,
    )
    print(
        "[DATA] target drift by season:\n"
        + train.groupby("season")[experiment_config.features.target_col]
        .agg(["size", "mean"])
        .to_string()
    )
    features, feature_state = build_feature_table(train, experiment_config)
    candidate_state, report = run_tabdpt_temporal_validation(
        train,
        features,
        experiment_config=experiment_config,
        tabdpt_config=tabdpt_config,
        weight_path=args.model_weight_path,
        device=args.tabdpt_device,
    )
    report_path = model_dir / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    gate = report["submission_gate"]
    print("[GATE] " + json.dumps(gate, ensure_ascii=False))
    if not gate["passed"]:
        raise RuntimeError(
            "TabDPT candidate failed the predeclared forward-validation gate. "
            f"Inspect {report_path}; no submit.zip will be built."
        )

    manifest = train_and_save_tabdpt_candidate(
        train,
        features,
        feature_state,
        candidate_state,
        experiment_config=experiment_config,
        tabdpt_config=tabdpt_config,
        source_weight_path=args.model_weight_path,
        model_dir=model_dir,
    )
    del train, features
    gc.collect()

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 88)
    print(f"[DONE] strategy={manifest['strategy']}")
    print(f"[DONE] weights={manifest['weights']}")
    print(f"[DONE] total_seconds={elapsed:.2f}")
    print("[NEXT] python scripts/build_tabdpt_submit.py")
    print("=" * 88)


if __name__ == "__main__":
    main()

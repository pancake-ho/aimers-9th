from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.comparison import run_pitcher_residual_comparison
from src.baseline.pitcher_residual import validate_residual_xgboost_backend
from src.config import ExperimentConfig, ModelConfig, NeuralConfig
from src.data import load_csv, validate_train_schema
from src.features import validate_feature_config_contract
from src.models import validate_xgboost_backend
from src.training import build_feature_table


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current direct XGBoost with leakage-safe pitcher-logit "
            "offset residual XGBoost variants."
        )
    )
    parser.add_argument("--no-trackman", action="store_true")
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prior-probability", type=float, default=0.5)
    parser.add_argument(
        "--context-strengths",
        nargs="+",
        type=float,
        default=(25.0, 100.0, 500.0),
        help="Shrinkage strengths for strict context-only residual models.",
    )
    parser.add_argument(
        "--full-strength",
        type=float,
        default=100.0,
        help="Shrinkage strength for the all-feature residual ablation.",
    )
    parser.add_argument(
        "--no-full-residual",
        action="store_true",
        help="Skip the all-feature residual ablation.",
    )
    parser.add_argument("--probability-clip", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test only: 250 rounds and 200 bootstrap replicates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pitcher_residual_comparison.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = ModelConfig(xgb_device=args.xgb_device)
    bootstrap_samples = int(args.bootstrap_samples)
    if args.quick:
        model_config = replace(
            model_config,
            xgb_num_boost_round=250,
            xgb_early_stopping_rounds=30,
        )
        bootstrap_samples = 200

    config = ExperimentConfig(
        use_trackman=not args.no_trackman,
        use_main_history=False,
        models=model_config,
        neural=NeuralConfig(enabled=False, device="cpu"),
    )
    validate_feature_config_contract(config.features)
    validate_xgboost_backend(config.models)
    validate_residual_xgboost_backend(config.models)

    train = load_csv(config.paths.train_path)
    validate_train_schema(
        train,
        target_col=config.features.target_col,
        id_col=config.features.id_col,
    )
    features, _ = build_feature_table(train, config)
    report = run_pitcher_residual_comparison(
        train,
        features,
        config,
        prior_probability=float(args.prior_probability),
        context_strengths=tuple(float(value) for value in args.context_strengths),
        full_strength=(None if args.no_full_residual else float(args.full_strength)),
        probability_clip=float(args.probability_clip),
        bootstrap_samples=bootstrap_samples,
    )
    report["execution"] = {
        "quick": bool(args.quick),
        "xgb_device": str(args.xgb_device),
        "trackman_enabled": bool(config.use_trackman),
        "xgb_num_boost_round": int(config.models.xgb_num_boost_round),
        "xgb_early_stopping_rounds": int(
            config.models.xgb_early_stopping_rounds
        ),
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print("\n" + "=" * 88)
    print("[RESIDUAL-SUMMARY] weighted Brier and replacement decision")
    print("=" * 88)
    reference = float(report["weighted_brier"]["direct_xgb"])
    for name, decision in report["decisions"].items():
        value = float(report["weighted_brier"][name])
        print(
            f"[RESIDUAL-SUMMARY] {name}: brier={value:.8f} "
            f"delta={value - reference:+.8f} status={decision['status']}"
        )
    print(f"[DONE] report={output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping


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


from src.config import (
    ExperimentConfig,
    ModelConfig,
    NeuralConfig,
)

from src.data import (
    load_csv,
    validate_train_schema,
)

from src.features import (
    validate_feature_config_contract,
)

from src.models import (
    validate_lightgbm_backend,
    validate_xgboost_backend,
)

from src.training import (
    build_feature_table,
    run_temporal_validation,
    train_and_save_final_models,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate, calibrate, train, and "
            "package the next Aimers model."
        )
    )

    parser.add_argument(
        "--clean",
        action="store_true",
    )

    parser.add_argument(
        "--no-trackman",
        action="store_true",
    )

    parser.add_argument(
        "--cat-task-type",
        choices=(
            "CPU",
            "GPU",
        ),
        default="CPU",
        help=(
            "GPU is faster for local training; "
            "CPU remains available for debugging."
        ),
    )

    parser.add_argument(
        "--xgb-device",
        choices=(
            "cpu",
            "cuda",
        ),
        default="cuda",
        help=(
            "Training backend for XGBoost. "
            "Submission inference remains CPU-capped."
        ),
    )

    parser.add_argument(
        "--nn-device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="cuda",
        help=(
            "Training device for ResNet and "
            "FT-Transformer."
        ),
    )

    parser.add_argument(
        "--disable-neural",
        action="store_true",
        help=(
            "Disable neural models for "
            "explicit debugging only."
        ),
    )

    parser.add_argument(
        "--require-quality-gate",
        action="store_true",
        help=(
            "Optional research-only strict mode. "
            "When enabled, a failed validation "
            "quality gate stops final training. "
            "Default production behavior is to "
            "always build an artifact."
        ),
    )

    return parser.parse_args()


def evaluate_quality_gate_policy(
    submission_gate: Mapping[
        str,
        object,
    ],
    *,
    require_quality_gate: bool,
) -> list[str]:
    """Validate gate consistency and return failed checks.

    Quality checks are advisory by default.

    Structural, leakage, dependency, runtime, feature-contract,
    and serialization errors continue to raise normally elsewhere.

    The optional strict mode exists only for research/debug runs.
    """

    checks_object = submission_gate.get(
        "checks"
    )

    if not isinstance(
        checks_object,
        Mapping,
    ):
        raise ValueError(
            "submission_gate['checks'] "
            "must be a mapping."
        )

    checks = {
        str(name): bool(value)
        for name, value
        in checks_object.items()
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    derived_passed = (
        len(failed_checks) == 0
    )

    reported_passed = bool(
        submission_gate.get(
            "passed",
            False,
        )
    )

    if (
        reported_passed
        != derived_passed
    ):
        raise ValueError(
            "Quality-gate report is "
            "internally inconsistent: "
            f"reported_passed="
            f"{reported_passed} "
            f"failed_checks="
            f"{failed_checks}"
        )

    if (
        failed_checks
        and require_quality_gate
    ):
        raise RuntimeError(
            "Validation quality gate failed "
            "in explicit strict mode: "
            f"{failed_checks}"
        )

    return failed_checks


def main():
    args = parse_args()

    config = ExperimentConfig(
        use_trackman=(
            not args.no_trackman
        ),
        models=ModelConfig(
            cat_task_type=(
                args.cat_task_type
            ),
            xgb_device=(
                args.xgb_device
            ),
        ),
        neural=NeuralConfig(
            enabled=(
                not args.disable_neural
            ),
            device=(
                args.nn_device
            ),
        ),
    )

    # --------------------------------------------------------
    # Hard structural/runtime contracts.
    #
    # These remain fatal.  The new artifact policy applies
    # only to model-quality thresholds, never to correctness.
    # --------------------------------------------------------

    validate_feature_config_contract(
        config.features
    )

    validate_lightgbm_backend(
        config.models
    )

    validate_xgboost_backend(
        config.models
    )

    if config.neural.enabled:
        from src.neural import (
            validate_neural_backend,
        )

        validate_neural_backend(
            config.neural
        )

    else:
        print(
            "[BACKEND] Neural models disabled; "
            "explicit debug configuration."
        )

    build_dir = (
        config.paths
        .submission_build_dir
    )

    model_dir = (
        build_dir
        / "model"
    )

    if (
        args.clean
        and build_dir.exists()
    ):
        shutil.rmtree(
            build_dir
        )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train = load_csv(
        config.paths.train_path
    )

    validate_train_schema(
        train,
        target_col=(
            config.features.target_col
        ),
        id_col=(
            config.features.id_col
        ),
    )

    season_stats = (
        train
        .groupby("season")[
            config.features.target_col
        ]
        .agg(
            [
                "size",
                "mean",
            ]
        )
    )

    print(
        "[DATA] target drift by season:\n"
        + season_stats.to_string()
    )

    # --------------------------------------------------------
    # Feature build
    # --------------------------------------------------------

    (
        features,
        feature_state,
    ) = build_feature_table(
        train,
        config,
    )

    # --------------------------------------------------------
    # Temporal validation
    # --------------------------------------------------------

    (
        ensemble_state,
        validation_report,
    ) = run_temporal_validation(
        train,
        features,
        config,
    )

    submission_gate = (
        validation_report[
            "submission_gate"
        ]
    )

    failed_checks = (
        evaluate_quality_gate_policy(
            submission_gate,
            require_quality_gate=(
                args.require_quality_gate
            ),
        )
    )

    # --------------------------------------------------------
    # Artifact-generation policy.
    #
    # IMPORTANT:
    # Quality score thresholds do NOT block model packaging.
    #
    # We retain the exact gate result in validation_report
    # so the eventual upload decision can be made separately.
    # --------------------------------------------------------

    validation_report[
        "artifact_policy"
    ] = {
        "mode": (
            "strict"
            if args.require_quality_gate
            else "package_always"
        ),
        "quality_gate_blocks_packaging": bool(
            args.require_quality_gate
        ),
        "quality_gate_passed": bool(
            submission_gate[
                "passed"
            ]
        ),
        "failed_quality_checks": (
            list(
                failed_checks
            )
        ),
    }

    # Keep quality provenance with the trained artifact too.
    ensemble_state = dict(
        ensemble_state
    )

    ensemble_state[
        "quality_assessment"
    ] = dict(
        submission_gate
    )

    # --------------------------------------------------------
    # Save validation report BEFORE final training.
    # --------------------------------------------------------

    report_path = (
        model_dir
        / "validation_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            validation_report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[QUALITY] "
        + json.dumps(
            submission_gate,
            ensure_ascii=False,
        )
    )

    if failed_checks:
        print(
            "[QUALITY-WARN] "
            "Validation quality checks failed: "
            f"{failed_checks}"
        )

        print(
            "[ARTIFACT-POLICY] "
            "Continuing final training and "
            "submit.zip creation. "
            "Quality gates are advisory; "
            "correctness/safety contracts "
            "remain mandatory."
        )

    else:
        print(
            "[QUALITY] "
            "All configured validation "
            "quality checks passed."
        )

        print(
            "[ARTIFACT-POLICY] "
            "Proceeding to final training "
            "and submit.zip creation."
        )

    # --------------------------------------------------------
    # Final full-data training.
    #
    # This now runs regardless of advisory quality score.
    # --------------------------------------------------------

    manifest = (
        train_and_save_final_models(
            train=train,
            features=features,
            feature_state=(
                feature_state
            ),
            ensemble_state=(
                ensemble_state
            ),
            config=config,
            model_dir=model_dir,
        )
    )

    del (
        features,
        train,
    )

    gc.collect()

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        "=" * 88
    )

    print(
        f"[DONE] model_dir="
        f"{model_dir}"
    )

    print(
        f"[DONE] weights="
        f"{manifest['weights']}"
    )

    print(
        f"[DONE] calibration="
        f"{manifest['calibration']}"
    )

    print(
        "[DONE] quality_gate_passed="
        f"{submission_gate['passed']}"
    )

    print(
        "[DONE] failed_quality_checks="
        f"{failed_checks}"
    )

    print(
        f"[DONE] total_train_seconds="
        f"{elapsed:.2f}"
    )

    print(
        "[NEXT] "
        "python3 scripts/build_submit.py"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
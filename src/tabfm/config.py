from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TabDPTExperimentConfig:
    """Resource and validation contract for the TabDPT-Turbo candidate.

    The public TabDPT-Turbo checkpoint was trained with contexts up to 32K
    rows and accepts at most 128 features.  The Aimers feature pipeline
    currently produces 127 numerical features, so this experiment deliberately
    drops ordinal-coded categoricals instead of applying PCA to arbitrary ID
    codes.
    """

    context_size: int = 32_768
    max_features: int = 128
    n_ensembles: int = 2
    inference_batch_size: int = 65_536
    random_seed: int = 2026
    use_flash_attention: bool = True
    compile_model: bool = False

    ensemble_grid_step: float = 0.025
    fold_importance: tuple[float, ...] = (0.15, 0.35, 0.50)
    protected_folds: tuple[str, ...] = ("2024", "2024_late_abs")
    non_degradation_tolerance: float = 0.0
    minimum_tabdpt_weight: float = 0.025
    minimum_weighted_gain: float = 0.00002
    maximum_2023_brier: float = 0.25018
    # Temporal validation runs on an RTX 3090, while DACON uses an L4 and has
    # a hard 600-second inference limit.  Extrapolate the full-2024 prediction
    # wall time with a deliberately conservative memory-bandwidth multiplier,
    # then reserve time for CSV/features/XGBoost/output.
    l4_runtime_multiplier: float = 2.25
    non_tabdpt_runtime_reserve_seconds: float = 75.0
    maximum_estimated_runtime_seconds: float = 540.0

    context_strategy: str = (
    "representative_v1"
    )

    representative_recent_fraction: float = 0.70
    representative_pitcher_fraction: float = 0.20
    representative_situation_fraction: float = 0.10

    def validate(self) -> None:
        if self.context_size < 1_024:
            raise ValueError("TabDPT context_size must be at least 1,024.")
        if self.max_features < 1 or self.max_features > 128:
            raise ValueError("TabDPT-Turbo supports at most 128 features.")
        if self.n_ensembles < 1:
            raise ValueError("n_ensembles must be positive.")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive.")
        if not 0.0 < self.ensemble_grid_step <= 1.0:
            raise ValueError("ensemble_grid_step must be in (0, 1].")
        if not 0.0 <= self.minimum_tabdpt_weight <= 1.0:
            raise ValueError("minimum_tabdpt_weight must be in [0, 1].")
        if self.minimum_weighted_gain < 0.0:
            raise ValueError("minimum_weighted_gain must be non-negative.")
        if self.l4_runtime_multiplier < 1.0:
            raise ValueError("l4_runtime_multiplier must be at least one.")
        if self.non_tabdpt_runtime_reserve_seconds < 0.0:
            raise ValueError("Runtime reserve must be non-negative.")
        if not 0.0 < self.maximum_estimated_runtime_seconds < 600.0:
            raise ValueError("Runtime gate must stay below DACON's 600 seconds.")
        if self.context_strategy not in {
            "recent_proportional",
            "representative_v1",
        }:
            raise ValueError(
                "Unsupported TabDPT context strategy: "
                f"{self.context_strategy}"
            )

        context_fractions = (
            self.representative_recent_fraction
            + self.representative_pitcher_fraction
            + self.representative_situation_fraction
        )

        if not abs(
            context_fractions - 1.0
        ) <= 1e-9:
            raise ValueError(
                "Representative context fractions "
                "must sum to one."
            )
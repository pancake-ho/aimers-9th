from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PitcherResidualSourceContractTests(unittest.TestCase):
    def test_residual_uses_logit_offset_not_probability_residual_target(self) -> None:
        source = (PROJECT_ROOT / "src" / "baseline" / "pitcher_residual.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("base_margin=margin", source)
        self.assertIn("label=np.asarray(y_train", source)
        self.assertNotIn("y_train -", source)
        self.assertNotIn("y_valid -", source)

    def test_comparison_keeps_current_direct_xgb_as_reference(self) -> None:
        source = (PROJECT_ROOT / "src" / "baseline" / "comparison.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('DIRECT_MODEL_NAME = "direct_xgb"', source)
        self.assertIn("train_xgboost_fold(", source)
        self.assertIn("train_residual_xgboost_fold(", source)
        self.assertIn("make_abs_late_fold(", source)
        self.assertIn("clustered_brier_delta_bootstrap(", source)

    def test_slurm_job_does_not_pin_or_exclude_nodes(self) -> None:
        source = (PROJECT_ROOT / "run" / "run_residual_compare.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("#SBATCH -w", source)
        self.assertNotIn("#SBATCH --nodelist", source)
        self.assertNotIn("#SBATCH --exclude", source)
        self.assertIn("scripts/compare_pitcher_residual.py", source)
        self.assertIn("--context-strengths 25 100 500", source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RunScriptContractTests(unittest.TestCase):
    def test_gpu_job_is_not_pinned_to_a_single_node(self) -> None:
        script = (PROJECT_ROOT / "run" / "run.sh").read_text(encoding="utf-8")

        self.assertNotIn("#SBATCH -w ", script)
        self.assertNotIn("#SBATCH --nodelist=", script)
        self.assertNotIn("#SBATCH --exclude=", script)
        self.assertNotIn("requested_node=", script)
        self.assertNotIn("excluded_nodes=", script)
        self.assertNotIn("node_constraints=", script)
        self.assertNotIn("moana-y3", script)
        self.assertNotIn("moana-y5", script)

    def test_cuda_driver_is_probed_before_data_staging(self) -> None:
        script = (PROJECT_ROOT / "run" / "run.sh").read_text(encoding="utf-8")

        self.assertIn("cuda.cuInit(0)", script)
        self.assertIn("/dev/nvidia-uvm", script)
        self.assertLess(
            script.index("cuda.cuInit(0)"),
            script.index("[STAGE] Copying source code"),
        )

    def test_gpu_failure_selects_cpu_fallback_instead_of_exiting(self) -> None:
        script = (PROJECT_ROOT / "run" / "run.sh").read_text(encoding="utf-8")

        self.assertIn("[MODE] gbdt_cpu_fallback", script)
        self.assertIn('TRAIN_ARGS+=("--disable-neural")', script)
        self.assertIn("Four-model training failed after CUDA preflight", script)
        self.assertIn("retry command: python scripts/train_submit.py", script)
        self.assertNotIn("exit 70", script)
        self.assertNotIn("exit 71", script)
        self.assertNotIn("exit 72", script)

    def test_submission_requirements_are_checked_before_training(self) -> None:
        script = (PROJECT_ROOT / "run" / "run.sh").read_text(encoding="utf-8")

        self.assertIn("_validate_submission_requirements", script)
        self.assertLess(
            script.index("_validate_submission_requirements"),
            script.index("python scripts/train_submit.py"),
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


class RunScriptContractTests(
    unittest.TestCase
):
    def _script(self) -> str:
        return (
            PROJECT_ROOT
            / "run"
            / "run.sh"
        ).read_text(
            encoding="utf-8"
        )

    def test_gpu_job_is_not_pinned_to_a_single_node(
        self,
    ) -> None:
        script = self._script()

        self.assertNotIn(
            "#SBATCH -w ",
            script,
        )

        self.assertNotIn(
            "#SBATCH --nodelist=",
            script,
        )

        self.assertNotIn(
            "requested_node=",
            script,
        )

        self.assertNotIn(
            "moana-y3",
            script,
        )

    def test_dataset_staging_does_not_duplicate_archive_and_csvs(
        self,
    ) -> None:
        script = self._script()

        # Training must consume the immutable extracted data cache
        # through the existing configuration contract.
        self.assertIn(
            'export AIMERS_DATA_DIR="${DATA_CACHE_DIR}"',
            script,
        )

        # Do not recreate the old high-disk-footprint pipeline:
        #
        # open.zip copy
        # -> extracted copy
        # -> baseline/data second copy
        self.assertNotIn(
            'cp -- "${DATA_ARCHIVE}" "${LOCAL_JOB_ROOT}/open.zip"',
            script,
        )

        self.assertNotIn(
            'rsync -a "${LOCAL_JOB_ROOT}/open/data/"',
            script,
        )

        self.assertIn(
            "MIN_LOCAL_FREE_BYTES",
            script,
        )

        self.assertIn(
            "shared-fallback",
            script,
        )

        self.assertIn(
            "dataset_cache_ready",
            script,
        )

    def test_cuda_driver_is_probed_before_data_staging(
        self,
    ) -> None:
        script = self._script()

        self.assertIn(
            "cuda.cuInit(0)",
            script,
        )

        self.assertIn(
            "/dev/nvidia-uvm",
            script,
        )

        self.assertLess(
            script.index(
                "cuda.cuInit(0)"
            ),
            script.index(
                "[STAGE] Copying source code"
            ),
        )

    def test_pytorch_compatibility_is_functionally_probed(
        self,
    ) -> None:
        script = self._script()

        self.assertIn(
            "torch.cuda.is_available()",
            script,
        )

        self.assertIn(
            "y.backward()",
            script,
        )

        # Exact local wheel equality must not
        # replace an actual CUDA compatibility test.
        self.assertNotIn(
            "TORCH_BUILD_OK",
            script,
        )

        self.assertNotIn(
            'torch_base_version != "2.5.1"',
            script,
        )
        
    def test_dataset_cache_uses_bash_arithmetic_not_subshells(
        self,
    ) -> None:
        script = self._script()

        self.assertIn(
            "REQUIRED_CACHE_BYTES=$((",
            script,
        )

        self.assertIn(
            "UNCOMPRESSED_BYTES + CACHE_SAFETY_BYTES",
            script,
        )

        self.assertIn(
            "if (( SHARED_FREE_BYTES < REQUIRED_CACHE_BYTES )); then",
            script,
        )

        self.assertNotIn(
            "REQUIRED_CACHE_BYTES=$(\n            (",
            script,
        )

    def test_gpu_failure_refuses_a_silent_weaker_fallback(
        self,
    ) -> None:
        script = self._script()

        self.assertIn(
            "[MODE] "
            "xgb+lgb+cat+resnet+"
            "ft_transformer temporal_v9",
            script,
        )

        self.assertIn(
            "--exclude=moana-y5",
            script,
        )

        self.assertIn(
            "preserve_diagnostics",
            script,
        )

        self.assertIn(
            '"--nn-device" "cuda"',
            script,
        )

        self.assertIn(
            "No valid CUDA backend",
            script,
        )

        self.assertNotIn(
            '"--disable-neural"',
            script,
        )

    def test_submission_requirements_are_checked_before_training(
        self,
    ) -> None:
        script = self._script()

        self.assertIn(
            "_validate_submission_requirements",
            script,
        )

        self.assertLess(
            script.index(
                "_validate_submission_requirements"
            ),
            script.index(
                "python scripts/train_submit.py"
            ),
        )


if __name__ == "__main__":
    unittest.main()
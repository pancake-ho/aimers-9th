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

        # Normalize formatting-only whitespace.
        #
        # Shell arithmetic remains equivalent whether operands are written
        # on one line or split over several lines.  This test should protect
        # semantics, not code formatting.
        normalized = " ".join(
            script.split()
        )

        # Correct arithmetic expansion:
        #
        # REQUIRED_CACHE_BYTES=$((
        #     UNCOMPRESSED_BYTES
        #     + CACHE_SAFETY_BYTES
        # ))
        self.assertIn(
            (
                "REQUIRED_CACHE_BYTES=$(( "
                "UNCOMPRESSED_BYTES "
                "+ CACHE_SAFETY_BYTES "
                "))"
            ),
            normalized,
        )

        # Correct arithmetic conditional.
        self.assertIn(
            (
                "if (( "
                "SHARED_FREE_BYTES "
                "< REQUIRED_CACHE_BYTES "
                ")); then"
            ),
            normalized,
        )

        # Regression guard for the original job-136323 bug:
        #
        # $(...) is command substitution, whereas
        # $((...)) is arithmetic expansion.
        self.assertNotIn(
            (
                "REQUIRED_CACHE_BYTES=$( ( "
                "UNCOMPRESSED_BYTES "
                "+ CACHE_SAFETY_BYTES "
                ") )"
            ),
            normalized,
        )
    
    def test_dataset_archive_contract_uses_python_zipfile(
        self,
    ) -> None:
        script = self._script()

        # Human-readable `unzip -l` output must not be parsed
        # to determine whether the official data files exist.
        self.assertNotIn(
            'unzip -l "${DATA_ARCHIVE}"',
            script,
        )

        # Archive validation/extraction must use Python's
        # deterministic standard-library ZIP API.
        self.assertIn(
            "import zipfile",
            script,
        )

        self.assertIn(
            "zipfile.ZipFile",
            script,
        )

        self.assertIn(
            "archive.getinfo(name).file_size",
            script,
        )

        self.assertIn(
            '"data/train.csv"',
            script,
        )

        self.assertIn(
            '"data/test.csv"',
            script,
        )

        self.assertIn(
            '"data/sample_submission.csv"',
            script,
        )

        self.assertIn(
            '"data/trackman_history.csv"',
            script,
        )

        # Extraction should only publish a cache after the
        # four official files were checked.
        self.assertIn(
            'validate_data_dir "${TEMP_CACHE}/data"',
            script,
        )

    def test_gpu_failure_refuses_a_silent_weaker_fallback(
        self,
    ) -> None:
        script = self._script()

        # This test protects execution semantics only.
        # Strategy/version naming is checked separately.
        self.assertIn(
            "--exclude=moana-y5",
            script,
        )

        # Validation diagnostics must survive
        # a deliberate quality-gate failure.
        self.assertIn(
            "preserve_diagnostics",
            script,
        )

        # Neural models must explicitly use CUDA.
        self.assertIn(
            '"--nn-device" "cuda"',
            script,
        )

        # A missing/broken CUDA backend must terminate
        # instead of silently running a weaker model.
        self.assertIn(
            "No valid CUDA backend",
            script,
        )

        self.assertNotIn(
            '"--disable-neural"',
            script,
        )

    def test_production_mode_declares_xgb_bag3_v11b(
        self,
    ) -> None:
        script = self._script()

        self.assertIn(
            "#SBATCH -J ensemble-v11b",
            script,
        )

        self.assertIn(
            (
                "[MODE] "
                "xgb-bag3+lgb+cat+resnet+"
                "ft_transformer "
                "fixedchamp-calibrated-v11b"
            ),
            script,
        )

        # Regression guards against accidentally
        # launching an obsolete strategy.
        self.assertNotIn(
            "calibrated-shrink-v10",
            script,
        )

        self.assertNotIn(
            "calibrated-shrink-v11",
            script,
        )

        self.assertNotIn(
            "temporal_v9",
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
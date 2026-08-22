from __future__ import annotations

import unittest

from scripts.train_submit import (
    evaluate_quality_gate_policy,
)


class ArtifactPolicyTests(
    unittest.TestCase
):
    @staticmethod
    def _failed_gate():
        return {
            "passed": False,
            "checks": {
                "2023_guard": False,
                "2024_non_degradation": True,
                "calibration_accepted": True,
            },
        }

    def test_quality_failure_does_not_block_default_artifact_policy(
        self,
    ) -> None:
        failed = (
            evaluate_quality_gate_policy(
                self._failed_gate(),
                require_quality_gate=False,
            )
        )

        self.assertEqual(
            failed,
            [
                "2023_guard",
            ],
        )

    def test_explicit_strict_mode_can_still_fail(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "explicit strict mode",
        ):
            evaluate_quality_gate_policy(
                self._failed_gate(),
                require_quality_gate=True,
            )

    def test_passed_gate_returns_no_failures(
        self,
    ) -> None:
        gate = {
            "passed": True,
            "checks": {
                "2023_guard": True,
                "2024_non_degradation": True,
            },
        }

        failed = (
            evaluate_quality_gate_policy(
                gate,
                require_quality_gate=False,
            )
        )

        self.assertEqual(
            failed,
            [],
        )

    def test_inconsistent_gate_report_is_rejected(
        self,
    ) -> None:
        gate = {
            "passed": True,
            "checks": {
                "2023_guard": False,
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "internally inconsistent",
        ):
            evaluate_quality_gate_policy(
                gate,
                require_quality_gate=False,
            )


if __name__ == "__main__":
    unittest.main()
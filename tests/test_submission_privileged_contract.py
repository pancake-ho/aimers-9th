from __future__ import annotations

import unittest
from pathlib import Path


class SubmissionPrivilegedTests(
    unittest.TestCase
):
    def test_submission_never_reads_trackman(
        self,
    ) -> None:
        source = Path(
            "submission/script.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "trackman_history.csv",
            source,
        )

        self.assertNotIn(
            "priv_pitch_type_group",
            source,
        )

        self.assertNotIn(
            "priv_rel_speed",
            source,
        )

    def test_privileged_training_module_is_not_imported(
        self,
    ) -> None:
        source = Path(
            "submission/script.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "privileged_trackman",
            source,
        )


if __name__ == "__main__":
    unittest.main()
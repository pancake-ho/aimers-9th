from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TabDPTSubmissionContractTests(unittest.TestCase):
    def test_submission_uses_bundled_context_and_checkpoint(self):
        source = (ROOT / "submission_tabdpt" / "script.py").read_text(encoding="utf-8")
        self.assertIn('MODEL_DIR / "tabdpt_context.npz"', source)
        self.assertIn('MODEL_DIR / "tabdpt1_2.safetensors"', source)
        self.assertNotIn('"train.csv"', source)
        self.assertIn("context_size=None", source)
        self.assertIn('context_reduction="subsample"', source)
        self.assertIn('compile=bool(config["compile_model"])', source)

    def test_requirements_are_pinned_without_reinstalling_torch(self):
        requirements = (
            ROOT / "submission_tabdpt" / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(requirements, ["xgboost==3.2.0", "tabdpt==1.2.0"])

    def test_builder_has_exact_expected_archive_members(self):
        source = (ROOT / "scripts" / "build_tabdpt_submit.py").read_text(
            encoding="utf-8"
        )
        for name in (
            '"script.py"',
            '"requirements.txt"',
            '"THIRD_PARTY_LICENSES/TabDPT-LICENSE.txt"',
            '"THIRD_PARTY_LICENSES/NOTICE.md"',
            '"model/runtime.py"',
            '"model/xgb_model.json"',
            '"model/tabdpt_context.npz"',
            '"model/tabdpt1_2.safetensors"',
            '"model/bundle.pkl"',
            '"model/manifest.json"',
        ):
            self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()

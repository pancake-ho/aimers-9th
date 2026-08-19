from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_submit


class OptionalNeuralSubmissionTests(unittest.TestCase):
    def _required_files_for(self, model_order: list[str]) -> dict[str, Path]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "manifest.json").write_text(
                json.dumps({"model_order": model_order}),
                encoding="utf-8",
            )
            with patch.object(build_submit, "MODEL_DIR", model_dir):
                return build_submit._required_files()

    def test_gbdt_fallback_archive_does_not_require_neural_files(self) -> None:
        files = self._required_files_for(["xgb", "cat"])

        self.assertNotIn("model/neural_runtime.py", files)
        self.assertNotIn("model/resnet.pt", files)

    def test_full_archive_requires_both_neural_checkpoints(self) -> None:
        files = self._required_files_for(
            ["xgb", "cat", "resnet", "ft_transformer"]
        )

        self.assertIn("model/neural_runtime.py", files)
        self.assertIn("model/resnet.pt", files)
        self.assertIn("model/ft_transformer.pt", files)

    def test_unknown_model_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported manifest model_order"):
            self._required_files_for(["cat", "ft_transformer"])


if __name__ == "__main__":
    unittest.main()

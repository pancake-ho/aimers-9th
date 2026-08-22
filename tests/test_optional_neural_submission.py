from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_submit


class OptionalNeuralSubmissionTests(
    unittest.TestCase
):
    def _required_files_for(
        self,
        model_order: list[str],
        xgb_model_files: list[str]
        | None = None,
    ) -> dict[str, Path]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            model_dir = (
                root / "model"
            )
            model_dir.mkdir()

            manifest = {
                "model_order": (
                    model_order
                )
            }

            if (
                xgb_model_files
                is not None
            ):
                manifest[
                    "xgb_bagging"
                ] = {
                    "seeds": [
                        2026,
                        2027,
                        2028,
                    ],
                    "model_files": (
                        xgb_model_files
                    ),
                }

            (
                model_dir
                / "manifest.json"
            ).write_text(
                json.dumps(
                    manifest
                ),
                encoding="utf-8",
            )

            with patch.object(
                build_submit,
                "MODEL_DIR",
                model_dir,
            ):
                return (
                    build_submit
                    ._required_files()
                )

    def test_xgb_bagging_artifacts_are_packaged(
        self,
    ) -> None:
        files = (
            self._required_files_for(
                [
                    "xgb",
                    "lgb",
                    "cat",
                    "resnet",
                    "ft_transformer",
                ],
                xgb_model_files=[
                    "xgb_model.json",
                    "xgb_model_seed2027.json",
                    "xgb_model_seed2028.json",
                ],
            )
        )

        self.assertIn(
            "model/xgb_model.json",
            files,
        )

        self.assertIn(
            "model/xgb_model_seed2027.json",
            files,
        )

        self.assertIn(
            "model/xgb_model_seed2028.json",
            files,
        )

    def test_gbdt_fallback_archive_does_not_require_neural_files(
        self,
    ) -> None:
        files = self._required_files_for(
            [
                "xgb",
                "lgb",
                "cat",
            ]
        )

        self.assertNotIn(
            "model/neural_runtime.py",
            files,
        )
        self.assertNotIn(
            "model/resnet.pt",
            files,
        )
        self.assertNotIn(
            "model/ft_transformer.pt",
            files,
        )

        # Regression guard:
        # CatBoost is a GBDT artifact,
        # never a torch checkpoint.
        self.assertNotIn(
            "model/cat.pt",
            files,
        )

    def test_resnet_archive_requires_only_resnet_checkpoint(
        self,
    ) -> None:
        files = self._required_files_for(
            [
                "xgb",
                "lgb",
                "cat",
                "resnet",
            ]
        )

        self.assertIn(
            "model/neural_runtime.py",
            files,
        )
        self.assertIn(
            "model/resnet.pt",
            files,
        )
        self.assertNotIn(
            "model/ft_transformer.pt",
            files,
        )
        self.assertNotIn(
            "model/cat.pt",
            files,
        )

    def test_full_archive_requires_both_neural_checkpoints(
        self,
    ) -> None:
        files = self._required_files_for(
            [
                "xgb",
                "lgb",
                "cat",
                "resnet",
                "ft_transformer",
            ]
        )

        self.assertIn(
            "model/neural_runtime.py",
            files,
        )
        self.assertIn(
            "model/resnet.pt",
            files,
        )
        self.assertIn(
            "model/ft_transformer.pt",
            files,
        )
        self.assertNotIn(
            "model/cat.pt",
            files,
        )

    def test_unknown_model_order_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported manifest model_order",
        ):
            self._required_files_for(
                [
                    "cat",
                    "ft_transformer",
                ]
            )


if __name__ == "__main__":
    unittest.main()
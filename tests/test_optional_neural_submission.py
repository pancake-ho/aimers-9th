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
        *,
        xgb_multiview: bool = False,
        xgb_temporal: bool = False,
    ) -> dict[str, Path]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(
                temp_dir
            )

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
                        2026 + index
                        for index in range(
                            len(
                                xgb_model_files
                            )
                        )
                    ],

                    "model_files": (
                        xgb_model_files
                    ),
                }

            if xgb_multiview:
                manifest[
                    "xgb_multiview"
                ] = {
                    "view_order": [
                        "base",
                        "representation",
                        "representative",
                    ],

                    "weights": [
                        0.70,
                        0.20,
                        0.10,
                    ],

                    "representation_model_file": (
                        "xgb_representation.json"
                    ),

                    "representative_model_file": (
                        "xgb_representative.json"
                    ),

                    "representation_feature_count": 100,

                    "representation_raw_feature_count": 92,

                    "pca_component_count": 8,
                }
            
            if xgb_temporal:
                manifest[
                    "xgb_temporal_views"
                ] = {
                    "view_order": [
                        "base",
                        "recent1",
                        "recent2",
                    ],

                    "weights": [
                        0.5,
                        0.25,
                        0.25,
                    ],

                    "model_files": {
                        "recent1": (
                            "xgb_recent1.json"
                        ),

                        "recent2": (
                            "xgb_recent2.json"
                        ),
                    },
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

    def test_xgb_multiview_artifacts_are_packaged(
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
                xgb_multiview=True,
            )
        )

        self.assertIn(
            "model/xgb_representation.json",
            files,
        )

        self.assertIn(
            "model/xgb_representative.json",
            files,
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

    def test_xgb_temporal_artifacts_are_packaged(
        self,
    ) -> None:
        files = self._required_files_for(
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
            xgb_temporal=True,
        )

        self.assertIn(
            "model/xgb_recent1.json",
            files,
        )

        self.assertIn(
            "model/xgb_recent2.json",
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
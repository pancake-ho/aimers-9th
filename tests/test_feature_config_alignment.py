from __future__ import annotations

import ast
import unittest
from dataclasses import fields
from pathlib import Path

from src.config import FeatureConfig
from src.features import LeakageSafeFeatureEngineer, validate_feature_config_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FeatureConfigAlignmentTests(unittest.TestCase):
    def test_all_feature_config_attributes_are_declared(self) -> None:
        source = (PROJECT_ROOT / "src" / "features.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        declared = {field.name for field in fields(FeatureConfig)}
        referenced = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and node.value.attr == "config"
        }

        self.assertTrue(referenced)
        self.assertEqual(referenced - declared, set())

    def test_runtime_state_matches_current_non_tabm_contract(self) -> None:
        state = LeakageSafeFeatureEngineer(FeatureConfig()).export_runtime_state()
        self.assertEqual(
            set(state),
            {
                "feature_version",
                "smoothing_prior",
                "pitcher_prior_strength",
                "batter_prior_strength",
                "cold_start_threshold",
                "trackman",
                "main_history",
            },
        )
        validate_feature_config_contract(FeatureConfig())

    def test_rejected_residual_contract_is_absent(self) -> None:
        source = (PROJECT_ROOT / "src" / "features.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        rejected_references = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr
            in {"residual_prior_strength", "residual_probability_clip"}
        }
        self.assertEqual(rejected_references, set())


if __name__ == "__main__":
    unittest.main()

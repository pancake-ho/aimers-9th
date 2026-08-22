from __future__ import annotations

import unittest

from dataclasses import replace

from src.config import (
    ModelConfig,
)

from src.models import (
    _validated_xgb_bagging_seeds,
    _xgb_params,
)


class XGBoostBaggingContractTests(
    unittest.TestCase
):
    def test_default_seed_bag(
        self,
    ) -> None:
        config = ModelConfig()

        seeds = (
            _validated_xgb_bagging_seeds(
                config
            )
        )

        self.assertEqual(
            seeds,
            (
                2026,
                2027,
                2028,
            ),
        )

        self.assertEqual(
            seeds[0],
            config.random_seed,
        )

    def test_explicit_seed_reaches_xgb_params(
        self,
    ) -> None:
        config = ModelConfig()

        params = _xgb_params(
            config,
            seed=2028,
        )

        self.assertEqual(
            params["seed"],
            2028,
        )

        self.assertLess(
            float(
                params["subsample"]
            ),
            1.0,
        )

        self.assertLess(
            float(
                params[
                    "colsample_bytree"
                ]
            ),
            1.0,
        )

    def test_duplicate_seeds_are_rejected(
        self,
    ) -> None:
        config = replace(
            ModelConfig(),
            xgb_bagging_seeds=(
                2026,
                2026,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            _validated_xgb_bagging_seeds(
                config
            )

    def test_baseline_seed_must_be_first(
        self,
    ) -> None:
        config = replace(
            ModelConfig(),
            xgb_bagging_seeds=(
                2027,
                2028,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            _validated_xgb_bagging_seeds(
                config
            )


if __name__ == "__main__":
    unittest.main()
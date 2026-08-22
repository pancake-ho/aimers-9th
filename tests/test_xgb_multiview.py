from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.xgb_multiview import (
    apply_numeric_pca_state,
    blend_xgb_views,
    fit_numeric_pca_state,
    make_representative_sample_weight,
    select_xgb_multiview_weights,
)


class XGBoostMultiViewTests(
    unittest.TestCase
):
    def test_pca_is_train_fitted_and_finite(
        self,
    ) -> None:
        rng = np.random.default_rng(
            2026
        )

        X = pd.DataFrame(
            rng.normal(
                size=(
                    500,
                    12,
                )
            ),
            columns=[
                f"f{index}"
                for index in range(12)
            ],
        )

        state = (
            fit_numeric_pca_state(
                X,
                X.columns,
                n_components=4,
            )
        )

        transformed = (
            apply_numeric_pca_state(
                X,
                state,
            )
        )

        self.assertEqual(
            transformed.shape,
            (
                500,
                4,
            ),
        )

        self.assertTrue(
            np.isfinite(
                transformed
                .to_numpy()
            ).all()
        )

    def test_representative_weight_preserves_total_scale(
        self,
    ) -> None:
        X = pd.DataFrame(
            {
                "pitcher_count_combo": (
                    [0] * 80
                    + [1] * 20
                ),
                "pitcher_base_combo": (
                    [0] * 50
                    + [1] * 50
                ),
                "li_log": np.linspace(
                    0.0,
                    2.0,
                    100,
                ),
            }
        )

        base = np.ones(
            100,
            dtype=np.float32,
        )

        weight, diagnostics = (
            make_representative_sample_weight(
                X,
                base,
                group_columns=(
                    "pitcher_count_combo",
                    "pitcher_base_combo",
                ),
                leverage_column=(
                    "li_log"
                ),
                leverage_bins=5,
                clip_low=0.5,
                clip_high=2.0,
            )
        )

        self.assertAlmostEqual(
            float(
                weight.sum()
            ),
            float(
                base.sum()
            ),
            places=5,
        )

        self.assertGreater(
            diagnostics[
                "factor_std"
            ],
            0.0,
        )

    def test_multiview_blend(
        self,
    ) -> None:
        views = {
            "base": np.asarray(
                [0.2, 0.8]
            ),
            "representation": (
                np.asarray(
                    [0.3, 0.7]
                )
            ),
            "representative": (
                np.asarray(
                    [0.4, 0.6]
                )
            ),
        }

        result = blend_xgb_views(
            views,
            [
                0.5,
                0.25,
                0.25,
            ],
        )

        np.testing.assert_allclose(
            result,
            [
                0.275,
                0.725,
            ],
            atol=1e-12,
        )

    def test_selector_can_fallback_to_base(
        self,
    ) -> None:
        target = np.asarray(
            [
                0.0,
                1.0,
                0.0,
                1.0,
            ]
        )

        folds = []

        for label in (
            "2023",
            "2024",
            "2024_late_abs",
        ):
            folds.append(
                {
                    "validation_label": (
                        label
                    ),
                    "y_true": target,
                    "xgb_views": {
                        "base": np.asarray(
                            [
                                0.2,
                                0.8,
                                0.2,
                                0.8,
                            ]
                        ),
                        "representation": np.asarray(
                            [
                                0.5,
                                0.5,
                                0.5,
                                0.5,
                            ]
                        ),
                        "representative": np.asarray(
                            [
                                0.6,
                                0.4,
                                0.6,
                                0.4,
                            ]
                        ),
                    },
                }
            )

        weights, report = (
            select_xgb_multiview_weights(
                folds,
                fold_importance=(
                    0.05,
                    0.45,
                    0.50,
                ),
                grid_step=0.05,
                minimum_base_weight=0.45,
                maximum_aux_weight=0.40,
                protected_labels=(
                    "2024",
                    "2024_late_abs",
                ),
                protected_tolerance=0.0,
            )
        )

        np.testing.assert_allclose(
            weights,
            [
                1.0,
                0.0,
                0.0,
            ],
            atol=1e-12,
        )

        self.assertEqual(
            report["weights"],
            [
                1.0,
                0.0,
                0.0,
            ],
        )


if __name__ == "__main__":
    unittest.main()
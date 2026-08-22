from __future__ import annotations

import unittest

import numpy as np

from submission.script import (
    _predict_xgb_bag,
)


class _FakeBooster:
    def __init__(
        self,
        prediction,
    ) -> None:
        self.prediction = np.asarray(
            prediction,
            dtype=np.float64,
        )

    def predict(
        self,
        dmatrix,
    ):
        del dmatrix

        return self.prediction.copy()


class XGBoostBaggingSubmissionTests(
    unittest.TestCase
):
    def test_three_models_are_averaged(
        self,
    ) -> None:
        models = [
            _FakeBooster(
                [0.2, 0.4, 0.6]
            ),
            _FakeBooster(
                [0.3, 0.5, 0.7]
            ),
            _FakeBooster(
                [0.4, 0.6, 0.8]
            ),
        ]

        prediction = (
            _predict_xgb_bag(
                models,
                object(),
            )
        )

        np.testing.assert_allclose(
            prediction,
            np.asarray(
                [
                    0.3,
                    0.5,
                    0.7,
                ],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_single_model_remains_valid(
        self,
    ) -> None:
        prediction = (
            _predict_xgb_bag(
                [
                    _FakeBooster(
                        [0.1, 0.9]
                    )
                ],
                object(),
            )
        )

        np.testing.assert_allclose(
            prediction,
            [0.1, 0.9],
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_empty_bag_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "at least one model",
        ):
            _predict_xgb_bag(
                [],
                object(),
            )

    def test_shape_mismatch_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "shape mismatch",
        ):
            _predict_xgb_bag(
                [
                    _FakeBooster(
                        [0.2, 0.4]
                    ),
                    _FakeBooster(
                        [0.2, 0.4, 0.6]
                    ),
                ],
                object(),
            )

    def test_non_finite_prediction_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "NaN or infinity",
        ):
            _predict_xgb_bag(
                [
                    _FakeBooster(
                        [
                            0.2,
                            np.nan,
                        ]
                    )
                ],
                object(),
            )


if __name__ == "__main__":
    unittest.main()
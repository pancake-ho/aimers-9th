from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from src.config import NeuralConfig
    from src.neural import (
        NeuralPreprocessor,
        build_model,
        load_checkpoint,
        make_model_spec,
        predict_model,
        prepare_neural_arrays,
        save_checkpoint,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class NeuralContractTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "cat_a": np.asarray([0, 1, -1, 0, 1, 0, 1, -1], dtype=np.int32),
                "cat_b": np.asarray([0, 0, 1, 1, 2, 2, -1, 0], dtype=np.int32),
                "num_a": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
                "num_b": np.asarray([1, 1, 2, 3, 5, 8, 13, 21], dtype=np.float32),
            }
        )
        processor = NeuralPreprocessor(
            categorical_cols=("cat_a", "cat_b"),
            numerical_cols=("num_a", "num_b"),
        ).fit(self.X)
        self.state = processor.export_state()
        self.arrays = prepare_neural_arrays(self.X, self.state)

    def test_preprocessor_reserves_unknown_embedding_index(self):
        x_num, x_cat = self.arrays
        self.assertTrue(np.isfinite(x_num).all())
        self.assertTrue((x_cat >= 0).all())
        self.assertEqual(int(x_cat[2, 0]), 0)

    def test_both_models_round_trip_checkpoint(self):
        config = NeuralConfig(
            resnet_d_main=16,
            resnet_d_hidden=24,
            resnet_n_blocks=2,
            ft_d_token=8,
            ft_n_heads=2,
            ft_n_layers=2,
            ft_d_ffn=16,
        )
        for kind in ("resnet", "ft_transformer"):
            spec = make_model_spec(kind, self.state, config)
            model = build_model(spec)
            model.eval()
            expected = predict_model(model, self.arrays, device="cpu", batch_size=4)
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / f"{kind}.pt"
                save_checkpoint(path, model, spec)
                restored = load_checkpoint(path, device="cpu")
                actual = predict_model(restored, self.arrays, device="cpu", batch_size=4)
            self.assertEqual(actual.shape, (len(self.X),))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

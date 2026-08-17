from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
HAS_CATBOOST = importlib.util.find_spec("catboost") is not None

if HAS_XGBOOST and HAS_CATBOOST:
    from src.config import ModelConfig
    from src.models import _xgb_params, _quantile_dmatrix, validate_xgboost_backend


@unittest.skipUnless(
    HAS_XGBOOST and HAS_CATBOOST,
    "training backends are not installed in this test environment",
)
class XGBoostBackendContractTests(unittest.TestCase):
    def test_quantile_dmatrix_and_booster_share_max_bin(self):
        import numpy as np
        import pandas as pd

        config = ModelConfig(xgb_max_bin=257, xgb_device="cpu", num_threads=1)
        X = pd.DataFrame(
            {
                "a": np.asarray([0, 1, 2, 3], dtype=np.float32),
                "b": np.asarray([1, 0, 1, 0], dtype=np.float32),
            }
        )
        y = np.asarray([0, 0, 1, 1], dtype=np.float32)
        dtrain = _quantile_dmatrix(X, config, label=y)

        self.assertEqual(_xgb_params(config)["max_bin"], 257)
        # This exact update raised the reported 255-vs-256 exception before
        # max_bin was forwarded to QuantileDMatrix.
        validate_xgboost_backend(config)
        self.assertEqual(dtrain.num_row(), 4)


if __name__ == "__main__":
    unittest.main()

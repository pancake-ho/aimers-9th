from __future__ import annotations

import unittest

import pandas as pd

from src.splits import make_abs_late_fold


class AbsLateSplitTests(unittest.TestCase):
    def test_late_2024_validation_is_strictly_disjoint(self):
        rows = pd.DataFrame(
            {
                "season": [2023, 2024, 2024, 2024, 2024],
                "game_month": [9, 4, 6, 7, 9],
            }
        )
        fold = make_abs_late_fold(rows, train_month_max=6, valid_months=(7, 8, 9))
        self.assertEqual(fold.validation_label, "2024_late_abs")
        self.assertEqual(fold.train_idx.tolist(), [0, 1, 2])
        self.assertEqual(fold.valid_idx.tolist(), [3, 4])
        self.assertTrue(set(fold.train_idx).isdisjoint(set(fold.valid_idx)))


if __name__ == "__main__":
    unittest.main()

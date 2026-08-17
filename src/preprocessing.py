from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


class TabularPreprocessor:
    def __init__(
        self,
        categorical_cols: Sequence[str],
        excluded_cols: Sequence[str],
    ) -> None:
        self.requested_cat_cols = list(categorical_cols)
        self.excluded_cols = set(excluded_cols)
        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.numeric_medians_: Dict[str, float] = {}
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self.cat_cols_ = [
            c for c in self.requested_cat_cols
            if c in df.columns and c not in self.excluded_cols
        ]
        numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        self.num_cols_ = [
            c for c in numeric_candidates
            if c not in self.excluded_cols and c not in self.cat_cols_
        ]

        self.category_maps_.clear()
        for col in self.cat_cols_:
            values = df[col].fillna("__MISSING__").astype(str)
            self.category_maps_[col] = {
                value: idx for idx, value in enumerate(pd.unique(values))
            }

        self.numeric_medians_.clear()
        for col in self.num_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            median = values.median()
            self.numeric_medians_[col] = float(median) if np.isfinite(median) else 0.0

        self.feature_names_ = self.cat_cols_ + self.num_cols_
        print(
            f"[PREPROCESS] categorical={len(self.cat_cols_)}, "
            f"numerical={len(self.num_cols_)}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        result: Dict[str, pd.Series] = {}
        for col in self.cat_cols_:
            values = df[col].fillna("__MISSING__").astype(str)
            result[col] = (
                values.map(self.category_maps_[col]).fillna(-1).astype(np.int32)
            )

        for col in self.num_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            result[col] = values.fillna(self.numeric_medians_[col]).astype(np.float32)

        return pd.DataFrame(result, index=df.index)[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def export_state(self) -> Dict[str, object]:
        return {
            "cat_cols": list(self.cat_cols_),
            "num_cols": list(self.num_cols_),
            "feature_names": list(self.feature_names_),
            "category_maps": self.category_maps_,
            "numeric_medians": self.numeric_medians_,
        }

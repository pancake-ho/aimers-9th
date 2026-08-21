from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.runtime import preprocess_frame


class TabularPreprocessor:
    """Train-fitted ordinal coding and median imputation.

    Category vocabularies and numeric medians are learned only from the
    training side of each temporal fold. Unknown future categories map to -1.
    """

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

    @property
    def categorical_indices(self) -> list[int]:
        return [self.feature_names_.index(col) for col in self.cat_cols_]

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        if not df.columns.is_unique:
            duplicated = (
                df.columns[
                    df.columns.duplicated(
                        keep=False
                    )
                ]
                .tolist()
            )

            counts = {
                name: duplicated.count(name)
                for name in sorted(
                    set(duplicated)
                )
            }

            raise ValueError(
                "Input feature table contains "
                "duplicate column names: "
                f"{counts}"
            )

        self.cat_cols_ = [
            col
            for col in self.requested_cat_cols
            if (
                col in df.columns
                and col
                not in self.excluded_cols
            )
        ]
        numeric_candidates = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.num_cols_ = [
            col
            for col in numeric_candidates
            if col not in self.excluded_cols and col not in self.cat_cols_
        ]

        self.category_maps_.clear()
        for col in self.cat_cols_:
            values = df[col].fillna("__MISSING__").astype(str)
            self.category_maps_[col] = {
                value: index for index, value in enumerate(pd.unique(values))
            }

        self.numeric_medians_.clear()
        for col in self.num_cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            median = values.median()
            self.numeric_medians_[col] = float(median) if np.isfinite(median) else 0.0

        self.feature_names_ = self.cat_cols_ + self.num_cols_
        if len(self.feature_names_) != len(set(self.feature_names_)):
            raise RuntimeError("Duplicate preprocessed feature name.")
        print(
            f"[PREPROCESS] categorical={len(self.cat_cols_)}, "
            f"numerical={len(self.num_cols_)}, total={len(self.feature_names_)}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        return preprocess_frame(df, self.export_state())

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def export_state(self) -> Dict[str, object]:
        return {
            "state_version": 1,
            "cat_cols": list(self.cat_cols_),
            "num_cols": list(self.num_cols_),
            "feature_names": list(self.feature_names_),
            "category_maps": self.category_maps_,
            "numeric_medians": self.numeric_medians_,
        }

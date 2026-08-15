# src/preprocessing.py

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


class TabularPreprocessor:

    def __init__(
        self,
        categorical_cols: Sequence[str],
        excluded_cols: Sequence[str],
    ):
        self.requested_cat_cols = list(
            categorical_cols
        )

        self.excluded_cols = set(
            excluded_cols
        )

        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []

        self.category_maps_: Dict[
            str,
            Dict[str, int],
        ] = {}

        self.numeric_medians_: Dict[
            str,
            float,
        ] = {}

        self.feature_names_: List[
            str
        ] = []

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "TabularPreprocessor":

        self.cat_cols_ = [
            c
            for c in self.requested_cat_cols
            if c in df.columns
            and c not in self.excluded_cols
        ]

        numeric_candidates = (
            df
            .select_dtypes(
                include=[np.number]
            )
            .columns
            .tolist()
        )

        self.num_cols_ = [
            c
            for c in numeric_candidates
            if c not in self.excluded_cols
            and c not in self.cat_cols_
        ]

        self.category_maps_.clear()

        for col in self.cat_cols_:

            values = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

            unique_values = (
                pd.unique(values)
            )

            self.category_maps_[
                col
            ] = {
                value: idx
                for idx, value
                in enumerate(
                    unique_values
                )
            }

        self.numeric_medians_.clear()

        for col in self.num_cols_:

            values = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            median = values.median()

            if not np.isfinite(median):
                median = 0.0

            self.numeric_medians_[
                col
            ] = float(median)

        self.feature_names_ = (
            self.cat_cols_
            + self.num_cols_
        )

        print(
            "[PREPROCESS] "
            f"categorical={len(self.cat_cols_)}, "
            f"numerical={len(self.num_cols_)}"
        )

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.feature_names_:
            raise RuntimeError(
                "Preprocessor must be fitted."
            )

        result = {}

        for col in self.cat_cols_:

            mapping = (
                self.category_maps_[
                    col
                ]
            )

            values = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

            # unseen category -> -1
            encoded = (
                values
                .map(mapping)
                .fillna(-1)
                .astype(np.int32)
            )

            result[col] = encoded

        for col in self.num_cols_:

            values = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            values = values.fillna(
                self.numeric_medians_[
                    col
                ]
            )

            result[col] = (
                values.astype(
                    np.float32
                )
            )

        return pd.DataFrame(
            result,
            index=df.index,
        )

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        self.fit(df)
        return self.transform(df)

    @property
    def categorical_indices(
        self,
    ) -> List[int]:

        return list(
            range(
                len(
                    self.cat_cols_
                )
            )
        )
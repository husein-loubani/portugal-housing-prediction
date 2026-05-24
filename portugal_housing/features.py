"""
features.py
-----------
Preprocessing pipeline construction for the Portugal Housing dataset.

All transformations are encapsulated in a leakage-safe ColumnTransformer.
The transformer is always fitted only on training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from portugal_housing.config import (
    BINARY_FEATURES,
    NUMERICAL_FEATURES,
)

# ── Feature groups for the base preprocessor ─────────────────────────────────

NUM_SCALE: list[str] = NUMERICAL_FEATURES
BIN_PASS: list[str] = BINARY_FEATURES
CAT_OHE: list[str] = ["type", "energy_certificate", "district"]
CAT_DROP: list[str] = ["city", "town"]


# ── Preprocessor factory ──────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that handles all feature groups.

    Numerical features are median-imputed before scaling so downstream models
    that don't tolerate NaN (PCA, linear models, sklearn trees) work end-to-end.
    Categorical features are imputed with a constant 'missing' token before
    one-hot encoding.

    Returns a fresh (unfitted) transformer. Fit it on training data only.
    """
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    bin_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",    OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
            min_frequency=50,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_SCALE),
            ("bin", bin_pipe, BIN_PASS),
            ("cat", cat_pipe, CAT_OHE),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(clf) -> Pipeline:
    """
    Wrap a fresh preprocessor and a regressor into a single sklearn Pipeline.

    A new preprocessor is built per call so pipelines are fully independent
    and can be fitted in separate CV folds without sharing state.
    """
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", clf),
    ])


# ── Feature engineering ──────────────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-level feature engineering. No fitting required, so leakage-safe.

    New features:
    - property_age: 2026 - construction_year (0 if missing or future year)
    - log_total_area: log1p of total_area
    - log_living_area: log1p of living_area
    """
    d = df.copy()

    # Property age
    current_year = 2026
    d["property_age"] = current_year - d["construction_year"]
    d["property_age"] = d["property_age"].clip(lower=0)
    d.loc[d["construction_year"].isna(), "property_age"] = np.nan

    # Log-transforms on area features (handle zeros with log1p)
    if "total_area" in d.columns:
        d["log_total_area"] = np.log1p(d["total_area"].clip(lower=0))
    if "living_area" in d.columns:
        d["log_living_area"] = np.log1p(d["living_area"].clip(lower=0))

    return d


# ── Engineered-feature groups ────────────────────────────────────────────────

ENG_NUM_SCALE: list[str] = [
    "total_area", "parking", "construction_year", "total_rooms",
    "living_area", "number_of_bathrooms",
    "property_age", "log_total_area", "log_living_area",
]
ENG_BIN_PASS: list[str] = BINARY_FEATURES
ENG_CAT_OHE: list[str] = ["type", "energy_certificate", "district"]
ENG_FEATURES: list[str] = ENG_NUM_SCALE + ENG_BIN_PASS + ENG_CAT_OHE


def build_eng_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for the expanded engineered feature set (with imputation)."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    bin_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",    OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore",
            min_frequency=50,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, ENG_NUM_SCALE),
            ("bin", bin_pipe, ENG_BIN_PASS),
            ("cat", cat_pipe, ENG_CAT_OHE),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_eng_pipeline(clf) -> Pipeline:
    """Pipeline using the engineered feature preprocessor."""
    return Pipeline([("pre", build_eng_preprocessor()), ("clf", clf)])

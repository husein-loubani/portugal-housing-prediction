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
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from portugal_housing.config import (
    NUMERICAL_FEATURES,
)

# ── Feature groups for the base preprocessor ─────────────────────────────────

NUM_SCALE: list[str] = NUMERICAL_FEATURES
# Categoricals one-hot encoded. `elevator` is a binary categorical and is
# encoded here with the rest (drop="first" turns it into a single 0/1 column).
CAT_OHE: list[str] = ["type", "energy_certificate", "district", "elevator"]
CAT_DROP: list[str] = ["city", "town"]


def _build_cat_pipe() -> Pipeline:
    """Impute -> cast to string -> one-hot. Shared by both preprocessors.

    The boolean `elevator` column otherwise mixes Python bools with the imputed
    'missing' token, which OneHotEncoder rejects. The cast uses np.vectorize(str)
    rather than a project-local function so the serialized model stays
    self-contained, it loads at inference without needing this package installed.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("to_str", FunctionTransformer(np.vectorize(str), feature_names_out="one-to-one")),
        ("ohe",    OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
            min_frequency=50,
        )),
    ])


# ── Preprocessor factory ──────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that handles all feature groups.

    Numerical features are median-imputed before scaling so downstream models
    that don't tolerate NaN (PCA, linear models, sklearn trees) work end-to-end.
    Categorical features (including the boolean `elevator`) are imputed with a
    constant 'missing' token before one-hot encoding.

    Returns a fresh (unfitted) transformer. Fit it on training data only.
    """
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_SCALE),
            ("cat", _build_cat_pipe(), CAT_OHE),
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
ENG_CAT_OHE: list[str] = ["type", "energy_certificate", "district", "elevator"]
ENG_FEATURES: list[str] = ENG_NUM_SCALE + ENG_CAT_OHE


def build_eng_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for the expanded engineered feature set (with imputation)."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, ENG_NUM_SCALE),
            ("cat", _build_cat_pipe(), ENG_CAT_OHE),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_eng_pipeline(clf) -> Pipeline:
    """Pipeline using the engineered feature preprocessor."""
    return Pipeline([("pre", build_eng_preprocessor()), ("clf", clf)])

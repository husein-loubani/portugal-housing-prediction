"""
train.py
--------
Model training and cross-validation utilities.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from portugal_housing.config import RANDOM_SEED


def cv_compare(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> pd.DataFrame:
    """
    Run k-fold CV on a dict of named pipelines.

    Returns a DataFrame with mean / std of the chosen scoring metric,
    mean train score, and mean fit time, sorted by mean score.

    For regression metrics (neg_*), higher (less negative) is better,
    so we sort descending.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        rows.append({
            "model":                    name,
            f"cv_{scoring}_mean":       round(scores["test_score"].mean(), 4),
            f"cv_{scoring}_std":        round(scores["test_score"].std(), 4),
            "train_score_mean":         round(scores["train_score"].mean(), 4),
            "fit_time_s":               round(scores["fit_time"].mean(), 2),
        })

    sort_ascending = False  # neg_* metrics: higher (less negative) is better
    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values(f"cv_{scoring}_mean", ascending=sort_ascending)
    )

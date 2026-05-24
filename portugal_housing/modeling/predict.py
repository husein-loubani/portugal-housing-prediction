"""
predict.py
----------
Model inference and evaluation utilities for regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Comprehensive evaluation on a held-out test set.

    Returns a metrics dict and prints a formatted report.
    The test set should only be passed here once, after all tuning is complete.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Residual stats
    residuals = y_test - y_pred

    metrics = {
        "rmse":         round(rmse, 2),
        "mae":          round(mae, 2),
        "r2":           round(r2, 4),
        "mape":         round(mape, 4),
        "residual_mean": round(residuals.mean(), 2),
        "residual_std":  round(residuals.std(), 2),
        "y_pred":       y_pred,
    }

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {model_name}: Test Set Evaluation")
    print(sep)
    print(f"  RMSE            : €{rmse:,.2f}")
    print(f"  MAE             : €{mae:,.2f}")
    print(f"  R²              : {r2:.4f}")
    print(f"  MAPE            : {mape:.2%}")
    print(f"  Residual mean   : €{residuals.mean():,.2f}")
    print(f"  Residual std    : €{residuals.std():,.2f}")
    print(sep)
    return metrics


def compare_final_metrics(metrics_dict: dict) -> pd.DataFrame:
    """
    Combine per-model evaluation dicts into a sortable comparison DataFrame.
    Excludes the y_pred array.
    """
    rows = []
    for name, m in metrics_dict.items():
        row = {"model": name}
        row.update({k: v for k, v in m.items() if k != "y_pred"})
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("rmse", ascending=True)
    )

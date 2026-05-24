"""
dataset.py
----------
Functions for loading, auditing, cleaning, and splitting the Portugal Housing dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from portugal_housing.config import (
    ALPHA,
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    RENAME_COLS,
    TARGET,
    TEST_SIZE,
)

# ── Loading ───────────────────────────────────────────────────────────────────

def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV with low_memory=False to handle mixed-type columns."""
    df = pd.read_csv(path, low_memory=False)
    return df


# ── Auditing ──────────────────────────────────────────────────────────────────

def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a structured data-quality report and return a per-column summary DataFrame.

    Covers: shape, dtypes, missing values, duplicates, unique counts, and
    numeric range statistics.
    """
    n_rows, n_cols = df.shape
    print(f"Shape          : {n_rows:,} rows × {n_cols} columns")
    print(f"Duplicated rows: {df.duplicated().sum():,}")
    print()

    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    miss_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    miss_df = miss_df[miss_df["missing_count"] > 0].sort_values("missing_%", ascending=False)
    if miss_df.empty:
        print("Missing values : None detected.")
    else:
        print("Missing values detected:")
        print(miss_df.to_string())
    print()

    rows = []
    for col in df.columns:
        row = {
            "column":   col,
            "dtype":    str(df[col].dtype),
            "n_unique": df[col].nunique(),
            "missing":  df[col].isnull().sum(),
            "missing_%": round(df[col].isnull().sum() / n_rows * 100, 1),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            row["min"]    = df[col].min()
            row["max"]    = df[col].max()
            row["mean"]   = round(df[col].mean(), 2)
            row["median"] = df[col].median()
        else:
            row["min"] = row["max"] = row["mean"] = row["median"] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).set_index("column")


def anomaly_screen(
    df: pd.DataFrame,
    features: list[str],
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Light IQR-based anomaly screen for numeric features (initial audit stage).

    Computes Tukey fences (Q1 - factor*IQR, Q3 + factor*IQR) and counts values
    outside those bounds.
    """
    indices, rows = [], []
    for col in features:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            rows.append({
                "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": 0.0,
                "lower_fence": np.nan, "upper_fence": np.nan,
                "n_flagged": np.nan, "pct_flagged": np.nan,
                "note": "IQR=0; screen not applicable",
            })
        else:
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            n_flagged = int(((s < lower) | (s > upper)).sum())
            note = "potential anomalies" if n_flagged > 0 else "none flagged"
            rows.append({
                "Q1": round(q1, 2), "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "lower_fence": round(lower, 2), "upper_fence": round(upper, 2),
                "n_flagged": n_flagged,
                "pct_flagged": round(n_flagged / len(s) * 100, 1),
                "note": note,
            })
        indices.append(col)
    return pd.DataFrame(rows, index=pd.Index(indices, name="feature"))


def descriptive_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a transposed describe() for the given features, rounded to 2 dp."""
    return df[features].describe().T.round(2)


def target_stats(df: pd.DataFrame, target: str = None) -> pd.DataFrame:
    """Return descriptive statistics for the continuous target column."""
    col = target if target is not None else TARGET
    s = df[col].dropna()
    return pd.DataFrame({
        "count":  [int(s.count())],
        "mean":   [round(s.mean(), 2)],
        "median": [round(s.median(), 2)],
        "std":    [round(s.std(), 2)],
        "min":    [round(s.min(), 2)],
        "max":    [round(s.max(), 2)],
        "skew":   [round(s.skew(), 2)],
        "kurtosis": [round(s.kurtosis(), 2)],
    }, index=[col])


def inspect_categoricals(df: pd.DataFrame, features: list = None) -> None:
    """Print distinct value counts for every categorical feature."""
    cols = features if features is not None else CATEGORICAL_FEATURES
    for col in cols:
        if col in df.columns:
            vc = df[col].value_counts()
            top5 = vc.head(5).to_dict()
            others = len(vc) - 5 if len(vc) > 5 else 0
            print(f"  {col:<25} -> {top5}" + (f"  (+{others} more)" if others else ""))


# ── Data Cleaning ─────────────────────────────────────────────────────────────

def duplicate_summary(df: pd.DataFrame) -> None:
    """Print a concise duplicate-row summary for the DataFrame."""
    n_dups = df.duplicated().sum()
    print(f"Original  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Unique    : {df.drop_duplicates().shape[0]:,} rows")
    print(f"Duplicates: {n_dups:,} rows ({n_dups / len(df) * 100:.1f}%)")


def duplicate_impact_numerical(
    df: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Assess the impact of duplicate rows on numerical feature distributions.

    Compares unique records against the extra duplicate rows using a
    Mann-Whitney U test (non-parametric, no normality assumption).
    """
    if features is None:
        features = NUMERICAL_FEATURES
    features = [f for f in features if f in df.columns]
    uniq_recs = df.drop_duplicates()
    dup_extra = df[df.duplicated(keep="first")]

    if len(dup_extra) == 0:
        print("No duplicate rows found.")
        return pd.DataFrame()

    rows = []
    for col in features:
        if df[col].dropna().empty or dup_extra[col].dropna().empty:
            continue
        mean_all   = round(df[col].mean(), 2)
        mean_dedup = round(uniq_recs[col].mean(), 2)
        med_all    = round(df[col].median(), 2)
        med_dedup  = round(uniq_recs[col].median(), 2)
        _, p_mw    = stats.mannwhitneyu(
            uniq_recs[col].dropna(), dup_extra[col].dropna(), alternative="two-sided"
        )
        rows.append({
            "feature":           col,
            "mean (all)":        mean_all,
            "mean (dedup)":      mean_dedup,
            "mean Δ%":           round((mean_dedup - mean_all) / mean_all * 100, 2) if mean_all else 0,
            "median (all)":      med_all,
            "median (dedup)":    med_dedup,
            "median Δ%":         round((med_dedup - med_all) / med_all * 100, 2) if med_all else 0,
            "p (Mann-Whitney)":  round(p_mw, 4),
            f"sig (α={alpha})":  "Yes" if p_mw < alpha else "No",
        })
    return pd.DataFrame(rows).set_index("feature")


def duplicate_impact_categorical(
    df: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Assess the impact of duplicate rows on categorical feature distributions.

    Compares category proportions between unique records and extra duplicate
    rows using a chi-square test.
    """
    if features is None:
        features = CATEGORICAL_FEATURES
    features = [f for f in features if f in df.columns]
    uniq_recs = df.drop_duplicates()
    dup_flag  = df.duplicated(keep="first").map({True: "duplicate", False: "unique"})

    if dup_flag.sum() == 0:
        print("No duplicate rows found.")
        return pd.DataFrame()

    rows = []
    for col in features:
        ct = pd.crosstab(df[col], dup_flag)
        if ct.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(ct)
        prop_all   = df[col].value_counts(normalize=True).to_dict()
        prop_dedup = uniq_recs[col].value_counts(normalize=True).to_dict()
        max_shift  = max(
            abs(prop_dedup.get(k, 0) - prop_all.get(k, 0)) for k in prop_all
        )
        rows.append({
            "feature":          col,
            "chi2":             round(chi2, 3),
            "p-value":          round(p, 4),
            "max prop. shift":  round(max_shift * 100, 2),
            f"sig (α={alpha})": "Yes" if p < alpha else "No",
        })
    return pd.DataFrame(rows).set_index("feature")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Drop high-missing columns (>50%).
    2. Rename columns to snake_case.
    3. Remove rows with missing/invalid target (price <= 0).
    4. Remove extreme price outliers (> 99.5th percentile or < 1st percentile).
    5. Remove rows with negative area values.
    6. Drop remaining duplicates.
    7. Reset index.

    Returns the cleaned DataFrame and prints a summary.
    """
    n_raw = len(df)

    # 1. Drop high-missing columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Rename
    df = df.rename(columns=RENAME_COLS)

    # 3. Remove invalid target
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    # 4. Remove extreme price outliers
    p995 = df[TARGET].quantile(0.995)
    p01  = df[TARGET].quantile(0.01)
    df = df[(df[TARGET] >= p01) & (df[TARGET] <= p995)]

    # 5. Remove negative area values
    area_cols = [c for c in ["total_area", "living_area"] if c in df.columns]
    for col in area_cols:
        df = df[df[col].fillna(0) >= 0]

    # 6. Drop duplicates
    df = df.drop_duplicates()

    # 7. Reset index
    df = df.reset_index(drop=True)

    print(f"Cleaned dataset : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Rows removed    : {n_raw - len(df):,}")
    print(f"Median price    : €{df[TARGET].median():,.0f}")
    print(f"Mean price      : €{df[TARGET].mean():,.0f}")
    return df


def drop_duplicates_clean(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows, reset the index, and print a cleaning summary.
    """
    cleaned = df.drop_duplicates().reset_index(drop=True)
    print(f"Cleaned dataset : {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns")
    print(f"Rows removed    : {len(raw_df) - len(cleaned):,} duplicate rows")
    print(f"Median price    : €{cleaned[TARGET].median():,.0f}")
    return cleaned


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    80/20 train / test split (no stratification — continuous target).

    Returns (train_df, test_df). The test set is treated as a held-out set
    and must not be touched until final evaluation.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"Train set : {len(train_df):,} rows  (median price: €{train_df[TARGET].median():,.0f})")
    print(f"Test set  : {len(test_df):,} rows  (median price: €{test_df[TARGET].median():,.0f})")
    return train_df, test_df

"""
plots.py
--------
All Matplotlib / Seaborn visualization functions.

Design rules:
  - Every function returns a Figure without calling plt.show().
  - apply_global_style() sets project-wide aesthetics; call once at notebook start.
  - No hardcoded colors: all palettes come from portugal_housing.config.
  - Axes always carry title, x-label, and y-label.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats as sp_stats

from portugal_housing.config import (
    CMAP_DIV,
    CMAP_SEQ,
    PALETTE_ACCENT,
    PALETTE_LIST,
    PALETTE_PRIMARY,
    TARGET,
)

# ── Global style ──────────────────────────────────────────────────────────────

def apply_global_style() -> None:
    """Apply project-wide Matplotlib/Seaborn styling. Call once at notebook start."""
    sns.set_theme(style="whitegrid", palette=PALETTE_LIST, font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi":        120,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#E8E8E8",
        "grid.linewidth":    0.7,
        "legend.frameon":    False,
        "font.size":         11,
    })


# ── Target distribution ───────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> Figure:
    """
    Side-by-side histograms of price: raw scale (left) and log scale (right).
    Vertical lines mark mean and median.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw scale
    ax = axes[0]
    sns.histplot(df[TARGET], bins=80, color=PALETTE_PRIMARY, alpha=0.7, ax=ax, kde=True)
    med = df[TARGET].median()
    mean = df[TARGET].mean()
    ax.axvline(med, color=PALETTE_ACCENT, linestyle="--", linewidth=1.5, label=f"Median: €{med:,.0f}")
    ax.axvline(mean, color="#C44E52", linestyle=":", linewidth=1.5, label=f"Mean: €{mean:,.0f}")
    ax.set_title("Price Distribution (Raw)", fontsize=12)
    ax.set_xlabel("Price (€)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # Log scale
    ax = axes[1]
    log_price = np.log1p(df[TARGET])
    sns.histplot(log_price, bins=80, color=PALETTE_PRIMARY, alpha=0.7, ax=ax, kde=True)
    log_med = np.log1p(med)
    ax.axvline(log_med, color=PALETTE_ACCENT, linestyle="--", linewidth=1.5,
               label=f"Median: {log_med:.2f}")
    ax.set_title("Price Distribution (Log Scale)", fontsize=12)
    ax.set_xlabel("log(1 + Price)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    fig.suptitle("Target Variable: Asking Price", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ── Split verification ───────────────────────────────────────────────────────

def plot_split_verification(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Figure:
    """
    KDE overlay comparing price distributions in train and test sets.
    Confirms that the random split preserved the price distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Raw price
    ax = axes[0]
    sns.kdeplot(train_df[TARGET], ax=ax, color=PALETTE_PRIMARY, lw=2, label=f"Train (n={len(train_df):,})")
    sns.kdeplot(test_df[TARGET], ax=ax, color=PALETTE_ACCENT, lw=2, label=f"Test (n={len(test_df):,})")
    ax.set_title("Price Distribution: Train vs Test", fontsize=11)
    ax.set_xlabel("Price (€)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    # Log price
    ax = axes[1]
    sns.kdeplot(np.log1p(train_df[TARGET]), ax=ax, color=PALETTE_PRIMARY, lw=2, label="Train")
    sns.kdeplot(np.log1p(test_df[TARGET]), ax=ax, color=PALETTE_ACCENT, lw=2, label="Test")
    ax.set_title("Log-Price Distribution: Train vs Test", fontsize=11)
    ax.set_xlabel("log(1 + Price)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

    fig.suptitle("Split Verification: Price Distribution Preserved", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ── Numerical distributions ──────────────────────────────────────────────────

def plot_numerical_distributions(df: pd.DataFrame, features: list[str]) -> Figure:
    """
    Histplots with KDE for each numerical feature.
    Vertical dashed line marks the median.
    """
    n     = len(features)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.5 * nrows))
    axes = axes.flatten()

    for ax, feat in zip(axes, features, strict=False):
        s = df[feat].dropna()
        sns.histplot(s, kde=True, ax=ax, color=PALETTE_PRIMARY, alpha=0.5, bins=50)
        median = s.median()
        ax.axvline(median, color=PALETTE_ACCENT, linestyle="--", linewidth=1.4)
        ax.text(
            median, ax.get_ylim()[1] * 0.88, f"med={median:,.1f}",
            color=PALETTE_ACCENT, fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ── Categorical distributions ────────────────────────────────────────────────

def plot_categorical_distributions(
    df: pd.DataFrame,
    features: list[str],
    top_n: int = 10,
) -> Figure:
    """
    Barplots showing median price per category level for each feature.
    Only the top_n categories (by count) are shown.
    """
    n     = len(features)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, feat in zip(axes_flat, features, strict=False):
        top_cats = df[feat].value_counts().head(top_n).index.tolist()
        sub = df[df[feat].isin(top_cats)]
        rate_df = (
            sub.groupby(feat)[TARGET]
            .agg(["median", "count"])
            .rename(columns={"median": "median_price", "count": "n"})
            .reset_index()
            .sort_values("median_price", ascending=False)
        )

        sns.barplot(
            data=rate_df, x="median_price", y=feat,
            hue=feat, palette=PALETTE_LIST * 3, legend=False,
            ax=ax, orient="h",
        )
        for i, (_, row) in enumerate(rate_df.iterrows()):
            ax.text(
                row["median_price"] + rate_df["median_price"].max() * 0.02, i,
                f"€{row['median_price']:,.0f}  (n={row['n']:,})",
                va="center", fontsize=9,
            )
        ax.set_title(f"Median Price by {feat}", fontsize=11)
        ax.set_xlabel("Median Price (€)")
        ax.set_ylabel("")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Categorical Features vs. Price", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ── Single-metric bar (for query results) ────────────────────────────────────

def plot_metric_by_category(
    series: pd.Series,
    title: str,
    xlabel: str,
    top_n: int = 10,
    euro: bool = True,
) -> Figure:
    """
    Horizontal bar chart of one metric per category, sorted high to low.

    Turns a query/groupby result (index = category, values = metric) into a
    chart so differences read at a glance instead of scanning a table.
    """
    s = series.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(3, len(s) * 0.5)))
    sns.barplot(x=s.values, y=s.index.astype(str),
                hue=s.index.astype(str), palette=PALETTE_LIST * 3,
                legend=False, ax=ax, orient="h")
    pad = s.max() * 0.02
    for i, v in enumerate(s.values):
        label = f"€{v:,.0f}" if euro else f"{v:,.0f}"
        ax.text(v + pad, i, label, va="center", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_xlim(0, s.max() * 1.15)
    fig.tight_layout()
    return fig


# ── Boxplots by category ─────────────────────────────────────────────────────

def plot_boxplots_by_category(
    df: pd.DataFrame,
    category: str,
    top_n: int = 8,
) -> Figure:
    """
    Boxplots of price by a categorical feature (e.g., district, type).
    Uses log-price for better visual spread.
    """
    top_cats = df[category].value_counts().head(top_n).index.tolist()
    sub = df[df[category].isin(top_cats)].copy()
    sub["log_price"] = np.log1p(sub[TARGET])

    order = sub.groupby(category)["log_price"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.6)))
    sns.boxplot(
        data=sub, x="log_price", y=category,
        order=order, palette=PALETTE_LIST * 3,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        ax=ax,
    )
    ax.set_title(f"Log-Price Distribution by {category}", fontsize=12)
    ax.set_xlabel("log(1 + Price)")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


# ── Correlation heatmap ──────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> Figure:
    """
    Side-by-side lower-triangle heatmaps: Pearson (left) and Spearman (right).
    The target column/row is bolded in both panels.
    """
    cols = features + [TARGET]
    corr_p = df[cols].corr(method="pearson")
    corr_s = df[cols].corr(method="spearman")
    mask   = np.triu(np.ones_like(corr_p, dtype=bool), k=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, corr, label, cbar_label in zip(
        axes,
        [corr_p, corr_s],
        ["Pearson Correlation", "Spearman Correlation"],
        ["Pearson r", "Spearman ρ"],
        strict=False,
    ):
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap=CMAP_DIV, center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.6, linecolor="white",
            annot_kws={"size": 10, "weight": "bold"},
            cbar_kws={"shrink": 0.8, "label": cbar_label},
        )
        tick_labels = ax.get_xticklabels()
        for lbl in tick_labels:
            if lbl.get_text() == TARGET:
                lbl.set_fontweight("bold")
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(label, fontsize=12, pad=12)

    fig.tight_layout()
    return fig


def plot_phik_heatmap(df: pd.DataFrame, columns: list[str],
                      interval_cols: list[str]) -> Figure:
    """
    Phi-k correlation heatmap over a mix of numerical and categorical columns.

    Pearson and Spearman only see numeric pairs; phi-k also captures the
    association between categoricals like district or type and the price,
    putting every feature on one comparable 0-to-1 scale. interval_cols lists
    which of the columns are continuous; the rest are treated as categorical.
    """
    import phik  # noqa: F401  (registers the .phik_matrix accessor)

    corr = df[columns].phik_matrix(interval_cols=interval_cols)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=CMAP_SEQ, vmin=0, vmax=1, ax=ax,
        linewidths=0.6, linecolor="white",
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "phi-k"},
    )
    tick_labels = ax.get_xticklabels()
    for lbl in tick_labels:
        if lbl.get_text() == TARGET:
            lbl.set_fontweight("bold")
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("Phi-k correlation: numerical and categorical features together",
                 fontsize=12, pad=12)
    fig.tight_layout()
    return fig


# ── Scatter plots ────────────────────────────────────────────────────────────

def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str = TARGET,
    hue: str | None = None,
    log_y: bool = True,
) -> Figure:
    """
    Scatter plot of x vs y (default: price). Optional hue for categorical split.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_y = np.log1p(df[y]) if log_y else df[y]
    y_label = f"log(1 + {y})" if log_y else y

    if hue and hue in df.columns:
        top_cats = df[hue].value_counts().head(5).index.tolist()
        sub = df[df[hue].isin(top_cats)]
        plot_y_sub = np.log1p(sub[y]) if log_y else sub[y]
        sns.scatterplot(
            data=sub, x=x, y=plot_y_sub,
            hue=hue, palette=PALETTE_LIST, alpha=0.3, s=12, ax=ax,
        )
    else:
        sns.scatterplot(
            x=df[x], y=plot_y,
            color=PALETTE_PRIMARY, alpha=0.3, s=12, ax=ax,
        )

    ax.set_xlabel(x, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{x} vs. {y_label}", fontsize=12)
    fig.tight_layout()
    return fig


# ── CV comparison ─────────────────────────────────────────────────────────────

def plot_cv_comparison(cv_df: pd.DataFrame, metric: str = "neg_root_mean_squared_error") -> Figure:
    """
    Horizontal barplot of mean CV scores with std error bars.
    For neg_* metrics, values are negated for display (so bars show positive RMSE).
    """
    col_mean = f"cv_{metric}_mean"
    col_std  = f"cv_{metric}_std"

    plot_df = cv_df.reset_index().copy()

    # For neg_* metrics, negate so the bars are positive and lower is better
    is_neg = metric.startswith("neg_")
    if is_neg:
        plot_df[col_mean] = -plot_df[col_mean]
        plot_df[col_std]  = plot_df[col_std]  # std stays positive
        display_metric = metric.replace("neg_", "")
    else:
        display_metric = metric

    # Sort so the best model sits at the top. For error metrics (neg_*, lower is
    # better) that means ascending; for score metrics (higher is better) descending.
    plot_df = plot_df.sort_values(col_mean, ascending=is_neg).reset_index(drop=True)
    best_model = plot_df.loc[0, "model"]

    # Highlight only the winner; mute the rest so the eye goes to the best model.
    plot_df["_color"] = [
        PALETTE_PRIMARY if m == best_model else "#C9C9C9"
        for m in plot_df["model"]
    ]
    palette_map = dict(zip(plot_df["model"], plot_df["_color"], strict=False))

    fig, ax = plt.subplots(figsize=(9, max(3, len(plot_df) * 0.75)))
    sns.barplot(
        data=plot_df, y="model", x=col_mean, hue="model",
        palette=palette_map, ax=ax, orient="h", legend=False,
    )
    for i, (_, row) in enumerate(plot_df.iterrows()):
        is_best = row["model"] == best_model
        ax.errorbar(
            row[col_mean], i, xerr=row[col_std],
            fmt="none", color="black", capsize=4, linewidth=1.5,
        )
        ax.text(
            row[col_mean] + row[col_std] + plot_df[col_mean].max() * 0.01, i,
            f"{row[col_mean]:,.0f} ± {row[col_std]:,.0f}" + ("  (best)" if is_best else ""),
            va="center", fontsize=9,
            fontweight="bold" if is_best else "normal",
        )

    ax.set_xlabel(f"CV {display_metric.replace('_', ' ').upper()} (mean)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(f"Cross-Validation Comparison: {display_metric.upper()}", fontsize=12)
    fig.tight_layout()
    return fig


# ── Actual vs Predicted ──────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> Figure:
    """
    Scatter plot of actual vs predicted values with identity line.
    Uses log scale for both axes since price is right-skewed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw scale
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.2, s=8, color=PALETTE_PRIMARY)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="Perfect prediction")
    ax.set_xlabel("Actual Price (€)", fontsize=11)
    ax.set_ylabel("Predicted Price (€)", fontsize=11)
    ax.set_title(f"{model_name}: Actual vs Predicted", fontsize=12)
    ax.legend(fontsize=9)

    # Log scale
    ax = axes[1]
    ax.scatter(y_true, y_pred, alpha=0.2, s=8, color=PALETTE_PRIMARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="Perfect prediction")
    ax.set_xlabel("Actual Price (€, log scale)", fontsize=11)
    ax.set_ylabel("Predicted Price (€, log scale)", fontsize=11)
    ax.set_title(f"{model_name}: Actual vs Predicted (Log Scale)", fontsize=12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


# ── Residual plots ───────────────────────────────────────────────────────────

def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> Figure:
    """
    Two-panel residual diagnostics:
    Left: residuals vs fitted values (check homoscedasticity).
    Right: Q-Q plot (check normality of residuals).
    """
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs fitted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.15, s=8, color=PALETTE_PRIMARY)
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Predicted Price (€)", fontsize=11)
    ax.set_ylabel("Residual (€)", fontsize=11)
    ax.set_title(f"{model_name}: Residuals vs Fitted", fontsize=12)

    # Q-Q plot
    ax = axes[1]
    sp_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{model_name}: Q-Q Plot of Residuals", fontsize=12)
    ax.get_lines()[0].set(color=PALETTE_PRIMARY, markersize=3, alpha=0.4)
    ax.get_lines()[1].set(color=PALETTE_ACCENT, linewidth=1.5)

    fig.tight_layout()
    return fig


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str,
    top_n: int = 15,
) -> Figure:
    """
    Horizontal barplot of feature importances.
    """
    idx   = np.argsort(importances)[-top_n:]
    names = np.array(feature_names)[idx]
    vals  = importances[idx]
    total = vals.sum()

    plot_df = pd.DataFrame({"feature": names, "importance": vals})
    plot_df["_color"] = [
        PALETTE_PRIMARY if v >= np.median(vals) else "#AAAAAA" for v in vals
    ]
    palette_map = dict(zip(plot_df["feature"], plot_df["_color"], strict=False))

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    sns.barplot(data=plot_df, y="feature", x="importance", hue="feature",
                palette=palette_map, ax=ax, orient="h", legend=False)

    for i, (val, _name) in enumerate(zip(vals, names, strict=False)):
        ax.text(val + total * 0.005, i, f"{val/total*100:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Importance (mean decrease in impurity)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(f"Feature Importances: {model_name} (top {top_n})", fontsize=12)
    fig.tight_layout()
    return fig


# ── Learning curve ───────────────────────────────────────────────────────────

def plot_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> Figure:
    """
    Learning curve with shaded std band.
    For neg_* metrics, values are negated for display.
    """
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, random_state=42,
    )

    is_neg = scoring.startswith("neg_")
    if is_neg:
        train_scores = -train_scores
        val_scores   = -val_scores
        display_metric = scoring.replace("neg_", "").replace("_", " ").upper()
    else:
        display_metric = scoring.upper()

    t_mean = train_scores.mean(axis=1)
    t_std  = train_scores.std(axis=1)
    v_mean = val_scores.mean(axis=1)
    v_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_sizes, t_mean, "o-", color=PALETTE_PRIMARY, lw=2, label="Train")
    ax.fill_between(train_sizes, t_mean - t_std, t_mean + t_std,
                    alpha=0.15, color=PALETTE_PRIMARY)
    ax.plot(train_sizes, v_mean, "s-", color=PALETTE_ACCENT, lw=2, label="CV Validation")
    ax.fill_between(train_sizes, v_mean - v_std, v_mean + v_std,
                    alpha=0.15, color=PALETTE_ACCENT)

    ax.annotate(
        f"Train {t_mean[-1]:,.0f}",
        xy=(train_sizes[-1], t_mean[-1]),
        xytext=(train_sizes[-1] * 0.82, t_mean[-1] + t_mean[-1] * 0.02),
        fontsize=9, color=PALETTE_PRIMARY, fontweight="bold",
    )
    ax.annotate(
        f"CV {v_mean[-1]:,.0f}",
        xy=(train_sizes[-1], v_mean[-1]),
        xytext=(train_sizes[-1] * 0.82, v_mean[-1] - v_mean[-1] * 0.04),
        fontsize=9, color=PALETTE_ACCENT, fontweight="bold",
    )

    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel(display_metric, fontsize=11)
    ax.set_title(f"Learning Curve: {model_name}", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ── Export ────────────────────────────────────────────────────────────────────

def save_figure(fig: Figure, name: str, figures_dir) -> None:
    """Save a figure to figures_dir as PNG at 150 dpi."""
    from pathlib import Path
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / f"{name}.png", dpi=150, bbox_inches="tight")


# ── Interactive Plotly Dashboard ─────────────────────────────────────────────

def dashboard_portugal_housing(
    df: pd.DataFrame,
    cv_results: pd.DataFrame,
    final_metrics: dict,
    final_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shap_values=None,
    feature_names: list[str] = None,
    *,
    out_path: str | None = None,
):
    """Interactive dark-themed executive dashboard for Portugal Housing Prediction."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_total = len(df)
    median_price = df[TARGET].median()

    top_model_name = sorted(
        final_metrics, key=lambda k: final_metrics[k].get("rmse", float("inf"))
    )[0]
    top_m = final_metrics[top_model_name]

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.33, 0.33, 0.34],
        column_widths=[0.5, 0.5],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Cross-Validation: RMSE Comparison",
            "Actual vs Predicted (Test Set)",
            "Feature Importances",
            "Residual Distribution",
            "Price Distribution (Log)",
            "Final Test Metrics",
        ],
    )

    # Row 1, Col 1: CV comparison
    cv_col = [c for c in cv_results.columns if "mean" in c][0]
    cv_std_col = [c for c in cv_results.columns if "std" in c][0]
    cv_sorted = cv_results.reset_index().copy()
    cv_sorted[cv_col] = -cv_sorted[cv_col]  # negate neg_rmse
    cv_sorted = cv_sorted.sort_values(cv_col, ascending=False)

    # Highlight only the best model (lowest RMSE); mute the rest.
    best_rmse = cv_sorted[cv_col].min()
    bar_colors = ["#6366f1" if v == best_rmse else "#C9C9C9" for v in cv_sorted[cv_col]]
    fig.add_trace(
        go.Bar(
            y=cv_sorted["model"], x=cv_sorted[cv_col], orientation="h",
            marker_color=bar_colors,
            error_x=dict(type="data", array=cv_sorted[cv_std_col].values, color="#94a3b8"),
            text=[f"€{v:,.0f}" for v in cv_sorted[cv_col]],
            textposition="outside", textfont=dict(size=11), showlegend=False,
        ), row=1, col=1,
    )
    fig.update_xaxes(title_text="RMSE (€)", row=1, col=1)

    # Row 1, Col 2: Actual vs predicted
    y_pred = top_m["y_pred"]
    fig.add_trace(
        go.Scatter(
            x=y_test.values, y=y_pred, mode="markers",
            marker=dict(size=3, color="#6366f1", opacity=0.3),
            name="Predictions", showlegend=False,
        ), row=1, col=2,
    )
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    fig.add_trace(
        go.Scatter(x=lims, y=lims, mode="lines",
                   line=dict(color="#f97316", dash="dash", width=1.5),
                   showlegend=False), row=1, col=2,
    )
    fig.update_xaxes(title_text="Actual (€)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted (€)", row=1, col=2)

    # Row 2, Col 1: Feature importance.
    # The top model by RMSE may be an ensemble (e.g. VotingRegressor) that has
    # no feature_importances_, so pull them from the first tree model that does.
    # LightGBM is preferred to match the SHAP/importance analysis in Section 14.
    if feature_names:
        ordered = sorted(final_models.items(),
                         key=lambda kv: 0 if "LightGBM" in kv[0] else 1)
        for _, m in ordered:
            clf = m.named_steps.get("clf") if hasattr(m, "named_steps") else None
            if clf is not None and hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
                top_idx = np.argsort(imp)[-10:]
                fig.add_trace(
                    go.Bar(
                        y=[feature_names[i] for i in top_idx],
                        x=imp[top_idx], orientation="h",
                        marker_color="#6366f1", showlegend=False,
                    ), row=2, col=1,
                )
                break
    fig.update_xaxes(title_text="Importance", row=2, col=1)

    # Row 2, Col 2: Residual distribution
    residuals = y_test.values - y_pred
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=80, marker_color="#6366f1",
                     opacity=0.7, showlegend=False),
        row=2, col=2,
    )
    fig.update_xaxes(title_text="Residual (€)", row=2, col=2)

    # Row 3, Col 1: Log price distribution
    log_prices = np.log1p(df[TARGET])
    fig.add_trace(
        go.Histogram(x=log_prices, nbinsx=80, marker_color="#6366f1",
                     opacity=0.7, showlegend=False),
        row=3, col=1,
    )
    fig.update_xaxes(title_text="log(1+Price)", row=3, col=1)

    # Row 3, Col 2: Metrics table
    model_names = list(final_metrics.keys())
    metric_keys = ["rmse", "mae", "r2", "mape"]
    table_vals = {k: [] for k in ["Model"] + metric_keys}
    for name in model_names:
        table_vals["Model"].append(name)
        for mk in metric_keys:
            v = final_metrics[name].get(mk, 0)
            fmt = f"€{v:,.0f}" if mk in ("rmse", "mae") else f"{v:.4f}"
            table_vals[mk].append(fmt)

    header_labels = ["Model", "RMSE", "MAE", "R²", "MAPE"]
    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_labels],
                fill_color="#1e1b4b", font=dict(color="white", size=12),
                align="center", height=32,
            ),
            cells=dict(
                values=[table_vals[k] for k in ["Model"] + metric_keys],
                fill_color=[["#2d2a5e"] * len(model_names)],
                font=dict(color="white", size=11), align="center", height=28,
            ),
        ), row=3, col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f0e1a", plot_bgcolor="#1a1933",
        font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0", size=13),
        title=dict(
            text=(
                "<b>Portugal Housing Price Prediction: ML Dashboard</b>"
                f"<br><span style='font-size:13px; color:#94a3b8'>"
                f"Samples: {n_total:,} | Median Price: €{median_price:,.0f} | "
                f"Best Model: {top_model_name} "
                f"(RMSE: €{top_m['rmse']:,.0f}, R²: {top_m['r2']:.4f})</span>"
            ),
            font=dict(size=18, color="#818cf8"), x=0.5, xanchor="center",
        ),
        height=1150, margin=dict(t=100, b=40, l=60, r=60),
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color="#a5b4fc")

    if out_path is not None:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs=True)

    return fig

"""
auto_insights/viz.py

Visualization module — produces matplotlib figures for the report.

Public API
----------
generate_all_figures(df, stats_result)  ->  dict[str, Figure]
    Top-level entry point called by core.py.
    Returns a flat dict of {figure_label: matplotlib Figure}.

Figure inventory
----------------
Numeric columns (one figure each):
    "hist_{col}"        – histogram with KDE overlay + IQR fence lines
    "box_{col}"         – horizontal boxplot with outlier jitter

Categorical / boolean columns:
    "bar_{col}"         – horizontal bar chart (top-N categories)

Dataset-level:
    "missing_heatmap"   – column × row missingness heatmap (if nulls exist)
    "null_bar"          – bar chart of null % per column (if nulls exist)
    "numeric_overview"  – small-multiples grid of histograms for all numeric cols

Notes
-----
- All figures use a clean, paper-friendly style (no heavy gridlines, muted palette).
- Figures are returned as Figure objects; fig_to_base64() in report.py handles
  serialization. Callers must close figures after serialization to free memory.
- seaborn is used for KDE and styling only; core plotting stays in matplotlib
  so callers have full control over figure objects.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in notebooks and scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn as sns

from auto_insights.utils import (
    ColType,
    count_outliers,
    get_logger,
    iqr_bounds,
    safe_numeric_cols,
    split_columns_by_type,
    validate_dataframe,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_PALETTE = {
    "primary"   : "#5B5EA6",   # muted indigo  — histograms, bars
    "secondary" : "#48A999",   # teal          — KDE line
    "accent"    : "#E07B54",   # coral         — fences, highlights
    "neutral"   : "#9B9EA4",   # gray          — boxplot bodies
    "danger"    : "#C0392B",   # red           — outlier points
    "bg"        : "#FAFAFA",
    "grid"      : "#EBEBEB",
}

_FIG_DPI       = 120
_BAR_MAX_CATS  = 15      # max categories shown in bar charts
_HIST_BINS     = "auto"  # passed to plt.hist; numpy picks the count
_JITTER_ALPHA  = 0.35    # transparency of outlier jitter points
_SMALL_MULT_COLS = 3     # columns in the numeric overview grid


def _apply_base_style(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent axis styling across all figure types."""
    ax.set_facecolor(_PALETTE["bg"])
    ax.figure.set_facecolor("white")
    ax.grid(axis="y", color=_PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.grid(axis="x", color=_PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_PALETTE["grid"])
    if title:
        ax.set_title(title, fontsize=11, fontweight="medium", pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    df: pd.DataFrame,
    stats_result: dict[str, Any] | None = None,
    max_numeric_individual: int = 20,
    max_categorical_individual: int = 15,
) -> dict[str, plt.Figure]:
    """
    Generate all figures for the report.

    Parameters
    ----------
    df                       : input DataFrame
    stats_result             : output of stats.profile_dataframe(df).
                               If None, skips fence lines on histograms.
    max_numeric_individual   : cap on individual hist+box figures to avoid
                               generating hundreds of plots on wide DataFrames.
    max_categorical_individual: cap on individual bar chart figures.

    Returns
    -------
    dict mapping figure label -> matplotlib Figure.
    Keys follow the naming convention described in the module docstring.
    """
    validate_dataframe(df)
    groups    = split_columns_by_type(df)
    num_cols  = groups.get(ColType.NUMERIC,     [])[:max_numeric_individual]
    cat_cols  = (
        groups.get(ColType.CATEGORICAL, []) + groups.get(ColType.BOOLEAN, [])
    )[:max_categorical_individual]

    figures: dict[str, plt.Figure] = {}

    # --- per numeric column ---
    numeric_stats = (stats_result or {}).get("numeric", {})
    for col in num_cols:
        col_stats = numeric_stats.get(col, {})
        try:
            figures[f"hist_{col}"] = _histogram(df[col], col, col_stats)
            figures[f"box_{col}"]  = _boxplot(df[col], col, col_stats)
        except Exception as exc:
            logger.warning("viz: skipping %s — %s", col, exc)

    # --- per categorical column ---
    cat_stats = (stats_result or {}).get("categorical", {})
    for col in cat_cols:
        col_stats = cat_stats.get(col, {})
        try:
            figures[f"bar_{col}"] = _bar_chart(df[col], col, col_stats)
        except Exception as exc:
            logger.warning("viz: skipping bar_%s — %s", col, exc)

    # --- dataset-level figures ---
    try:
        figures["numeric_overview"] = _numeric_overview_grid(df, num_cols)
    except Exception as exc:
        logger.warning("viz: skipping numeric_overview — %s", exc)

    if df.isnull().any().any():
        try:
            figures["null_bar"] = _null_bar(df)
        except Exception as exc:
            logger.warning("viz: skipping null_bar — %s", exc)

        if len(df) <= 500:   # heatmap is expensive for large DataFrames
            try:
                figures["missing_heatmap"] = _missing_heatmap(df)
            except Exception as exc:
                logger.warning("viz: skipping missing_heatmap — %s", exc)

    logger.info("viz: generated %d figures.", len(figures))
    return figures


# ---------------------------------------------------------------------------
# Histogram with KDE
# ---------------------------------------------------------------------------

def _histogram(
    series: pd.Series,
    col: str,
    col_stats: dict,
) -> plt.Figure:
    """
    Histogram with a seaborn KDE overlay and IQR fence lines.

    Layout: the histogram occupies the full axes. Fence lines are drawn as
    vertical dashed lines with small annotations. A text box in the top-right
    corner shows mean ± std.
    """
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=_FIG_DPI)

    # Histogram
    ax.hist(
        s, bins=_HIST_BINS,
        color=_PALETTE["primary"], alpha=0.75,
        edgecolor="white", linewidth=0.4, zorder=2,
        density=True,
    )

    # KDE overlay
    try:
        sns.kdeplot(s, ax=ax, color=_PALETTE["secondary"], linewidth=1.5, zorder=3)
    except Exception:
        pass   # KDE fails on near-constant series; skip gracefully

    # IQR fence lines
    lower, upper = iqr_bounds(s)
    ymax = ax.get_ylim()[1]
    for fence, label in [(lower, "Q1−1.5·IQR"), (upper, "Q3+1.5·IQR")]:
        ax.axvline(fence, color=_PALETTE["accent"], linewidth=1.0,
                   linestyle="--", zorder=4, alpha=0.8)
        ax.text(fence, ymax * 0.92, label, fontsize=6.5,
                color=_PALETTE["accent"], ha="center", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    # Mean ± std annotation
    mean = col_stats.get("mean", float(s.mean()))
    std  = col_stats.get("std",  float(s.std()))
    ax.text(
        0.98, 0.97,
        f"mean {mean:.3g}\nstd  {std:.3g}",
        transform=ax.transAxes, fontsize=7.5,
        va="top", ha="right",
        bbox=dict(fc="white", ec=_PALETTE["grid"], alpha=0.85, pad=4, boxstyle="round,pad=0.3"),
    )

    _apply_base_style(ax, title=col, xlabel=col, ylabel="density")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Boxplot
# ---------------------------------------------------------------------------

def _boxplot(
    series: pd.Series,
    col: str,
    col_stats: dict,
) -> plt.Figure:
    """
    Horizontal boxplot with a jittered strip of outlier points overlaid.

    The jitter is generated with a fixed random seed so reports are
    reproducible. Outliers are coloured in the danger palette so they
    stand out clearly.
    """
    s = series.dropna()
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=_FIG_DPI)

    # Boxplot (suppress fliers — we draw our own)
    bp = ax.boxplot(
        s, vert=False, patch_artist=True,
        showfliers=False, widths=0.5,
        boxprops    =dict(facecolor=_PALETTE["neutral"], alpha=0.5, linewidth=0.8),
        medianprops =dict(color=_PALETTE["primary"], linewidth=2),
        whiskerprops=dict(color=_PALETTE["neutral"], linewidth=0.8),
        capprops    =dict(color=_PALETTE["neutral"], linewidth=0.8),
    )

    # Jittered outliers
    lower, upper = iqr_bounds(s)
    outliers = s[(s < lower) | (s > upper)]
    if len(outliers) > 0:
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.2, 0.2, size=len(outliers))
        ax.scatter(
            outliers, np.ones(len(outliers)) + jitter,
            color=_PALETTE["danger"], s=18, alpha=_JITTER_ALPHA,
            zorder=5, label=f"{len(outliers)} outlier(s)",
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # Stat annotation: median and IQR
    median = col_stats.get("median", float(s.median()))
    iqr    = col_stats.get("iqr",    float(s.quantile(0.75) - s.quantile(0.25)))
    ax.text(
        0.01, 0.97,
        f"median {median:.3g}   IQR {iqr:.3g}",
        transform=ax.transAxes, fontsize=7.5,
        va="top", ha="left",
        bbox=dict(fc="white", ec=_PALETTE["grid"], alpha=0.85, pad=4, boxstyle="round,pad=0.3"),
    )

    ax.set_yticks([])
    ax.set_xlabel(col, fontsize=9)
    _apply_base_style(ax, title=f"{col} — distribution")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bar chart (categorical)
# ---------------------------------------------------------------------------

def _bar_chart(
    series: pd.Series,
    col: str,
    col_stats: dict,
) -> plt.Figure:
    """
    Horizontal bar chart showing the top-N category frequencies.

    Bars are annotated with both count and percentage. If the column has
    more than _BAR_MAX_CATS categories, only the top-N are shown and a
    subtitle notes how many were omitted.
    """
    s  = series.dropna()
    vc = s.value_counts(dropna=True)
    n_total  = len(s)
    n_cats   = len(vc)
    vc_trunc = vc.head(_BAR_MAX_CATS)
    n_shown  = len(vc_trunc)

    fig_height = max(2.5, 0.4 * n_shown + 1.0)
    fig, ax    = plt.subplots(figsize=(6, fig_height), dpi=_FIG_DPI)

    bars = ax.barh(
        range(n_shown), vc_trunc.values,
        color=_PALETTE["primary"], alpha=0.80,
        edgecolor="white", linewidth=0.4, zorder=2,
    )
    ax.set_yticks(range(n_shown))
    ax.set_yticklabels(
        [str(v)[:30] for v in vc_trunc.index],  # truncate long labels
        fontsize=8,
    )
    ax.invert_yaxis()

    # Annotate each bar with count + pct
    x_max = vc_trunc.values.max()
    for i, (val, cnt) in enumerate(zip(vc_trunc.index, vc_trunc.values)):
        pct  = cnt / n_total * 100
        text = f"{cnt:,}  ({pct:.1f}%)"
        x_pos = cnt + x_max * 0.01
        ax.text(x_pos, i, text, va="center", ha="left", fontsize=7.5, color="#444")

    # Subtitle if truncated
    title = col
    if n_cats > n_shown:
        title += f"  (top {n_shown} of {n_cats} categories)"

    _apply_base_style(ax, title=title, xlabel="count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(right=x_max * 1.22)   # room for annotations
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Numeric overview grid (small multiples)
# ---------------------------------------------------------------------------

def _numeric_overview_grid(
    df: pd.DataFrame,
    num_cols: list[str],
) -> plt.Figure:
    """
    Small-multiples grid of simple histograms for all numeric columns.

    Useful as a single at-a-glance figure for the report summary page.
    Each cell gets a minimal histogram (no KDE, no fences) — the full
    detail is in the individual figures.
    """
    cols    = safe_numeric_cols(df)
    cols    = [c for c in cols if c in (num_cols or cols)]
    n       = len(cols)

    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 2), dpi=_FIG_DPI)
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return fig

    ncols   = min(_SMALL_MULT_COLS, n)
    nrows   = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 2.5 * nrows),
        dpi=_FIG_DPI,
    )
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        s  = df[col].dropna()
        ax.hist(s, bins=30, color=_PALETTE["primary"], alpha=0.75,
                edgecolor="white", linewidth=0.3)
        ax.set_title(col, fontsize=8, fontweight="medium", loc="left", pad=4)
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor(_PALETTE["bg"])
        ax.grid(axis="y", color=_PALETTE["grid"], linewidth=0.5)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric columns — distribution overview", fontsize=10,
                 fontweight="medium", x=0.02, ha="left", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Null / missing data figures
# ---------------------------------------------------------------------------

def _null_bar(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart showing null percentage per column.
    Only columns with at least one null are included.
    Bars are coloured by severity: <5% gray, 5-20% amber, >20% red.
    """
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]

    n       = len(null_pct)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.38 * n + 1.0)), dpi=_FIG_DPI)

    colors = [
        _PALETTE["danger"]   if p > 20
        else "#E07B54"       if p > 5
        else _PALETTE["neutral"]
        for p in null_pct.values
    ]

    ax.barh(range(n), null_pct.values, color=colors, alpha=0.82,
            edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(null_pct.index.tolist(), fontsize=8)
    ax.invert_yaxis()

    for i, pct in enumerate(null_pct.values):
        ax.text(pct + 0.3, i, f"{pct:.1f}%", va="center", ha="left", fontsize=7.5)

    ax.axvline(20, color=_PALETTE["danger"], linewidth=0.8,
               linestyle="--", alpha=0.6, label="20% threshold")
    ax.set_xlim(right=max(null_pct.values) * 1.18)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    _apply_base_style(ax, title="Missing values by column", xlabel="null %")
    fig.tight_layout()
    return fig


def _missing_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Binary heatmap where each cell is white (present) or coloured (missing).
    Rows are observations, columns are features.
    Capped at 500 rows; a random sample is taken for larger DataFrames.
    """
    MAX_ROWS = 500
    sample   = df if len(df) <= MAX_ROWS else df.sample(MAX_ROWS, random_state=42)

    # Only include columns that have at least one null
    cols_with_nulls = sample.columns[sample.isnull().any()].tolist()
    sample = sample[cols_with_nulls]

    fig_w = min(12, max(5, len(cols_with_nulls) * 0.5))
    fig_h = min(8,  max(3, len(sample) * 0.015 + 1.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=_FIG_DPI)

    mask_matrix = sample.isnull().astype(int).values
    ax.imshow(
        mask_matrix,
        aspect="auto",
        cmap=matplotlib.colors.ListedColormap(["white", _PALETTE["danger"]]),
        interpolation="nearest",
    )

    ax.set_xticks(range(len(cols_with_nulls)))
    ax.set_xticklabels(cols_with_nulls, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks([])
    ax.set_ylabel("observations", fontsize=8)

    subtitle = f"({MAX_ROWS}-row sample)" if len(df) > MAX_ROWS else f"({len(df)} rows)"
    _apply_base_style(ax, title=f"Missingness heatmap  {subtitle}")
    fig.tight_layout()
    return fig
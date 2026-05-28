"""
auto_insights/corr.py

Correlation analysis module.

Public API
----------
compute_correlations(df)              ->  dict
    Numerical correlation payloads passed to the LLM and report.

generate_correlation_figures(df, corr_result)  ->  dict[str, Figure]
    All correlation-related matplotlib figures.

Figure inventory
----------------
    "corr_heatmap_pearson"   – annotated Pearson r heatmap
    "corr_heatmap_spearman"  – annotated Spearman ρ heatmap (rank-based)
    "corr_pairplot"          – seaborn pairplot grid (capped at 8 columns)
    "corr_top_pairs"         – horizontal bar chart of strongest |r| pairs

Notes
-----
- Pearson r assumes linearity and normality; Spearman ρ is rank-based and
  more robust to outliers and non-linear monotonic relationships. Both are
  always computed; the LLM is given both so it can comment on divergences
  (a pair where |Pearson| << |Spearman| hints at a monotone but nonlinear
  relationship or outlier influence).
- Point-biserial correlation is computed for binary categorical columns
  paired against numeric columns.
- The pairplot is capped at _PAIRPLOT_MAX_COLS columns and _PAIRPLOT_MAX_ROWS
  rows to keep rendering time manageable.
- All figures share the same visual style as viz.py.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn as sns

from auto_insights.utils import (
    ColType,
    get_logger,
    safe_numeric_cols,
    split_columns_by_type,
    validate_dataframe,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAIRPLOT_MAX_COLS = 8      # cap on columns included in the pairplot
_PAIRPLOT_MAX_ROWS = 2_000  # sample size cap for pairplot rendering speed
_TOP_PAIRS_N       = 20     # strongest pairs shown in the bar chart
_STRONG_THRESHOLD  = 0.70   # |r| above this is flagged as "strong"
_MODERATE_THRESHOLD = 0.40  # |r| above this is flagged as "moderate"

_PALETTE = {
    "pos"     : "#5B5EA6",   # indigo  — positive correlations
    "neg"     : "#E07B54",   # coral   — negative correlations
    "neutral" : "#9B9EA4",   # gray
    "bg"      : "#FAFAFA",
    "grid"    : "#EBEBEB",
    "text"    : "#333333",
}

# Diverging colormap centered at 0 for heatmaps
_HEATMAP_CMAP = "coolwarm"


# ---------------------------------------------------------------------------
# Public entry point — numerical payloads
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute all correlation payloads for the report and LLM prompt.

    Returns
    -------
    dict with keys:
        "pearson"           – full n×n Pearson r matrix (as nested dict)
        "spearman"          – full n×n Spearman ρ matrix (as nested dict)
        "top_pairs_pearson" – list of strongest |r| pairs with metadata
        "top_pairs_spearman"– same for Spearman
        "point_biserial"    – binary-vs-numeric correlations
        "flags"             – human-readable strings for the LLM
        "column_list"       – numeric columns that were analysed
    """
    validate_dataframe(df)
    num_cols = safe_numeric_cols(df)

    if len(num_cols) < 2:
        logger.warning("corr: fewer than 2 numeric columns — skipping correlation analysis.")
        return {"note": "Fewer than 2 numeric columns; correlation analysis skipped."}

    logger.info("corr: computing correlations for %d numeric columns.", len(num_cols))

    num_df = df[num_cols].copy()

    pearson_mat  = _correlation_matrix(num_df, method="pearson")
    spearman_mat = _correlation_matrix(num_df, method="spearman")

    top_pearson  = _top_pairs(pearson_mat,  num_cols, n=_TOP_PAIRS_N)
    top_spearman = _top_pairs(spearman_mat, num_cols, n=_TOP_PAIRS_N)

    pb_results = _point_biserial(df)

    flags = _build_flags(top_pearson, top_spearman, pb_results)

    return {
        "pearson"           : pearson_mat.round(4).to_dict(),
        "spearman"          : spearman_mat.round(4).to_dict(),
        "top_pairs_pearson" : top_pearson,
        "top_pairs_spearman": top_spearman,
        "point_biserial"    : pb_results,
        "flags"             : flags,
        "column_list"       : num_cols,
    }


# ---------------------------------------------------------------------------
# Public entry point — figures
# ---------------------------------------------------------------------------

def generate_correlation_figures(
    df: pd.DataFrame,
    corr_result: dict[str, Any] | None = None,
) -> dict[str, plt.Figure]:
    """
    Generate all correlation figures.

    Parameters
    ----------
    df          : input DataFrame
    corr_result : output of compute_correlations(df). If None, matrices are
                  recomputed from df (slightly slower).

    Returns
    -------
    dict mapping figure label -> matplotlib Figure.
    """
    validate_dataframe(df)
    num_cols = safe_numeric_cols(df)

    if len(num_cols) < 2:
        logger.warning("corr: fewer than 2 numeric columns — no figures generated.")
        return {}

    if corr_result is None or "pearson" not in corr_result:
        corr_result = compute_correlations(df)

    pearson_mat  = pd.DataFrame(corr_result["pearson"])
    spearman_mat = pd.DataFrame(corr_result["spearman"])

    figures: dict[str, plt.Figure] = {}

    try:
        figures["corr_heatmap_pearson"]  = _heatmap(pearson_mat,  "Pearson r")
    except Exception as exc:
        logger.warning("corr: heatmap_pearson failed — %s", exc)

    try:
        figures["corr_heatmap_spearman"] = _heatmap(spearman_mat, "Spearman ρ")
    except Exception as exc:
        logger.warning("corr: heatmap_spearman failed — %s", exc)

    try:
        figures["corr_top_pairs"] = _top_pairs_bar(
            corr_result.get("top_pairs_pearson", [])
        )
    except Exception as exc:
        logger.warning("corr: top_pairs bar failed — %s", exc)

    try:
        figures["corr_pairplot"] = _pairplot(df, num_cols)
    except Exception as exc:
        logger.warning("corr: pairplot failed — %s", exc)

    logger.info("corr: generated %d figures.", len(figures))
    return figures


# ---------------------------------------------------------------------------
# Internal — matrix computation
# ---------------------------------------------------------------------------

def _correlation_matrix(num_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Compute a correlation matrix, handling NaNs via pairwise complete obs.
    Falls back to scipy for pairwise if pandas raises.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mat = num_df.corr(method=method, min_periods=2)
        except Exception:
            # Manual pairwise fallback
            cols = num_df.columns.tolist()
            n    = len(cols)
            data = np.full((n, n), np.nan)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        data[i, j] = 1.0
                        continue
                    a = num_df[cols[i]].dropna()
                    b = num_df[cols[j]].dropna()
                    common = a.index.intersection(b.index)
                    if len(common) < 3:
                        continue
                    fn = (scipy_stats.pearsonr if method == "pearson"
                          else scipy_stats.spearmanr)
                    r, _ = fn(a.loc[common], b.loc[common])
                    data[i, j] = r
            mat = pd.DataFrame(data, index=cols, columns=cols)
    return mat


# ---------------------------------------------------------------------------
# Internal — top pairs extraction
# ---------------------------------------------------------------------------

def _top_pairs(
    mat: pd.DataFrame,
    cols: list[str],
    n: int = _TOP_PAIRS_N,
) -> list[dict]:
    """
    Extract the top-n strongest |correlation| pairs from the upper triangle.

    Returns a list of dicts sorted by |r| descending:
        {"col_a", "col_b", "r", "abs_r", "direction", "strength"}
    """
    pairs: list[dict] = []

    for col_a, col_b in combinations(cols, 2):
        r = mat.loc[col_a, col_b]
        if pd.isna(r):
            continue
        abs_r = abs(r)
        strength = (
            "strong"   if abs_r >= _STRONG_THRESHOLD
            else "moderate" if abs_r >= _MODERATE_THRESHOLD
            else "weak"
        )
        pairs.append({
            "col_a"    : col_a,
            "col_b"    : col_b,
            "r"        : round(float(r), 4),
            "abs_r"    : round(float(abs_r), 4),
            "direction": "positive" if r >= 0 else "negative",
            "strength" : strength,
        })

    pairs.sort(key=lambda x: x["abs_r"], reverse=True)
    return pairs[:n]


# ---------------------------------------------------------------------------
# Internal — point-biserial
# ---------------------------------------------------------------------------

def _point_biserial(df: pd.DataFrame) -> list[dict]:
    """
    Compute point-biserial correlation between every binary column
    (0/1 integer or bool) and every numeric column.

    Point-biserial r is mathematically equivalent to Pearson r between
    a dichotomous and a continuous variable.
    """
    groups    = split_columns_by_type(df)
    num_cols  = safe_numeric_cols(df)
    bool_cols = groups.get(ColType.BOOLEAN, [])

    # Also catch 0/1 integer columns classified as categorical
    binary_cat = [
        col for col in groups.get(ColType.CATEGORICAL, [])
        if df[col].dropna().nunique() == 2
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    dichotomous = list(set(bool_cols + binary_cat))

    results: list[dict] = []

    for d_col in dichotomous:
        for n_col in num_cols:
            if d_col == n_col:
                continue
            combined = df[[d_col, n_col]].dropna()
            if len(combined) < 10:
                continue
            try:
                r, p = scipy_stats.pointbiserialr(
                    combined[d_col].astype(int),
                    combined[n_col],
                )
                results.append({
                    "binary_col" : d_col,
                    "numeric_col": n_col,
                    "r"          : round(float(r), 4),
                    "p_value"    : round(float(p), 4),
                    "abs_r"      : round(abs(float(r)), 4),
                    "strength"   : (
                        "strong"   if abs(r) >= _STRONG_THRESHOLD
                        else "moderate" if abs(r) >= _MODERATE_THRESHOLD
                        else "weak"
                    ),
                })
            except Exception:
                continue

    results.sort(key=lambda x: x["abs_r"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Internal — flags for LLM
# ---------------------------------------------------------------------------

def _build_flags(
    top_pearson : list[dict],
    top_spearman: list[dict],
    pb_results  : list[dict],
) -> list[str]:
    """
    Produce a short list of human-readable flag strings that the LLM can
    use directly in its narrative, e.g.:
        "Strong positive Pearson r=0.92 between 'spin_rate' and 'velocity'"
        "Pearson and Spearman disagree on ('x','y'): r=0.21 vs ρ=0.74 — possible nonlinearity"
    """
    flags: list[str] = []

    # Strong / moderate Pearson pairs
    for p in top_pearson:
        if p["strength"] in ("strong", "moderate"):
            flags.append(
                f"{p['strength'].capitalize()} {p['direction']} Pearson r={p['r']:+.3f} "
                f"between '{p['col_a']}' and '{p['col_b']}'"
            )

    # Pearson vs Spearman divergence (possible nonlinearity / outlier influence)
    spearman_map = {(p["col_a"], p["col_b"]): p["r"] for p in top_spearman}
    spearman_map.update({(p["col_b"], p["col_a"]): p["r"] for p in top_spearman})

    for p in top_pearson:
        rho = spearman_map.get((p["col_a"], p["col_b"]))
        if rho is None:
            continue
        divergence = abs(p["r"] - rho)
        if divergence > 0.20 and abs(rho) > _MODERATE_THRESHOLD:
            flags.append(
                f"Pearson/Spearman divergence on ('{p['col_a']}', '{p['col_b']}'): "
                f"r={p['r']:+.3f} vs ρ={rho:+.3f} — possible nonlinearity or outlier influence"
            )

    # Strong point-biserial results
    for pb in pb_results:
        if pb["strength"] in ("strong", "moderate"):
            flags.append(
                f"{pb['strength'].capitalize()} point-biserial r={pb['r']:+.3f} "
                f"between binary '{pb['binary_col']}' and '{pb['numeric_col']}' "
                f"(p={pb['p_value']:.3f})"
            )

    return flags


# ---------------------------------------------------------------------------
# Figures — heatmap
# ---------------------------------------------------------------------------

def _heatmap(mat: pd.DataFrame, method_label: str) -> plt.Figure:
    """
    Annotated correlation heatmap.

    - Cells are annotated with r values (hidden if |r| < 0.10 to reduce clutter).
    - The diagonal is masked (always 1.0, not informative).
    - Figure height and font sizes scale with the number of columns.
    """
    n      = len(mat)
    cell   = max(0.55, min(0.85, 5.5 / n))    # cell size in inches
    figsize = (n * cell + 1.5, n * cell + 1.0)
    annot_size = max(5.5, min(8.5, 60 / n))

    # Mask the diagonal
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Build annotation matrix: show value only if |r| >= 0.10
    annot_mat = mat.copy()
    annot_mat[mat.abs() < 0.10] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.heatmap(
            mat,
            mask=mask,
            ax=ax,
            cmap=_HEATMAP_CMAP,
            vmin=-1, vmax=1,
            center=0,
            annot=annot_mat.applymap(
                lambda v: f"{v:.2f}" if not np.isnan(v) else ""
            ),
            fmt="",
            annot_kws={"size": annot_size, "color": _PALETTE["text"]},
            linewidths=0.4,
            linecolor=_PALETTE["grid"],
            square=True,
            cbar_kws={"shrink": 0.75, "label": method_label},
        )

    ax.set_title(
        f"Correlation matrix — {method_label}",
        fontsize=11, fontweight="medium", loc="left", pad=10,
    )
    ax.tick_params(axis="x", labelsize=max(6, min(9, 70 / n)),
                   rotation=40, labelrotation=40)
    ax.tick_params(axis="y", labelsize=max(6, min(9, 70 / n)),
                   rotation=0)
    ax.set_facecolor(_PALETTE["bg"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figures — top pairs bar chart
# ---------------------------------------------------------------------------

def _top_pairs_bar(top_pairs: list[dict]) -> plt.Figure:
    """
    Horizontal bar chart of the strongest |r| pairs.

    Bars are coloured by direction: indigo for positive, coral for negative.
    The full r value (signed) is annotated on each bar.
    """
    if not top_pairs:
        fig, ax = plt.subplots(figsize=(6, 2), dpi=120)
        ax.text(0.5, 0.5, "No pairs to display", ha="center", va="center")
        ax.axis("off")
        return fig

    # Limit to top-15 for readability
    pairs   = top_pairs[:15]
    labels  = [f"{p['col_a']}  ×  {p['col_b']}" for p in pairs]
    abs_rs  = [p["abs_r"] for p in pairs]
    signed  = [p["r"]     for p in pairs]
    colors  = [_PALETTE["pos"] if p["r"] >= 0 else _PALETTE["neg"] for p in pairs]

    fig_h = max(3.0, 0.42 * len(pairs) + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_h), dpi=120)

    bars = ax.barh(range(len(pairs)), abs_rs, color=colors,
                   alpha=0.82, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([lbl[:50] for lbl in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.axvline(_STRONG_THRESHOLD,   color="#999", linewidth=0.8,
               linestyle="--", alpha=0.6, label=f"strong (|r|={_STRONG_THRESHOLD})")
    ax.axvline(_MODERATE_THRESHOLD, color="#bbb", linewidth=0.8,
               linestyle=":",  alpha=0.6, label=f"moderate (|r|={_MODERATE_THRESHOLD})")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    for i, (abs_r, r) in enumerate(zip(abs_rs, signed)):
        ax.text(abs_r + 0.01, i, f"{r:+.3f}", va="center", ha="left",
                fontsize=7.5, color=_PALETTE["text"])

    ax.set_facecolor(_PALETTE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_PALETTE["grid"])
    ax.grid(axis="x", color=_PALETTE["grid"], linewidth=0.6)
    ax.set_xlabel("|Pearson r|", fontsize=9)
    ax.set_title("Strongest correlated pairs", fontsize=11,
                 fontweight="medium", loc="left", pad=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figures — pairplot
# ---------------------------------------------------------------------------

def _pairplot(df: pd.DataFrame, num_cols: list[str]) -> plt.Figure:
    """
    Seaborn pairplot of numeric columns.

    - Capped at _PAIRPLOT_MAX_COLS columns (highest-variance columns kept).
    - Capped at _PAIRPLOT_MAX_ROWS rows via stratified sample.
    - Lower triangle: scatter, upper triangle: KDE (faster than scatter × 2).
    - Diagonal: KDE of each variable.
    """
    # Select columns: prefer highest variance (most informative spread)
    variances = df[num_cols].var().sort_values(ascending=False)
    selected  = variances.index[:_PAIRPLOT_MAX_COLS].tolist()

    plot_df = df[selected].dropna()
    if len(plot_df) > _PAIRPLOT_MAX_ROWS:
        plot_df = plot_df.sample(_PAIRPLOT_MAX_ROWS, random_state=42)

    logger.info(
        "corr: pairplot on %d cols × %d rows.", len(selected), len(plot_df)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sns.PairGrid(plot_df, diag_sharey=False)
        g.map_diag(sns.kdeplot, color=_PALETTE["pos"], fill=True, alpha=0.4)
        g.map_lower(
            sns.scatterplot,
            color=_PALETTE["pos"], alpha=0.35, s=12, linewidth=0,
        )
        g.map_upper(
            sns.kdeplot,
            color=_PALETTE["neg"], fill=False, alpha=0.7,
        )

    g.figure.suptitle(
        f"Pairplot — top {len(selected)} numeric columns by variance",
        fontsize=10, fontweight="medium", y=1.01,
    )

    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.set_facecolor(_PALETTE["bg"])
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)

    g.figure.tight_layout()
    return g.figure
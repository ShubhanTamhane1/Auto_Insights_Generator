"""
auto_insights_generator/stats.py

Descriptive statistics module.

Public API
----------
profile_dataframe(df)  ->  dict
    Top-level entry point called by core.py. Returns the full stats payload.

Internal breakdown
------------------
_overview(df)               ->  dict   shape, dtypes, nulls, duplicates
_numeric_summary(df, cols)  ->  dict   per-column descriptive stats + outliers
_categorical_summary(df, cols) -> dict per-column frequency tables + entropy
_datetime_summary(df, cols) ->  dict   range, freq inference
_boolean_summary(df, cols)  ->  dict   true/false counts and rates
_text_summary(df, cols)     ->  dict   length stats for free-text columns
_dataset_level(df, num_cols) -> dict   skew ranking, zero-variance flags, high-null flags
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from auto_insights.utils import (
    ColType,
    classify_columns,
    count_outliers,
    dataframe_metadata,
    get_logger,
    iqr_bounds,
    safe_numeric_cols,
    split_columns_by_type,
    validate_dataframe,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_HIGH_NULL_THRESHOLD   = 0.20   # flag columns with >20% missing
_HIGH_SKEW_THRESHOLD   = 1.0    # |skew| above this is flagged
_FREQ_TABLE_MAX_ROWS   = 20     # max categories shown in frequency tables
_TEXT_SAMPLE_MAX       = 5      # example values shown for text columns


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute a full descriptive profile of *df*.

    Called by InsightsGenerator.run() in core.py.

    Returns
    -------
    dict with keys:
        "overview"      – shape, memory, null/duplicate summary
        "numeric"       – per-column stats for numeric columns
        "categorical"   – frequency tables for categorical/boolean columns
        "datetime"      – range and inferred frequency for datetime columns
        "text"          – length distribution for free-text columns
        "dataset_level" – cross-column flags (high null, zero variance, skew)
        "metadata"      – raw metadata snapshot (from utils)
    """
    validate_dataframe(df)

    groups   = split_columns_by_type(df)
    num_cols  = groups.get(ColType.NUMERIC,     [])
    cat_cols  = groups.get(ColType.CATEGORICAL, [])
    dt_cols   = groups.get(ColType.DATETIME,    [])
    bool_cols = groups.get(ColType.BOOLEAN,     [])
    text_cols = groups.get(ColType.TEXT,        [])

    logger.info(
        "Profiling DataFrame: %d rows × %d cols  "
        "(numeric=%d, categorical=%d, datetime=%d, boolean=%d, text=%d)",
        len(df), len(df.columns),
        len(num_cols), len(cat_cols), len(dt_cols), len(bool_cols), len(text_cols),
    )

    result: dict[str, Any] = {
        "overview"      : _overview(df, groups),
        "numeric"       : _numeric_summary(df, num_cols),
        "categorical"   : _categorical_summary(df, cat_cols + bool_cols),
        "datetime"      : _datetime_summary(df, dt_cols),
        "text"          : _text_summary(df, text_cols),
        "dataset_level" : _dataset_level(df, num_cols),
        "metadata"      : dataframe_metadata(df),
    }

    logger.info("Profiling complete.")
    return result


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def _overview(df: pd.DataFrame, groups: dict[str, list[str]]) -> dict:
    null_counts  = df.isnull().sum()
    total_cells  = df.size
    total_nulls  = int(null_counts.sum())

    return {
        "n_rows"            : len(df),
        "n_cols"            : len(df.columns),
        "total_cells"       : total_cells,
        "total_nulls"       : total_nulls,
        "null_pct_overall"  : round(total_nulls / total_cells * 100, 2) if total_cells else 0,
        "duplicate_rows"    : int(df.duplicated().sum()),
        "column_type_counts": {ctype: len(cols) for ctype, cols in groups.items() if cols},
        "columns_with_nulls": {
            col: {"count": int(null_counts[col]), "pct": round(null_counts[col] / len(df) * 100, 2)}
            for col in df.columns if null_counts[col] > 0
        },
    }


# ---------------------------------------------------------------------------
# Numeric summary
# ---------------------------------------------------------------------------

def _numeric_summary(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Per-column descriptive stats for numeric columns.

    For each column:
        count, mean, std, min, 25/50/75%, max
        skewness, kurtosis
        IQR, outlier count and pct
        coefficient of variation (CV)
        zero count
        is_integer flag (all non-null values are whole numbers)
    """
    if not cols:
        return {}

    result: dict[str, dict] = {}

    for col in cols:
        s = df[col].dropna()

        if len(s) < 2:
            result[col] = {"note": "insufficient non-null values"}
            continue

        q1    = float(s.quantile(0.25))
        q3    = float(s.quantile(0.75))
        iqr   = q3 - q1
        lower, upper = iqr_bounds(s)
        n_out = count_outliers(s)
        mean  = float(s.mean())
        std   = float(s.std())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skew = float(scipy_stats.skew(s))
            kurt = float(scipy_stats.kurtosis(s))   # excess kurtosis (normal=0)

        result[col] = {
            "count"       : int(s.count()),
            "null_count"  : int(df[col].isnull().sum()),
            "mean"        : round(mean, 6),
            "std"         : round(std, 6),
            "min"         : round(float(s.min()), 6),
            "p25"         : round(q1, 6),
            "median"      : round(float(s.median()), 6),
            "p75"         : round(q3, 6),
            "max"         : round(float(s.max()), 6),
            "range"       : round(float(s.max() - s.min()), 6),
            "iqr"         : round(iqr, 6),
            "skewness"    : round(skew, 4),
            "kurtosis"    : round(kurt, 4),
            "cv"          : round(std / mean, 4) if mean != 0 else None,
            "outlier_count": n_out,
            "outlier_pct" : round(n_out / len(s) * 100, 2),
            "iqr_lower_fence": round(lower, 6),
            "iqr_upper_fence": round(upper, 6),
            "zero_count"  : int((s == 0).sum()),
            "is_integer"  : bool((s % 1 == 0).all()),
            "is_constant" : bool(s.nunique() == 1),
        }

    return result


# ---------------------------------------------------------------------------
# Categorical summary
# ---------------------------------------------------------------------------

def _categorical_summary(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Frequency table and entropy for each categorical/boolean column.

    Entropy uses base-2 (bits). A uniform distribution over k categories
    has entropy log2(k). Higher entropy = more evenly spread.

    Includes:
        n_unique, mode, mode_freq, mode_pct
        top-N frequency table
        Shannon entropy
        null count
    """
    if not cols:
        return {}

    result: dict[str, dict] = {}

    for col in cols:
        s        = df[col].dropna()
        n        = len(s)
        vc       = s.value_counts(dropna=True)
        n_unique = int(s.nunique())

        # Shannon entropy
        probs   = vc / n
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        freq_table = [
            {
                "value": val,
                "count": int(cnt),
                "pct"  : round(cnt / n * 100, 2),
            }
            for val, cnt in vc.head(_FREQ_TABLE_MAX_ROWS).items()
        ]

        result[col] = {
            "count"       : n,
            "null_count"  : int(df[col].isnull().sum()),
            "n_unique"    : n_unique,
            "mode"        : vc.index[0] if n_unique > 0 else None,
            "mode_count"  : int(vc.iloc[0]) if n_unique > 0 else 0,
            "mode_pct"    : round(vc.iloc[0] / n * 100, 2) if n_unique > 0 and n > 0 else 0,
            "entropy_bits": round(entropy, 4),
            "is_binary"   : n_unique == 2,
            "freq_table"  : freq_table,
        }

    return result


# ---------------------------------------------------------------------------
# Datetime summary
# ---------------------------------------------------------------------------

def _datetime_summary(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Basic temporal profile for datetime columns.

    Infers frequency via pd.infer_freq on sorted, deduplicated values.
    Falls back gracefully if inference fails (irregular series).
    """
    if not cols:
        return {}

    result: dict[str, dict] = {}

    for col in cols:
        s = df[col].dropna().sort_values()

        if len(s) < 2:
            result[col] = {"note": "insufficient non-null values"}
            continue

        inferred_freq = None
        try:
            inferred_freq = pd.infer_freq(s.drop_duplicates())
        except Exception:
            pass

        deltas    = s.diff().dropna()
        median_gap = deltas.median()

        result[col] = {
            "count"         : int(s.count()),
            "null_count"    : int(df[col].isnull().sum()),
            "min"           : str(s.min()),
            "max"           : str(s.max()),
            "range_days"    : round((s.max() - s.min()).days, 1),
            "inferred_freq" : inferred_freq,
            "median_gap"    : str(median_gap),
            "n_unique"      : int(s.nunique()),
            "has_duplicates": int(s.duplicated().sum()) > 0,
        }

    return result


# ---------------------------------------------------------------------------
# Boolean summary
# ---------------------------------------------------------------------------
# Note: booleans are merged into categorical above (True/False are categories),
# but this helper can be called separately if you want explicit bool stats.

def _boolean_summary(df: pd.DataFrame, cols: list[str]) -> dict:
    if not cols:
        return {}

    result: dict[str, dict] = {}
    for col in cols:
        s       = df[col].dropna()
        n       = len(s)
        n_true  = int(s.sum())
        n_false = n - n_true

        result[col] = {
            "count"     : n,
            "null_count": int(df[col].isnull().sum()),
            "true_count": n_true,
            "false_count": n_false,
            "true_pct"  : round(n_true  / n * 100, 2) if n else 0,
            "false_pct" : round(n_false / n * 100, 2) if n else 0,
        }

    return result


# ---------------------------------------------------------------------------
# Text column summary
# ---------------------------------------------------------------------------

def _text_summary(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Length distribution and sample values for high-cardinality string columns.
    These columns are not suitable for frequency tables (too many unique values)
    but length statistics are still informative.
    """
    if not cols:
        return {}

    result: dict[str, dict] = {}

    for col in cols:
        s      = df[col].dropna().astype(str)
        lengths = s.str.len()

        result[col] = {
            "count"         : int(s.count()),
            "null_count"    : int(df[col].isnull().sum()),
            "n_unique"      : int(s.nunique()),
            "unique_pct"    : round(s.nunique() / len(s) * 100, 2) if len(s) else 0,
            "length_mean"   : round(float(lengths.mean()), 2),
            "length_std"    : round(float(lengths.std()), 2),
            "length_min"    : int(lengths.min()),
            "length_max"    : int(lengths.max()),
            "length_median" : int(lengths.median()),
            "sample_values" : s.head(_TEXT_SAMPLE_MAX).tolist(),
        }

    return result


# ---------------------------------------------------------------------------
# Dataset-level flags (cross-column)
# ---------------------------------------------------------------------------

def _dataset_level(df: pd.DataFrame, num_cols: list[str]) -> dict:
    """
    Cross-column diagnostics that apply to the dataset as a whole.

    Flags
    -----
    high_null_cols   – columns with > 20% missing
    zero_variance    – numeric columns where std == 0 (constant)
    high_skew        – numeric columns with |skewness| > 1.0
    perfect_corr     – pairs of numeric columns with |r| == 1.0 (exact duplicates)
    """
    null_pct     = df.isnull().mean()
    high_null    = null_pct[null_pct > _HIGH_NULL_THRESHOLD].index.tolist()

    zero_var: list[str] = []
    high_skew: list[dict] = []

    safe_cols = safe_numeric_cols(df)

    for col in safe_cols:
        s = df[col].dropna()
        if float(s.std()) == 0.0:
            zero_var.append(col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = float(scipy_stats.skew(s))
        if abs(sk) > _HIGH_SKEW_THRESHOLD:
            high_skew.append({"column": col, "skewness": round(sk, 4)})

    # Sort by absolute skewness descending
    high_skew.sort(key=lambda x: abs(x["skewness"]), reverse=True)

    # Perfect correlation pairs (only among safe numeric cols, limit computation)
    perfect_corr_pairs: list[dict] = []
    if len(safe_cols) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = df[safe_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        pairs = list(zip(*np.where(upper.values == 1.0)))
        for r, c in pairs:
            perfect_corr_pairs.append({
                "col_a": safe_cols[r],
                "col_b": safe_cols[c],
            })

    return {
        "high_null_cols"    : high_null,
        "zero_variance_cols": zero_var,
        "high_skew_cols"    : high_skew,
        "perfect_corr_pairs": perfect_corr_pairs,
    }
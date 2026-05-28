"""
auto_insights_generator/utils.py

Shared helpers used across all modules:
  - Column type classification
  - Figure serialization (matplotlib -> base64)
  - DataFrame validation and metadata extraction
  - LLM prompt truncation / summarization helpers
  - Logging setup
"""

from __future__ import annotations

import base64
import io
import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure



def get_logger(name: str = "auto_insights_generator") -> logging.Logger:
    """
    Return a module-level logger with a sensible default format.
    Call once at the top of each module:
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)




# Thresholds used to decide whether a low-cardinality numeric column should
# be treated as categorical (e.g. binary flags, coded groups).
_CATEGORICAL_CARDINALITY_RATIO = 0.05   # unique / total
_CATEGORICAL_CARDINALITY_MAX   = 20     # hard cap on unique values


class ColType:
    """String constants for column type labels."""
    NUMERIC     = "numeric"
    CATEGORICAL = "categorical"
    DATETIME    = "datetime"
    BOOLEAN     = "boolean"
    TEXT        = "text"        # high-cardinality string columns
    UNKNOWN     = "unknown"


def classify_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Classify every column in *df* into one of the ColType categories.

    Returns
    -------
    dict mapping column name -> ColType string.

    Logic
    -----
    1. bool dtype               -> BOOLEAN
    2. datetime dtype           -> DATETIME
    3. numeric dtype
       a. <=20 unique values and unique/total <= 5%  -> CATEGORICAL
       b. otherwise                                  -> NUMERIC
    4. object / string dtype
       a. pd.api reports categorical                 -> CATEGORICAL
       b. unique/total <= 5% and <=20 unique vals    -> CATEGORICAL
       c. otherwise                                  -> TEXT
    5. everything else          -> UNKNOWN
    """
    col_types: dict[str, str] = {}

    for col in df.columns:
        series = df[col]
        dtype  = series.dtype
        n      = len(series)
        n_unique = series.nunique(dropna=True)
        cardinality_ratio = n_unique / n if n > 0 else 0

        if pd.api.types.is_bool_dtype(dtype):
            col_types[col] = ColType.BOOLEAN

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_types[col] = ColType.DATETIME

        elif pd.api.types.is_numeric_dtype(dtype):
            if (
                n_unique <= _CATEGORICAL_CARDINALITY_MAX
                and cardinality_ratio <= _CATEGORICAL_CARDINALITY_RATIO
            ):
                col_types[col] = ColType.CATEGORICAL
            else:
                col_types[col] = ColType.NUMERIC

        elif pd.api.types.is_categorical_dtype(dtype):
            col_types[col] = ColType.CATEGORICAL

        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            if (
                n_unique <= _CATEGORICAL_CARDINALITY_MAX
                and cardinality_ratio <= _CATEGORICAL_CARDINALITY_RATIO
            ):
                col_types[col] = ColType.CATEGORICAL
            else:
                col_types[col] = ColType.TEXT

        else:
            col_types[col] = ColType.UNKNOWN

    return col_types


def split_columns_by_type(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Convenience wrapper around classify_columns.

    Returns
    -------
    dict with keys matching ColType constants, each mapping to a list of
    column names of that type.

    Example
    -------
    >>> groups = split_columns_by_type(df)
    >>> numeric_cols = groups[ColType.NUMERIC]
    """
    col_types = classify_columns(df)
    groups: dict[str, list[str]] = {t: [] for t in vars(ColType).values()
                                    if isinstance(t, str) and not t.startswith("_")}
    for col, ctype in col_types.items():
        groups.setdefault(ctype, []).append(col)
    return groups


# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------

class DataFrameValidationError(ValueError):
    """Raised when the input DataFrame fails basic sanity checks."""


def validate_dataframe(df: pd.DataFrame, min_rows: int = 2, min_cols: int = 1) -> None:
    """
    Run basic sanity checks on the input DataFrame.

    Raises DataFrameValidationError on failure so callers can catch it
    specifically without swallowing all ValueErrors.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataFrameValidationError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if df.empty:
        raise DataFrameValidationError("DataFrame is empty.")
    if len(df) < min_rows:
        raise DataFrameValidationError(
            f"DataFrame has only {len(df)} row(s); need at least {min_rows}."
        )
    if len(df.columns) < min_cols:
        raise DataFrameValidationError(
            f"DataFrame has only {len(df.columns)} column(s); need at least {min_cols}."
        )
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise DataFrameValidationError(
            f"DataFrame has duplicate column names: {duplicate_cols}. "
            "Rename them before running auto_insights_generator."
        )


# ---------------------------------------------------------------------------
# DataFrame metadata snapshot
# ---------------------------------------------------------------------------

def dataframe_metadata(df: pd.DataFrame) -> dict:
    """
    Return a lightweight metadata snapshot of the DataFrame.

    This is passed verbatim into LLM prompts so Claude has full context
    about the shape and quality of the data without seeing raw rows.

    Keys
    ----
    n_rows, n_cols, col_types, null_counts, null_pct,
    memory_mb, duplicate_rows
    """
    col_types  = classify_columns(df)
    null_counts = df.isnull().sum().to_dict()
    null_pct    = {
        col: round(count / len(df) * 100, 2)
        for col, count in null_counts.items()
        if count > 0
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    return {
        "n_rows"        : len(df),
        "n_cols"        : len(df.columns),
        "col_types"     : col_types,
        "null_counts"   : {c: v for c, v in null_counts.items() if v > 0},
        "null_pct"      : null_pct,
        "memory_mb"     : round(mem_mb, 3),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Figure serialization
# ---------------------------------------------------------------------------

def fig_to_base64(fig: "matplotlib.figure.Figure", fmt: str = "png", dpi: int = 120) -> str:
    """
    Serialize a matplotlib Figure to a base64-encoded string suitable for
    embedding directly in an HTML <img> src attribute.

    Parameters
    ----------
    fig : matplotlib Figure
    fmt : image format passed to savefig ("png" or "svg")
    dpi : dots per inch for raster formats

    Returns
    -------
    str — "data:image/png;base64,<data>" ready to drop into HTML.

    Example
    -------
    >>> src = fig_to_base64(fig)
    >>> html = f'<img src="{src}" />'
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    mime = "image/svg+xml" if fmt == "svg" else f"image/{fmt}"
    return f"data:{mime};base64,{encoded}"


def figs_to_base64_dict(figs: dict[str, "matplotlib.figure.Figure"]) -> dict[str, str]:
    """
    Batch-convert a dict of {label: Figure} to {label: base64 src string}.

    Use this at the end of viz.py / corr.py to package all figures for
    the report in one call.
    """
    return {label: fig_to_base64(fig) for label, fig in figs.items()}


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

# Maximum number of characters sent to the LLM for any single numeric summary.
# Keeps token usage predictable regardless of DataFrame width.
_LLM_STATS_CHAR_LIMIT = 6_000


def stats_to_llm_str(stats_dict: dict, char_limit: int = _LLM_STATS_CHAR_LIMIT) -> str:
    """
    Convert the aggregated stats dict to a compact, human-readable string
    suitable for inclusion in an LLM prompt.

    - Rounds floats to 4 significant figures to reduce noise.
    - Truncates the output if it exceeds char_limit to avoid blowing the
      context window on very wide DataFrames.

    Parameters
    ----------
    stats_dict : dict produced by stats.py, corr.py, tests.py
    char_limit : soft cap on output length in characters

    Returns
    -------
    Formatted string.
    """
    lines: list[str] = []

    def _fmt_value(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, (np.floating, np.integer)):
            return f"{v:.4g}"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {_fmt_value(val)}" for k, val in v.items()) + "}"
        if isinstance(v, list):
            preview = v[:10]
            suffix  = f"... (+{len(v)-10} more)" if len(v) > 10 else ""
            return "[" + ", ".join(_fmt_value(i) for i in preview) + suffix + "]"
        return str(v)

    def _walk(d: dict, indent: int = 0):
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}{k}:")
                _walk(v, indent + 1)
            else:
                lines.append(f"{prefix}{k}: {_fmt_value(v)}")

    _walk(stats_dict)
    result = "\n".join(lines)

    if len(result) > char_limit:
        result = result[:char_limit] + f"\n\n[... truncated at {char_limit} chars]"
        logger.warning(
            "stats_to_llm_str: output truncated to %d chars. "
            "Consider narrowing the DataFrame before analysis.", char_limit
        )

    return result


def df_sample_for_llm(df: pd.DataFrame, n_rows: int = 5) -> str:
    """
    Return a compact string representation of the first *n_rows* rows of *df*
    for inclusion in an LLM prompt.

    Useful for giving Claude concrete examples of data values so it can
    make more grounded observations (e.g. noticing a column holds pitch types
    vs. generic "string data").
    """
    sample = df.head(n_rows).to_string(index=False, max_cols=20)
    if len(df.columns) > 20:
        sample += f"\n[... {len(df.columns) - 20} more columns not shown]"
    return sample


# ---------------------------------------------------------------------------
# Miscellaneous numeric helpers
# ---------------------------------------------------------------------------

def safe_numeric_cols(df: pd.DataFrame) -> list[str]:
    """
    Return column names that are strictly numeric (int or float) and have
    at least 2 non-null values — the minimum needed for any meaningful stat.
    """
    return [
        col for col in df.select_dtypes(include=[np.number]).columns
        if df[col].notna().sum() >= 2
    ]


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """
    Return the standard IQR-based outlier fences (lower, upper).

    Parameters
    ----------
    series : numeric Series (NaNs are ignored)
    k      : multiplier applied to IQR (default 1.5; use 3.0 for extreme outliers)
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def count_outliers(series: pd.Series, k: float = 1.5) -> int:
    """
    Count values in *series* that fall outside the IQR fences.
    NaNs are excluded from both the fence calculation and the count.
    """
    lower, upper = iqr_bounds(series.dropna(), k)
    return int(((series < lower) | (series > upper)).sum())
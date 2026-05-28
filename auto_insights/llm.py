"""
auto_insights/llm.py

Anthropic API integration module.

Public API
----------
generate_insights(analysis_result, df_meta, model, max_tokens)  ->  InsightsNarrative
    Top-level entry point called by core.py.
    Sends the structured analysis payload to Claude and returns a
    structured narrative broken into labelled sections.

Design
------
Rather than one giant prompt, the analysis is split into focused calls —
one per section — so each response is grounded in the most relevant subset
of results. This gives more precise, less diluted commentary than pasting
everything into a single prompt.

Sections (in order):
    1. overview       — shape, nulls, duplicates, column type summary
    2. distributions  — per-column normality, skew, outlier narrative
    3. correlations   — strongest relationships and divergences
    4. group_tests    — ANOVA / KW / t-test results narrative
    5. associations   — chi-square categorical association narrative
    6. recommendations— actionable next steps based on all findings

Each section prompt is self-contained: it includes only the data
relevant to that section plus a shared context header (n_rows, n_cols,
column names) so Claude always knows what dataset it's looking at.

The final InsightsNarrative dataclass bundles all six sections and
exposes a .to_dict() method so report.py can embed them directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import anthropic

from .utils import get_logger, stats_to_llm_str

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL      = "claude-sonnet-4-20250514"
_DEFAULT_MAX_TOKENS = 1024
_RETRY_ATTEMPTS     = 3
_RETRY_BACKOFF      = 2.0    # seconds; doubled on each retry

_SYSTEM_PROMPT = """\
You are a senior data scientist writing an automated EDA (exploratory data \
analysis) report. Your audience is a technically literate analyst who wants \
concise, actionable insight — not a statistics lecture.

Rules:
- Be specific: reference actual column names, numbers, and test results.
- Be concise: 3–6 sentences per section unless the findings are rich enough \
to justify more. No padding.
- Prioritise anomalies and actionable findings over routine observations.
- When a statistical test result is provided, interpret the practical \
significance (effect size), not just whether p < 0.05.
- Do not repeat the raw numbers verbatim — paraphrase and interpret them.
- Write in plain prose. No bullet lists, no headers, no markdown formatting \
inside your response.
- If a section has nothing notable to report, say so in one sentence.
"""


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class InsightsNarrative:
    """
    Structured container for all LLM-generated narrative sections.

    Each field is a plain prose string written by Claude.
    Empty string means the section was skipped (insufficient data).
    """
    overview        : str = ""
    distributions   : str = ""
    correlations    : str = ""
    group_tests     : str = ""
    associations    : str = ""
    recommendations : str = ""
    model           : str = ""
    total_tokens    : int  = 0
    errors          : list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def full_text(self) -> str:
        """Concatenate all sections into a single readable string."""
        parts = []
        labels = [
            ("Dataset Overview",          self.overview),
            ("Distributions & Outliers",  self.distributions),
            ("Correlations",              self.correlations),
            ("Group Comparisons",         self.group_tests),
            ("Categorical Associations",  self.associations),
            ("Recommendations",           self.recommendations),
        ]
        for label, text in labels:
            if text:
                parts.append(f"### {label}\n{text}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_insights(
    analysis_result : dict[str, Any],
    df_meta         : dict[str, Any] | None = None,
    model           : str = _DEFAULT_MODEL,
    max_tokens      : int = _DEFAULT_MAX_TOKENS,
) -> InsightsNarrative:
    """
    Send structured analysis results to Claude and return narrative insights.

    Parameters
    ----------
    analysis_result : full output dict from core.py's _run_analysis(),
                      containing keys: stats, correlations, tests.
    df_meta         : lightweight metadata dict from utils.dataframe_metadata().
                      Used to build the shared context header for every prompt.
                      Falls back to analysis_result["stats"]["metadata"] if None.
    model           : Anthropic model string.
    max_tokens      : max tokens per API call (applied to each section call).

    Returns
    -------
    InsightsNarrative dataclass with one field per report section.
    """
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment

    meta = df_meta or analysis_result.get("stats", {}).get("metadata", {})
    ctx  = _build_context_header(meta)

    narrative = InsightsNarrative(model=model)
    total_tokens = 0

    sections: list[tuple[str, str]] = [
        ("overview",        _prompt_overview(ctx, analysis_result)),
        ("distributions",   _prompt_distributions(ctx, analysis_result)),
        ("correlations",    _prompt_correlations(ctx, analysis_result)),
        ("group_tests",     _prompt_group_tests(ctx, analysis_result)),
        ("associations",    _prompt_associations(ctx, analysis_result)),
        ("recommendations", _prompt_recommendations(ctx, analysis_result)),
    ]

    for section_name, user_prompt in sections:
        if not user_prompt.strip():
            logger.info("llm: skipping section '%s' — no data.", section_name)
            continue

        logger.info("llm: generating '%s' section...", section_name)
        text, tokens, error = _call_api(
            client, model, max_tokens, user_prompt
        )

        if error:
            logger.warning("llm: section '%s' failed — %s", section_name, error)
            narrative.errors.append(f"{section_name}: {error}")
        else:
            setattr(narrative, section_name, text)
            total_tokens += tokens
            logger.info(
                "llm: '%s' done (%d tokens).", section_name, tokens
            )

    narrative.total_tokens = total_tokens
    logger.info("llm: all sections complete. Total tokens used: %d.", total_tokens)
    return narrative


# ---------------------------------------------------------------------------
# API call wrapper with retry
# ---------------------------------------------------------------------------

def _call_api(
    client     : anthropic.Anthropic,
    model      : str,
    max_tokens : int,
    user_prompt: str,
) -> tuple[str, int, str | None]:
    """
    Call the Anthropic messages API with exponential-backoff retry.

    Returns
    -------
    (response_text, tokens_used, error_string_or_None)
    """
    backoff = _RETRY_BACKOFF

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            response = client.messages.create(
                model      = model,
                max_tokens = max_tokens,
                system     = _SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_prompt}],
            )
            text   = response.content[0].text.strip()
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens, None

        except anthropic.RateLimitError:
            if attempt < _RETRY_ATTEMPTS:
                logger.warning(
                    "llm: rate limit hit (attempt %d/%d) — retrying in %.1fs.",
                    attempt, _RETRY_ATTEMPTS, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                return "", 0, "Rate limit exceeded after retries."

        except anthropic.APIStatusError as exc:
            return "", 0, f"API error {exc.status_code}: {exc.message}"

        except Exception as exc:
            return "", 0, f"Unexpected error: {exc}"

    return "", 0, "All retry attempts exhausted."


# ---------------------------------------------------------------------------
# Shared context header
# ---------------------------------------------------------------------------

def _build_context_header(meta: dict) -> str:
    """
    Build a short shared preamble injected at the top of every section prompt.

    Gives Claude consistent orientation without repeating full stats in
    every single prompt.
    """
    n_rows  = meta.get("n_rows", "unknown")
    n_cols  = meta.get("n_cols", "unknown")
    mem_mb  = meta.get("memory_mb", "?")
    col_types = meta.get("col_types", {})

    type_summary = {}
    for col, ctype in col_types.items():
        type_summary.setdefault(ctype, []).append(col)

    type_lines = "\n".join(
        f"  {ctype} ({len(cols)}): {', '.join(cols[:8])}"
        + (" ..." if len(cols) > 8 else "")
        for ctype, cols in type_summary.items()
    )

    return (
        f"Dataset: {n_rows} rows × {n_cols} columns ({mem_mb} MB)\n"
        f"Column types:\n{type_lines}\n"
    )


# ---------------------------------------------------------------------------
# Section prompts
# ---------------------------------------------------------------------------

def _prompt_overview(ctx: str, result: dict) -> str:
    stats    = result.get("stats", {})
    overview = stats.get("overview", {})
    ds_level = stats.get("dataset_level", {})

    if not overview:
        return ""

    payload = {
        "overview"      : overview,
        "dataset_flags" : ds_level,
    }

    return (
        f"{ctx}\n"
        "Write a concise overview of this dataset. Cover: overall shape and "
        "memory footprint, missing data patterns (which columns, how severe), "
        "duplicate rows, the mix of column types, and any dataset-level flags "
        "(zero-variance columns, heavily skewed columns, perfectly correlated pairs). "
        "Flag anything that would require cleaning before modelling.\n\n"
        f"Analysis results:\n{stats_to_llm_str(payload, char_limit=3000)}"
    )


def _prompt_distributions(ctx: str, result: dict) -> str:
    stats     = result.get("stats", {})
    numeric   = stats.get("numeric", {})
    norm_res  = result.get("tests", {}).get("normality", {})
    ds_level  = stats.get("dataset_level", {})

    if not numeric:
        return ""

    # Merge normality verdict into each column's stats for a single payload
    merged: dict[str, Any] = {}
    for col, col_stats in numeric.items():
        entry = dict(col_stats)
        if col in norm_res:
            consensus = norm_res[col].get("consensus", {})
            entry["normality_verdict"] = consensus.get("verdict", "unknown")
        merged[col] = entry

    payload = {
        "numeric_columns" : merged,
        "high_skew_flags" : ds_level.get("high_skew_cols", []),
    }

    return (
        f"{ctx}\n"
        "Describe the distributions of the numeric columns. For each notable "
        "column comment on: shape (skew, kurtosis), the normality verdict, "
        "outlier counts, and the range vs IQR spread. Group similar columns "
        "together where possible rather than listing each one individually. "
        "Highlight any columns with extreme skew or unusually high outlier rates "
        "that would need transformation before modelling.\n\n"
        f"Analysis results:\n{stats_to_llm_str(payload, char_limit=4000)}"
    )


def _prompt_correlations(ctx: str, result: dict) -> str:
    corr = result.get("correlations", {})

    if not corr or "note" in corr:
        return ""

    payload = {
        "top_pairs_pearson"  : corr.get("top_pairs_pearson",  [])[:12],
        "top_pairs_spearman" : corr.get("top_pairs_spearman", [])[:12],
        "point_biserial"     : corr.get("point_biserial",     [])[:8],
        "flags"              : corr.get("flags",               []),
    }

    return (
        f"{ctx}\n"
        "Interpret the correlation structure of this dataset. Focus on: "
        "the strongest positive and negative Pearson correlations, any "
        "cases where Pearson and Spearman diverge significantly (possible "
        "nonlinearity or outlier influence), and notable point-biserial "
        "relationships between binary and numeric columns. Comment on "
        "multicollinearity risks for any modelling use case.\n\n"
        f"Analysis results:\n{stats_to_llm_str(payload, char_limit=3500)}"
    )


def _prompt_group_tests(ctx: str, result: dict) -> str:
    tests      = result.get("tests", {})
    group_comp = tests.get("group_comparisons", [])

    if not group_comp:
        return ""

    # Only include pairs with at least one significant result
    significant = [
        gc for gc in group_comp
        if (
            gc.get("kruskal_wallis", {}).get("significant")
            or gc.get("anova",        {}).get("significant")
            or gc.get("welch_t",      {}).get("significant")
        )
    ]

    payload = {
        "significant_pairs"  : significant[:10],
        "non_significant_n"  : len(group_comp) - len(significant),
        "test_flags"         : tests.get("flags", []),
    }

    if not significant:
        return (
            f"{ctx}\n"
            "No statistically significant group differences were found in any "
            "numeric-vs-categorical pair. Write one sentence confirming this.\n\n"
            f"Number of pairs tested: {len(group_comp)}"
        )

    return (
        f"{ctx}\n"
        "Interpret the group comparison test results. For each significant "
        "numeric ~ categorical pair, describe: which test was used and why "
        "(ANOVA vs Kruskal-Wallis), the direction and magnitude of the effect "
        "(use effect size, not just p-value), and what this means practically. "
        "Mention if ANOVA and Kruskal-Wallis disagree on significance.\n\n"
        f"Analysis results:\n{stats_to_llm_str(payload, char_limit=3500)}"
    )


def _prompt_associations(ctx: str, result: dict) -> str:
    tests        = result.get("tests", {})
    associations = tests.get("associations", [])

    if not associations:
        return ""

    significant = [a for a in associations if a.get("results", {}).get("significant")]

    payload = {
        "significant_associations": significant[:8],
        "total_pairs_tested"      : len(associations),
        "non_significant_n"       : len(associations) - len(significant),
    }

    if not significant:
        return (
            f"{ctx}\n"
            "No statistically significant associations were found between any "
            "categorical column pairs. Write one sentence confirming this.\n\n"
            f"Total pairs tested: {len(associations)}"
        )

    return (
        f"{ctx}\n"
        "Interpret the categorical association results. For each significant "
        "chi-square pair, describe the strength of association using Cramér's V "
        "(not just p-value), and comment on what the association might mean "
        "practically for analysis or modelling. Flag any results where the "
        "chi-square validity was questionable (low expected cell counts).\n\n"
        f"Analysis results:\n{stats_to_llm_str(payload, char_limit=2500)}"
    )


def _prompt_recommendations(ctx: str, result: dict) -> str:
    """
    The recommendations prompt is the only one that receives the full
    flags list from all modules — it synthesises everything into
    prioritised next steps.
    """
    stats_flags = result.get("stats",        {}).get("dataset_level", {})
    corr_flags  = result.get("correlations", {}).get("flags", [])
    test_flags  = result.get("tests",        {}).get("flags", [])

    all_flags = {
        "data_quality"  : {
            "high_null_cols"    : stats_flags.get("high_null_cols",     []),
            "zero_variance_cols": stats_flags.get("zero_variance_cols", []),
            "high_skew_cols"    : [
                h["column"] for h in stats_flags.get("high_skew_cols", [])
            ],
            "perfect_corr_pairs": stats_flags.get("perfect_corr_pairs", []),
        },
        "correlation_flags": corr_flags,
        "test_flags"       : test_flags,
    }

    return (
        f"{ctx}\n"
        "Based on the full analysis, write a prioritised list of 4–6 concrete "
        "recommendations for the analyst. These should cover: data cleaning "
        "steps (imputation strategy, duplicate handling, outlier treatment), "
        "feature engineering suggestions (transformations for skewed columns, "
        "encoding for categoricals), modelling considerations (multicollinearity "
        "risks, which tests confirm group separability, which variables are "
        "likely informative), and any additional analyses worth running. "
        "Be specific — reference actual column names.\n\n"
        f"All findings:\n{stats_to_llm_str(all_flags, char_limit=3000)}"
    )
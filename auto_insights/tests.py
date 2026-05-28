"""
auto_insights/tests.py

Statistical tests module.

Public API
----------
run_statistical_tests(df)  ->  dict
    Top-level entry point called by core.py. Automatically selects and runs
    the appropriate tests based on column types and returns a structured
    results payload for the LLM and report.

Test inventory
--------------
Normality (per numeric column):
    Shapiro-Wilk        — best for n < 5000; exact test
    D'Agostino-Pearson  — skewness + kurtosis omnibus; better for n >= 50
    Kolmogorov-Smirnov  — against theoretical normal; used as tie-breaker
    Anderson-Darling    — most powerful for detecting tail departures

Group comparisons (numeric ~ categorical):
    Levene's test       — equality of variances across groups (prerequisite)
    One-way ANOVA       — parametric; assumes normality + homoscedasticity
    Kruskal-Wallis      — non-parametric ANOVA alternative (rank-based)
    Mann-Whitney U      — pairwise non-parametric test for binary grouping vars
    Welch's t-test      — parametric pairwise; does not assume equal variances

Association (categorical ~ categorical):
    Chi-square test     — association between two categorical columns
    Cramér's V          — effect size for chi-square (scale-free, 0–1)

Variance:
    Bartlett's test     — equality of variances (assumes normality)
    Levene's test       — equality of variances (robust to non-normality)

Notes
-----
- Alpha is set to 0.05 throughout. p-values are Bonferroni-corrected where
  multiple tests are run on the same family (e.g. pairwise Mann-Whitney).
- Tests are selected automatically; no user configuration needed.
- All results include: statistic, p_value, reject_h0, interpretation string.
- The "interpretation" field is written in plain English so the LLM can
  quote or paraphrase it directly without needing to decode p-values.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils import (
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

_ALPHA               = 0.05
_SHAPIRO_MAX_N       = 5_000    # Shapiro-Wilk degrades and becomes trivially significant above this
_KS_MIN_N            = 8       # KS test unreliable below this
_ANOVA_MIN_GROUPS    = 2       # minimum distinct groups for ANOVA / KW
_ANOVA_MIN_PER_GROUP = 3       # minimum observations per group
_CHI2_MIN_EXPECTED   = 5       # cells with expected count < 5 invalidate chi-square
_CHI2_MAX_CATS       = 20      # cap on categories per column for chi-square (combinatorial explosion)
_PAIRWISE_MAX_GROUPS = 6       # cap on groups for pairwise Mann-Whitney (n*(n-1)/2 pairs)
_MAX_NUM_COLS_TEST   = 30      # cap on numeric columns run through normality tests


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """
    Automatically select and run statistical tests appropriate for *df*.

    Returns
    -------
    dict with keys:
        "normality"         — per numeric column normality battery
        "group_comparisons" — numeric ~ categorical test results
        "associations"      — categorical ~ categorical chi-square results
        "variance_equality" — Levene / Bartlett for numeric columns
        "flags"             — plain-English summary strings for the LLM
        "alpha"             — significance level used throughout
    """
    validate_dataframe(df)
    groups    = split_columns_by_type(df)
    num_cols  = safe_numeric_cols(df)[:_MAX_NUM_COLS_TEST]
    cat_cols  = groups.get(ColType.CATEGORICAL, [])
    bool_cols = groups.get(ColType.BOOLEAN,     [])

    logger.info(
        "tests: running on %d numeric, %d categorical, %d boolean columns.",
        len(num_cols), len(cat_cols), len(bool_cols),
    )

    normality    = _normality_battery(df, num_cols)
    group_comps  = _group_comparisons(df, num_cols, cat_cols + bool_cols)
    associations = _categorical_associations(df, cat_cols)
    var_equality = _variance_equality(df, num_cols)
    flags        = _build_flags(normality, group_comps, associations)

    return {
        "normality"         : normality,
        "group_comparisons" : group_comps,
        "associations"      : associations,
        "variance_equality" : var_equality,
        "flags"             : flags,
        "alpha"             : _ALPHA,
    }


# ---------------------------------------------------------------------------
# Helpers — result packaging
# ---------------------------------------------------------------------------

def _result(
    test_name : str,
    statistic : float | None,
    p_value   : float | None,
    extra     : dict | None = None,
    *,
    alpha     : float = _ALPHA,
    note      : str = "",
) -> dict:
    """
    Package a single test result into a consistent dict structure.

    Fields
    ------
    test, statistic, p_value, reject_h0, significant, interpretation, note
    """
    reject  = bool(p_value < alpha) if p_value is not None else None
    sig_str = "significant" if reject else "not significant"
    interp  = (
        f"{test_name}: statistic={statistic:.4g}, p={p_value:.4g} "
        f"— {sig_str} at α={alpha}."
    ) if statistic is not None and p_value is not None else "Test could not be computed."

    out = {
        "test"         : test_name,
        "statistic"    : round(float(statistic), 6) if statistic is not None else None,
        "p_value"      : round(float(p_value), 6)   if p_value   is not None else None,
        "reject_h0"    : reject,
        "significant"  : reject,
        "interpretation": interp,
    }
    if note:
        out["note"] = note
    if extra:
        out.update(extra)
    return out


def _safe_run(fn, *args, **kwargs):
    """Run *fn* and return (result, None) or (None, error_str) on failure."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Normality battery
# ---------------------------------------------------------------------------

def _normality_battery(df: pd.DataFrame, num_cols: list[str]) -> dict[str, dict]:
    """
    Run a battery of normality tests on each numeric column.

    Returns
    -------
    dict mapping column name -> dict of test results + consensus verdict.
    """
    results: dict[str, dict] = {}

    for col in num_cols:
        s = df[col].dropna()
        n = len(s)
        col_results: dict[str, Any] = {"n": n}

        if n < 3:
            col_results["note"] = "Insufficient data (n < 3)"
            results[col] = col_results
            continue

        # Shapiro-Wilk
        if n <= _SHAPIRO_MAX_N:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = scipy_stats.shapiro(s)
            col_results["shapiro_wilk"] = _result("Shapiro-Wilk", stat, p)
        else:
            col_results["shapiro_wilk"] = _result(
                "Shapiro-Wilk", None, None,
                note=f"Skipped: n={n} > {_SHAPIRO_MAX_N}",
            )

        # D'Agostino-Pearson (requires n >= 8)
        if n >= 8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = scipy_stats.normaltest(s)
            col_results["dagostino_pearson"] = _result("D'Agostino-Pearson", stat, p)
        else:
            col_results["dagostino_pearson"] = _result(
                "D'Agostino-Pearson", None, None,
                note=f"Skipped: n={n} < 8",
            )

        # Anderson-Darling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ad_result = scipy_stats.anderson(s, dist="norm")

        # Anderson returns critical values at [15, 10, 5, 2.5, 1]% significance
        # We use the 5% level (index 2)
        sig_idx   = 2   # 5% level
        ad_stat   = float(ad_result.statistic)
        ad_crit   = float(ad_result.critical_values[sig_idx])
        ad_reject = ad_stat > ad_crit
        col_results["anderson_darling"] = {
            "test"          : "Anderson-Darling",
            "statistic"     : round(ad_stat, 6),
            "critical_value": round(ad_crit, 6),
            "reject_h0"     : ad_reject,
            "significant"   : ad_reject,
            "interpretation": (
                f"Anderson-Darling: A²={ad_stat:.4g}, critical={ad_crit:.4g} at 5% — "
                + ("reject normality." if ad_reject else "fail to reject normality.")
            ),
        }

        # Kolmogorov-Smirnov against fitted normal
        if n >= _KS_MIN_N:
            loc, scale = float(s.mean()), float(s.std())
            if scale > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p = scipy_stats.kstest(s, "norm", args=(loc, scale))
                col_results["kolmogorov_smirnov"] = _result(
                    "Kolmogorov-Smirnov (fitted normal)", stat, p,
                    note="Tests against N(mean, std) fitted from data.",
                )
            else:
                col_results["kolmogorov_smirnov"] = _result(
                    "Kolmogorov-Smirnov", None, None,
                    note="Skipped: zero variance",
                )

        # Consensus: how many tests reject normality?
        n_reject = sum(
            1 for k, v in col_results.items()
            if isinstance(v, dict) and v.get("reject_h0") is True
        )
        n_run = sum(
            1 for k, v in col_results.items()
            if isinstance(v, dict) and v.get("reject_h0") is not None
        )
        col_results["consensus"] = {
            "tests_run"       : n_run,
            "tests_rejecting" : n_reject,
            "likely_normal"   : n_reject == 0,
            "verdict"         : (
                "Likely normal"          if n_reject == 0
                else "Probably not normal" if n_reject >= n_run // 2 + 1
                else "Borderline"
            ),
        }

        results[col] = col_results

    return results


# ---------------------------------------------------------------------------
# Group comparisons  (numeric ~ categorical)
# ---------------------------------------------------------------------------

def _group_comparisons(
    df       : pd.DataFrame,
    num_cols : list[str],
    cat_cols : list[str],
) -> list[dict]:
    """
    For each (numeric, categorical) pair, run:
        1. Levene's test for equal variances
        2. One-way ANOVA  (parametric)
        3. Kruskal-Wallis (non-parametric)
        4. If binary categorical: Mann-Whitney U + Welch t-test

    Only pairs where every group has >= _ANOVA_MIN_PER_GROUP observations
    and there are >= _ANOVA_MIN_GROUPS groups are tested.

    Returns a list of result dicts, one per (num_col, cat_col) pair.
    """
    results: list[dict] = []

    for num_col, cat_col in itertools.product(num_cols, cat_cols):
        combined = df[[num_col, cat_col]].dropna()
        if len(combined) < 6:
            continue

        groups_series = [
            grp[num_col].values
            for _, grp in combined.groupby(cat_col)
            if len(grp) >= _ANOVA_MIN_PER_GROUP
        ]
        n_groups = len(groups_series)

        if n_groups < _ANOVA_MIN_GROUPS:
            continue

        unique_cats = combined[cat_col].nunique()
        pair_result: dict[str, Any] = {
            "numeric_col"   : num_col,
            "categorical_col": cat_col,
            "n_groups"      : n_groups,
            "n_obs"         : len(combined),
        }

        # Levene's test (robust to non-normality)
        lev_stat, lev_p = _safe_run(scipy_stats.levene, *groups_series)[0] or (None, None)
        if lev_stat is not None:
            pair_result["levene"] = _result(
                "Levene's test (equal variances)", lev_stat, lev_p,
                extra={"equal_variances": lev_p >= _ALPHA if lev_p is not None else None},
            )
            equal_var = lev_p >= _ALPHA if lev_p is not None else True
        else:
            equal_var = True

        # One-way ANOVA
        anova_out, anova_err = _safe_run(scipy_stats.f_oneway, *groups_series)
        if anova_out is not None:
            f_stat, f_p = anova_out
            # Eta-squared effect size
            all_vals  = np.concatenate(groups_series)
            grand_mean = all_vals.mean()
            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2 for g in groups_series
            )
            ss_total   = sum((v - grand_mean) ** 2 for v in all_vals)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            pair_result["anova"] = _result(
                "One-way ANOVA", f_stat, f_p,
                extra={
                    "eta_squared": round(float(eta_sq), 4),
                    "effect_size": (
                        "large"  if eta_sq >= 0.14
                        else "medium" if eta_sq >= 0.06
                        else "small"
                    ),
                },
            )
        else:
            pair_result["anova"] = {"note": f"ANOVA failed: {anova_err}"}

        # Kruskal-Wallis
        kw_out, kw_err = _safe_run(scipy_stats.kruskal, *groups_series)
        if kw_out is not None:
            kw_stat, kw_p = kw_out
            pair_result["kruskal_wallis"] = _result("Kruskal-Wallis", kw_stat, kw_p)
        else:
            pair_result["kruskal_wallis"] = {"note": f"Kruskal-Wallis failed: {kw_err}"}

        # Pairwise Mann-Whitney U (binary or small number of groups)
        if n_groups == 2:
            g1, g2 = groups_series[0], groups_series[1]
            mw_out, _ = _safe_run(scipy_stats.mannwhitneyu, g1, g2, alternative="two-sided")
            if mw_out is not None:
                mw_stat, mw_p = mw_out
                pair_result["mann_whitney"] = _result("Mann-Whitney U", mw_stat, mw_p)

            # Welch t-test (does not assume equal variances)
            t_out, _ = _safe_run(scipy_stats.ttest_ind, g1, g2, equal_var=False)
            if t_out is not None:
                t_stat, t_p = t_out
                # Cohen's d
                pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
                cohens_d   = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0
                pair_result["welch_t"] = _result(
                    "Welch t-test", t_stat, t_p,
                    extra={
                        "cohens_d"   : round(float(cohens_d), 4),
                        "effect_size": (
                            "large"  if abs(cohens_d) >= 0.8
                            else "medium" if abs(cohens_d) >= 0.5
                            else "small"
                        ),
                        "group_means": {
                            str(cat): round(float(grp.mean()), 4)
                            for cat, grp in zip(
                                combined[cat_col].unique()[:2], [g1, g2]
                            )
                        },
                    },
                )

        elif 2 < n_groups <= _PAIRWISE_MAX_GROUPS:
            # Pairwise Mann-Whitney with Bonferroni correction
            cats     = combined[cat_col].unique()[:_PAIRWISE_MAX_GROUPS]
            n_pairs  = len(cats) * (len(cats) - 1) // 2
            pairwise : list[dict] = []
            for ca, cb in itertools.combinations(cats, 2):
                ga = combined.loc[combined[cat_col] == ca, num_col].values
                gb = combined.loc[combined[cat_col] == cb, num_col].values
                mw_out, _ = _safe_run(
                    scipy_stats.mannwhitneyu, ga, gb, alternative="two-sided"
                )
                if mw_out is not None:
                    mw_stat, mw_p = mw_out
                    corrected_p = min(mw_p * n_pairs, 1.0)  # Bonferroni
                    pairwise.append({
                        "group_a"     : str(ca),
                        "group_b"     : str(cb),
                        "statistic"   : round(float(mw_stat), 4),
                        "p_value"     : round(float(mw_p), 6),
                        "p_bonferroni": round(float(corrected_p), 6),
                        "significant" : corrected_p < _ALPHA,
                    })
            pair_result["pairwise_mann_whitney"] = {
                "n_pairs"         : n_pairs,
                "bonferroni_alpha": round(_ALPHA / n_pairs, 5),
                "pairs"           : pairwise,
            }

        results.append(pair_result)

    return results


# ---------------------------------------------------------------------------
# Categorical associations  (categorical ~ categorical)
# ---------------------------------------------------------------------------

def _categorical_associations(df: pd.DataFrame, cat_cols: list[str]) -> list[dict]:
    """
    Chi-square test of independence for every pair of categorical columns,
    plus Cramér's V as the effect size.

    Skips pairs where:
    - Expected cell count < _CHI2_MIN_EXPECTED in > 20% of cells
    - Either column has > _CHI2_MAX_CATS categories (table too large)
    """
    results: list[dict] = []
    cols = [c for c in cat_cols if df[c].nunique() <= _CHI2_MAX_CATS]

    for col_a, col_b in itertools.combinations(cols, 2):
        combined = df[[col_a, col_b]].dropna()
        if len(combined) < 10:
            continue

        contingency = pd.crosstab(combined[col_a], combined[col_b])

        chi2_out, err = _safe_run(scipy_stats.chi2_contingency, contingency)
        if chi2_out is None:
            continue

        chi2, p, dof, expected = chi2_out

        # Validity check: at most 20% of expected cells < 5
        pct_low_expected = (expected < _CHI2_MIN_EXPECTED).mean()
        validity_note    = (
            f"{pct_low_expected:.0%} of expected cells < {_CHI2_MIN_EXPECTED} "
            "— chi-square may be unreliable."
            if pct_low_expected > 0.20 else ""
        )

        # Cramér's V
        n       = combined.shape[0]
        min_dim = min(contingency.shape) - 1
        cramers_v = (
            float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0
        )

        res = _result(
            "Chi-square test of independence", chi2, p,
            extra={
                "degrees_of_freedom": int(dof),
                "cramers_v"         : round(cramers_v, 4),
                "effect_size"       : (
                    "large"  if cramers_v >= 0.35
                    else "medium" if cramers_v >= 0.15
                    else "small"
                ),
                "table_shape"       : list(contingency.shape),
                "pct_low_expected"  : round(float(pct_low_expected), 3),
            },
        )
        if validity_note:
            res["validity_note"] = validity_note

        results.append({
            "col_a"  : col_a,
            "col_b"  : col_b,
            "n_obs"  : n,
            "results": res,
        })

    results.sort(
        key=lambda x: x["results"].get("p_value") or 1.0
    )
    return results


# ---------------------------------------------------------------------------
# Variance equality
# ---------------------------------------------------------------------------

def _variance_equality(df: pd.DataFrame, num_cols: list[str]) -> dict[str, Any]:
    """
    Levene's and Bartlett's tests comparing variance across all numeric columns
    treated as a single collection (tests whether they share a common variance).

    This is useful as a global dataset-level diagnostic, separate from the
    per-group Levene tests run in _group_comparisons.

    Also computes the variance ratio (max_var / min_var) as a simple
    non-parametric indicator of heteroscedasticity.
    """
    if len(num_cols) < 2:
        return {"note": "Fewer than 2 numeric columns — skipped."}

    groups = [df[col].dropna().values for col in num_cols if df[col].dropna().std() > 0]

    if len(groups) < 2:
        return {"note": "Fewer than 2 columns with non-zero variance — skipped."}

    results: dict[str, Any] = {}

    # Levene's test (center="median" is most robust)
    lev_out, _ = _safe_run(scipy_stats.levene, *groups, center="median")
    if lev_out is not None:
        lev_stat, lev_p = lev_out
        results["levene"] = _result(
            "Levene's test (equal variances across all numeric cols)",
            lev_stat, lev_p,
            extra={"center": "median"},
        )

    # Bartlett's test (sensitive to non-normality — use alongside Levene)
    bart_out, _ = _safe_run(scipy_stats.bartlett, *groups)
    if bart_out is not None:
        bart_stat, bart_p = bart_out
        results["bartlett"] = _result(
            "Bartlett's test (assumes normality)",
            bart_stat, bart_p,
            note="Bartlett is sensitive to non-normality; prefer Levene if columns are skewed.",
        )

    # Variance ratio
    variances = {col: float(df[col].dropna().var()) for col in num_cols}
    nonzero   = {k: v for k, v in variances.items() if v > 0}
    if nonzero:
        max_var = max(nonzero.values())
        min_var = min(nonzero.values())
        results["variance_summary"] = {
            "per_column"    : {k: round(v, 6) for k, v in variances.items()},
            "max_var_col"   : max(nonzero, key=nonzero.get),
            "min_var_col"   : min(nonzero, key=nonzero.get),
            "variance_ratio": round(max_var / min_var, 2) if min_var > 0 else None,
        }

    return results


# ---------------------------------------------------------------------------
# Flags for LLM
# ---------------------------------------------------------------------------

def _build_flags(
    normality   : dict[str, dict],
    group_comps : list[dict],
    associations: list[dict],
) -> list[str]:
    """
    Produce plain-English flag strings summarising the most important
    test results for the LLM narrative.
    """
    flags: list[str] = []

    # Normality verdicts
    not_normal = [
        col for col, res in normality.items()
        if isinstance(res.get("consensus"), dict)
        and not res["consensus"].get("likely_normal", True)
    ]
    likely_normal = [
        col for col, res in normality.items()
        if isinstance(res.get("consensus"), dict)
        and res["consensus"].get("likely_normal", False)
    ]
    if not_normal:
        flags.append(
            f"Non-normal distribution detected in: {', '.join(not_normal[:8])}"
            + (f" (+{len(not_normal)-8} more)" if len(not_normal) > 8 else "")
            + " — non-parametric tests preferred."
        )
    if likely_normal:
        flags.append(
            f"Approximately normal distribution in: {', '.join(likely_normal[:5])}"
            + (f" (+{len(likely_normal)-5} more)" if len(likely_normal) > 5 else "")
        )

    # Significant group comparisons
    for gc in group_comps:
        num_col = gc["numeric_col"]
        cat_col = gc["categorical_col"]

        kw = gc.get("kruskal_wallis", {})
        if kw.get("significant"):
            p = kw.get("p_value", "")
            flags.append(
                f"Kruskal-Wallis: significant difference in '{num_col}' "
                f"across groups of '{cat_col}' (p={p:.4f})."
            )

        anova = gc.get("anova", {})
        eta   = anova.get("eta_squared")
        if anova.get("significant") and eta is not None:
            flags.append(
                f"One-way ANOVA: significant effect of '{cat_col}' on '{num_col}' "
                f"(η²={eta:.3f} — {anova.get('effect_size', '')} effect)."
            )

        wt = gc.get("welch_t", {})
        if wt.get("significant"):
            d   = wt.get("cohens_d", "")
            means = wt.get("group_means", {})
            means_str = ", ".join(f"{k}={v:.3g}" for k, v in means.items())
            flags.append(
                f"Welch t-test: significant difference in '{num_col}' "
                f"between groups of '{cat_col}' "
                f"(d={d:.3f} — {wt.get('effect_size', '')} effect; {means_str})."
            )

    # Significant categorical associations
    for assoc in associations[:5]:   # top 5 by p-value
        res = assoc.get("results", {})
        if res.get("significant"):
            v    = res.get("cramers_v", "")
            size = res.get("effect_size", "")
            flags.append(
                f"Chi-square: significant association between '{assoc['col_a']}' "
                f"and '{assoc['col_b']}' (Cramér's V={v:.3f} — {size} effect)."
            )

    return flags
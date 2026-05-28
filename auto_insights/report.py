"""
auto_insights/report.py

Report builder — assembles all analysis results, figures, and LLM narrative
into a self-contained, single-file HTML report.

Public API
----------
build_report(analysis_result, narrative, figures, output_path)  ->  str
    Top-level entry point called by core.py.
    Writes the HTML report to *output_path* and returns the HTML string.

build_markdown_report(analysis_result, narrative)  ->  str
    Lighter alternative — returns a Markdown string (no figures embedded).
    Useful for quick previews or LLM context injection.

Report structure
----------------
    Header bar         — dataset name, timestamp, model used
    Executive Summary  — LLM overview + key flags at a glance
    Distributions      — numeric overview grid + per-column hist/box pairs
                         + LLM distribution narrative
    Categorical        — bar charts + frequency table excerpts
    Correlations       — heatmap(s) + top-pairs bar + pairplot + LLM narrative
    Group Comparisons  — test result tables + LLM narrative
    Associations       — chi-square table + LLM narrative
    Recommendations    — LLM recommendations section
    Appendix           — raw stats tables (collapsible)

Design
------
- Pure HTML + inline CSS + minimal vanilla JS (no CDN dependencies).
- All figures are embedded as base64 data URIs → truly self-contained.
- Colour palette mirrors the viz.py / corr.py palette for visual consistency.
- Collapsible <details> elements for the appendix keep the report scannable.
- Tables are rendered from dicts; no pandas .to_html() so formatting is
  fully controlled.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .llm import InsightsNarrative
from .utils import fig_to_base64, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Palette (mirrors viz.py / corr.py)
# ---------------------------------------------------------------------------

_C = {
    "primary"  : "#5B5EA6",
    "secondary": "#48A999",
    "accent"   : "#E07B54",
    "neutral"  : "#9B9EA4",
    "danger"   : "#C0392B",
    "bg"       : "#FAFAFA",
    "surface"  : "#FFFFFF",
    "border"   : "#E8E8EC",
    "text"     : "#1A1A2E",
    "muted"    : "#6B7280",
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_report(
    analysis_result : dict[str, Any],
    narrative       : InsightsNarrative,
    figures         : dict[str, plt.Figure],
    output_path     : str | Path = "auto_insights_report.html",
    dataset_name    : str = "Dataset",
) -> str:
    """
    Build and write a self-contained HTML report.

    Parameters
    ----------
    analysis_result : full output dict from core.py
    narrative       : InsightsNarrative from llm.py
    figures         : dict of {label: Figure} from viz.py + corr.py
    output_path     : where to write the .html file
    dataset_name    : displayed in the report header

    Returns
    -------
    The full HTML string (also written to output_path).
    """
    logger.info("report: serializing %d figures...", len(figures))
    b64_figs = {
        label: fig_to_base64(fig, fmt="png", dpi=110)
        for label, fig in figures.items()
    }
    # Close all figures to free memory after serialization
    for fig in figures.values():
        plt.close(fig)

    logger.info("report: assembling HTML...")
    html_str = _render_html(
        analysis_result = analysis_result,
        narrative       = narrative,
        b64_figs        = b64_figs,
        dataset_name    = dataset_name,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_str, encoding="utf-8")
    logger.info("report: written to %s  (%d KB)", output_path, len(html_str) // 1024)
    return html_str


def build_markdown_report(
    analysis_result : dict[str, Any],
    narrative       : InsightsNarrative,
    dataset_name    : str = "Dataset",
) -> str:
    """
    Build a lightweight Markdown report (no embedded figures).

    Useful for quick previews, Jupyter display, or feeding back into an LLM.
    """
    meta     = analysis_result.get("stats", {}).get("metadata", {})
    n_rows   = meta.get("n_rows", "?")
    n_cols   = meta.get("n_cols", "?")
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        f"# Auto Insights — {dataset_name}",
        f"*Generated {ts} · {n_rows} rows × {n_cols} columns*",
        "",
        "## Dataset Overview",
        narrative.overview or "_No overview generated._",
        "",
        "## Distributions & Outliers",
        narrative.distributions or "_No distribution narrative generated._",
        "",
        "## Correlations",
        narrative.correlations or "_No correlation narrative generated._",
        "",
        "## Group Comparisons",
        narrative.group_tests or "_No group comparison narrative generated._",
        "",
        "## Categorical Associations",
        narrative.associations or "_No association narrative generated._",
        "",
        "## Recommendations",
        narrative.recommendations or "_No recommendations generated._",
    ]

    if narrative.errors:
        sections += ["", "## Errors", "\n".join(f"- {e}" for e in narrative.errors)]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

def _render_html(
    analysis_result : dict[str, Any],
    narrative       : InsightsNarrative,
    b64_figs        : dict[str, str],
    dataset_name    : str,
) -> str:
    stats     = analysis_result.get("stats",        {})
    corr      = analysis_result.get("correlations", {})
    tests     = analysis_result.get("tests",        {})
    meta      = stats.get("metadata", {})
    overview  = stats.get("overview", {})
    ds_level  = stats.get("dataset_level", {})

    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_rows    = meta.get("n_rows", "?")
    n_cols    = meta.get("n_cols", "?")
    mem_mb    = meta.get("memory_mb", "?")
    tok_used  = narrative.total_tokens

    body_parts: list[str] = []

    # 1 · Header
    body_parts.append(_section_header(dataset_name, ts, n_rows, n_cols, mem_mb, tok_used, narrative.model))

    # 2 · Executive summary
    body_parts.append(_section_executive_summary(narrative, overview, ds_level))

    # 3 · Distributions
    body_parts.append(_section_distributions(stats, narrative, b64_figs))

    # 4 · Categorical
    body_parts.append(_section_categorical(stats, b64_figs))

    # 5 · Correlations
    body_parts.append(_section_correlations(corr, narrative, b64_figs))

    # 6 · Group comparisons
    body_parts.append(_section_group_tests(tests, narrative))

    # 7 · Associations
    body_parts.append(_section_associations(tests, narrative))

    # 8 · Recommendations
    body_parts.append(_section_recommendations(narrative))

    # 9 · Appendix (collapsible raw stats)
    body_parts.append(_section_appendix(stats))

    # Errors footer
    if narrative.errors:
        body_parts.append(_errors_footer(narrative.errors))

    body = "\n".join(body_parts)
    return _html_shell(body, dataset_name)


# ---------------------------------------------------------------------------
# HTML shell (CSS + JS)
# ---------------------------------------------------------------------------

def _html_shell(body: str, title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auto Insights — {html.escape(title)}</title>
<style>
  /* ── Reset & base ── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Georgia', serif;
    background: {_C['bg']};
    color: {_C['text']};
    line-height: 1.65;
    font-size: 15px;
  }}

  /* ── Layout ── */
  .page-wrap {{ max-width: 1100px; margin: 0 auto; padding: 0 24px 80px; }}

  /* ── Header ── */
  .report-header {{
    background: {_C['primary']};
    color: white;
    padding: 36px 40px 28px;
    margin-bottom: 40px;
  }}
  .report-header h1 {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }}
  .report-header .subtitle {{ opacity: 0.82; font-size: 0.9rem; margin-top: 6px; font-family: monospace; }}
  .meta-chips {{ display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }}
  .chip {{
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.30);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-family: monospace;
    color: white;
  }}

  /* ── Section ── */
  .section {{ margin-bottom: 52px; }}
  .section-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {_C['primary']};
    border-bottom: 2px solid {_C['primary']};
    padding-bottom: 8px;
    margin-bottom: 20px;
    letter-spacing: 0.3px;
  }}
  .section-title span {{ font-size: 0.78rem; font-weight: 400; color: {_C['muted']}; margin-left: 8px; }}

  /* ── Narrative box ── */
  .narrative {{
    background: white;
    border-left: 3px solid {_C['secondary']};
    padding: 16px 20px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 24px;
    font-size: 0.95rem;
    color: #2d2d2d;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .narrative.accent {{ border-left-color: {_C['accent']}; }}
  .narrative.primary {{ border-left-color: {_C['primary']}; }}

  /* ── Flags ── */
  .flags {{ display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px; }}
  .flag {{
    display: flex; align-items: flex-start; gap: 10px;
    background: white; border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.86rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    border: 1px solid {_C['border']};
  }}
  .flag-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    margin-top: 5px; flex-shrink: 0;
  }}
  .flag-dot.warn  {{ background: {_C['accent']}; }}
  .flag-dot.ok    {{ background: {_C['secondary']}; }}
  .flag-dot.error {{ background: {_C['danger']}; }}

  /* ── Figures ── */
  .fig-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; margin-bottom: 24px; }}
  .fig-card {{
    background: white; border-radius: 8px;
    border: 1px solid {_C['border']};
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .fig-card img {{ width: 100%; display: block; }}
  .fig-label {{ padding: 8px 12px; font-size: 0.78rem; color: {_C['muted']}; font-family: monospace; background: {_C['bg']}; }}
  .fig-full {{ background: white; border-radius: 8px; border: 1px solid {_C['border']}; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 20px; }}
  .fig-full img {{ width: 100%; display: block; }}

  /* ── Tables ── */
  .tbl-wrap {{ overflow-x: auto; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th {{
    background: {_C['primary']}; color: white;
    padding: 9px 14px; text-align: left;
    font-weight: 600; font-size: 0.8rem; letter-spacing: 0.3px;
  }}
  td {{ padding: 8px 14px; border-bottom: 1px solid {_C['border']}; vertical-align: top; }}
  tr:nth-child(even) td {{ background: {_C['bg']}; }}
  tr:hover td {{ background: #f0f0f8; }}
  td.num {{ font-family: monospace; text-align: right; }}
  td.sig {{ color: {_C['accent']}; font-weight: 600; }}
  td.ok  {{ color: {_C['secondary']}; }}

  /* ── KV grid (stat cards) ── */
  .kv-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .kv-card {{
    background: white; border-radius: 8px; padding: 14px 16px;
    border: 1px solid {_C['border']};
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .kv-card .kv-val {{ font-size: 1.5rem; font-weight: 700; color: {_C['primary']}; font-family: monospace; }}
  .kv-card .kv-label {{ font-size: 0.78rem; color: {_C['muted']}; margin-top: 2px; }}

  /* ── Collapsible ── */
  details {{ margin-bottom: 16px; }}
  summary {{
    cursor: pointer; font-weight: 600; font-size: 0.9rem;
    padding: 10px 14px; background: white;
    border: 1px solid {_C['border']}; border-radius: 6px;
    list-style: none; display: flex; justify-content: space-between;
    color: {_C['primary']};
  }}
  summary::after {{ content: '▸'; transition: transform 0.2s; }}
  details[open] summary::after {{ transform: rotate(90deg); }}
  details[open] summary {{ border-radius: 6px 6px 0 0; }}
  .details-body {{ background: white; border: 1px solid {_C['border']}; border-top: none; border-radius: 0 0 6px 6px; padding: 16px; }}

  /* ── Errors ── */
  .errors {{ background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 16px 20px; margin-top: 40px; }}
  .errors h3 {{ color: {_C['danger']}; margin-bottom: 10px; font-size: 0.95rem; }}
  .errors li {{ font-size: 0.85rem; color: #7f1d1d; margin: 4px 0 4px 16px; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-thumb {{ background: {_C['border']}; border-radius: 3px; }}
</style>
</head>
<body>
<div class="page-wrap">
{body}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _section_header(
    dataset_name: str, ts: str, n_rows, n_cols, mem_mb, tok_used, model: str
) -> str:
    return f"""
<div class="report-header">
  <h1>Auto Insights — {html.escape(dataset_name)}</h1>
  <div class="subtitle">Generated {ts}</div>
  <div class="meta-chips">
    <span class="chip">⬛ {n_rows:,} rows</span>
    <span class="chip">⬛ {n_cols} columns</span>
    <span class="chip">⬛ {mem_mb} MB</span>
    <span class="chip">⬛ {tok_used:,} tokens</span>
    <span class="chip">⬛ {html.escape(model)}</span>
  </div>
</div>"""


def _section_executive_summary(
    narrative: InsightsNarrative,
    overview : dict,
    ds_level : dict,
) -> str:
    n_rows      = overview.get("n_rows", "?")
    n_cols      = overview.get("n_cols", "?")
    dup_rows    = overview.get("duplicate_rows", 0)
    null_overall= overview.get("null_pct_overall", 0)
    type_counts = overview.get("column_type_counts", {})

    kv_cards = "".join([
        _kv_card(f"{n_rows:,}", "rows"),
        _kv_card(str(n_cols), "columns"),
        _kv_card(f"{null_overall:.1f}%", "missing values"),
        _kv_card(str(dup_rows), "duplicate rows"),
        *[_kv_card(str(v), f"{k} cols") for k, v in type_counts.items()],
    ])

    flags_html = ""
    all_flags: list[tuple[str, str]] = []
    for col in ds_level.get("high_null_cols", []):
        all_flags.append(("warn", f"High missingness: <code>{col}</code>"))
    for col in ds_level.get("zero_variance_cols", []):
        all_flags.append(("error", f"Zero variance (constant): <code>{col}</code>"))
    for entry in ds_level.get("high_skew_cols", [])[:5]:
        all_flags.append(("warn", f"High skew ({entry['skewness']:+.2f}): <code>{entry['column']}</code>"))
    for pair in ds_level.get("perfect_corr_pairs", []):
        all_flags.append(("error", f"Perfect correlation: <code>{pair['col_a']}</code> ↔ <code>{pair['col_b']}</code>"))

    if all_flags:
        flags_html = '<div class="flags">' + "".join(
            f'<div class="flag"><div class="flag-dot {cls}"></div><div>{msg}</div></div>'
            for cls, msg in all_flags
        ) + '</div>'

    narrative_html = ""
    if narrative.overview:
        narrative_html = f'<div class="narrative primary">{html.escape(narrative.overview)}</div>'

    return f"""
<div class="section">
  <div class="section-title">Executive Summary</div>
  <div class="kv-grid">{kv_cards}</div>
  {flags_html}
  {narrative_html}
</div>"""


def _section_distributions(
    stats    : dict,
    narrative: InsightsNarrative,
    b64_figs : dict[str, str],
) -> str:
    parts: list[str] = []
    parts.append('<div class="section">')
    parts.append('<div class="section-title">Distributions &amp; Outliers</div>')

    if narrative.distributions:
        parts.append(f'<div class="narrative">{html.escape(narrative.distributions)}</div>')

    # Numeric overview grid
    if "numeric_overview" in b64_figs:
        parts.append(_fig_full("numeric_overview", b64_figs, "All numeric columns — distribution overview"))

    # Per-column hist + box pairs
    numeric_cols = list(stats.get("numeric", {}).keys())
    if numeric_cols:
        parts.append('<div class="fig-grid">')
        for col in numeric_cols:
            hist_key = f"hist_{col}"
            box_key  = f"box_{col}"
            if hist_key in b64_figs:
                parts.append(_fig_card(hist_key, b64_figs, f"{col} — histogram"))
            if box_key in b64_figs:
                parts.append(_fig_card(box_key,  b64_figs, f"{col} — boxplot"))
        parts.append('</div>')

    # Appendix: numeric stats table
    numeric_stats = stats.get("numeric", {})
    if numeric_stats:
        rows = []
        for col, s in numeric_stats.items():
            if "note" in s:
                continue
            rows.append({
                "column"  : col,
                "count"   : s.get("count", ""),
                "mean"    : _fmt(s.get("mean")),
                "std"     : _fmt(s.get("std")),
                "min"     : _fmt(s.get("min")),
                "median"  : _fmt(s.get("median")),
                "max"     : _fmt(s.get("max")),
                "skewness": _fmt(s.get("skewness")),
                "outliers": s.get("outlier_count", ""),
                "nulls"   : s.get("null_count", ""),
            })
        if rows:
            parts.append(
                _collapsible("Numeric statistics table", _dict_table(rows))
            )

    parts.append('</div>')
    return "\n".join(parts)


def _section_categorical(stats: dict, b64_figs: dict[str, str]) -> str:
    cat_stats = stats.get("categorical", {})
    if not cat_stats:
        return ""

    parts: list[str] = ['<div class="section">',
                         '<div class="section-title">Categorical Columns</div>']

    parts.append('<div class="fig-grid">')
    for col in cat_stats:
        key = f"bar_{col}"
        if key in b64_figs:
            parts.append(_fig_card(key, b64_figs, f"{col}"))
    parts.append('</div>')

    # Frequency table excerpts
    for col, s in cat_stats.items():
        freq = s.get("freq_table", [])
        if not freq:
            continue
        rows = [
            {"value": str(r["value"]), "count": r["count"], "pct": f"{r['pct']:.1f}%"}
            for r in freq[:10]
        ]
        parts.append(
            _collapsible(
                f"{col} — top {len(rows)} values  "
                f"(n_unique={s.get('n_unique','?')}, entropy={s.get('entropy_bits','?'):.2f} bits)",
                _dict_table(rows),
            )
        )

    parts.append('</div>')
    return "\n".join(parts)


def _section_correlations(
    corr     : dict,
    narrative: InsightsNarrative,
    b64_figs : dict[str, str],
) -> str:
    if not corr or "note" in corr:
        return ""

    parts: list[str] = ['<div class="section">',
                         '<div class="section-title">Correlations</div>']

    if narrative.correlations:
        parts.append(f'<div class="narrative">{html.escape(narrative.correlations)}</div>')

    # Heatmaps side-by-side
    pearson_key  = "corr_heatmap_pearson"
    spearman_key = "corr_heatmap_spearman"
    if pearson_key in b64_figs or spearman_key in b64_figs:
        parts.append('<div class="fig-grid">')
        if pearson_key  in b64_figs:
            parts.append(_fig_card(pearson_key,  b64_figs, "Pearson r"))
        if spearman_key in b64_figs:
            parts.append(_fig_card(spearman_key, b64_figs, "Spearman ρ"))
        parts.append('</div>')

    # Top pairs bar
    if "corr_top_pairs" in b64_figs:
        parts.append(_fig_full("corr_top_pairs", b64_figs, "Strongest correlated pairs"))

    # Pairplot
    if "corr_pairplot" in b64_figs:
        parts.append(_fig_full("corr_pairplot", b64_figs, "Pairplot"))

    # Top pairs table
    top_pairs = corr.get("top_pairs_pearson", [])[:15]
    if top_pairs:
        rows = [
            {
                "col A"    : p["col_a"],
                "col B"    : p["col_b"],
                "Pearson r": _fmt(p["r"]),
                "direction": p["direction"],
                "strength" : p["strength"],
            }
            for p in top_pairs
        ]
        parts.append(_collapsible("Top correlated pairs (Pearson)", _dict_table(rows)))

    parts.append('</div>')
    return "\n".join(parts)


def _section_group_tests(tests: dict, narrative: InsightsNarrative) -> str:
    group_comps = tests.get("group_comparisons", [])
    if not group_comps:
        return ""

    parts: list[str] = ['<div class="section">',
                         '<div class="section-title">Group Comparisons</div>']

    if narrative.group_tests:
        parts.append(f'<div class="narrative">{html.escape(narrative.group_tests)}</div>')

    # Summary table: one row per numeric×categorical pair
    rows = []
    for gc in group_comps:
        kw   = gc.get("kruskal_wallis", {})
        anova= gc.get("anova", {})
        wt   = gc.get("welch_t", {})

        rows.append({
            "numeric col"      : gc["numeric_col"],
            "group col"        : gc["categorical_col"],
            "n groups"         : gc.get("n_groups", ""),
            "KW p-value"       : _fmt(kw.get("p_value")),
            "KW sig"           : "✓" if kw.get("significant") else "",
            "ANOVA p-value"    : _fmt(anova.get("p_value")),
            "ANOVA η²"         : _fmt(anova.get("eta_squared")),
            "Welch t p-value"  : _fmt(wt.get("p_value")),
            "Cohen's d"        : _fmt(wt.get("cohens_d")),
        })

    parts.append(_collapsible("Group comparison results", _dict_table(rows)))
    parts.append('</div>')
    return "\n".join(parts)


def _section_associations(tests: dict, narrative: InsightsNarrative) -> str:
    associations = tests.get("associations", [])
    if not associations:
        return ""

    parts: list[str] = ['<div class="section">',
                         '<div class="section-title">Categorical Associations</div>']

    if narrative.associations:
        parts.append(f'<div class="narrative">{html.escape(narrative.associations)}</div>')

    rows = []
    for a in associations:
        res = a.get("results", {})
        rows.append({
            "col A"      : a["col_a"],
            "col B"      : a["col_b"],
            "n obs"      : a.get("n_obs", ""),
            "χ² stat"    : _fmt(res.get("statistic")),
            "p-value"    : _fmt(res.get("p_value")),
            "significant": "✓" if res.get("significant") else "",
            "Cramér's V" : _fmt(res.get("cramers_v")),
            "effect"     : res.get("effect_size", ""),
        })

    parts.append(_collapsible("Chi-square association results", _dict_table(rows)))
    parts.append('</div>')
    return "\n".join(parts)


def _section_recommendations(narrative: InsightsNarrative) -> str:
    if not narrative.recommendations:
        return ""

    return f"""
<div class="section">
  <div class="section-title">Recommendations</div>
  <div class="narrative accent">{html.escape(narrative.recommendations)}</div>
</div>"""


def _section_appendix(stats: dict) -> str:
    """Collapsible raw stats appendix."""
    parts = ['<div class="section">',
             '<div class="section-title">Appendix <span>raw statistics</span></div>']

    # Null summary
    nulls = stats.get("overview", {}).get("columns_with_nulls", {})
    if nulls:
        rows = [
            {"column": col, "null count": v["count"], "null %": f"{v['pct']:.1f}%"}
            for col, v in nulls.items()
        ]
        parts.append(_collapsible("Missing value summary", _dict_table(rows)))

    # Categorical full freq tables are already in the categorical section
    # — append variance equality results
    var_eq = stats.get("variance_equality", {})
    if var_eq and "variance_summary" in var_eq:
        vs = var_eq["variance_summary"]
        rows = [
            {"column": col, "variance": _fmt(v)}
            for col, v in vs.get("per_column", {}).items()
        ]
        note = (
            f"Variance ratio (max/min): {vs.get('variance_ratio', '?')}  |  "
            f"Highest variance: {vs.get('max_var_col', '?')}  |  "
            f"Lowest variance: {vs.get('min_var_col', '?')}"
        )
        parts.append(
            _collapsible(f"Variance by column — {note}", _dict_table(rows))
        )

    parts.append('</div>')
    return "\n".join(parts)


def _errors_footer(errors: list[str]) -> str:
    items = "".join(f"<li>{html.escape(e)}</li>" for e in errors)
    return f'<div class="errors"><h3>⚠ Errors during generation</h3><ul>{items}</ul></div>'


# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------

def _kv_card(value: str, label: str) -> str:
    return (
        f'<div class="kv-card">'
        f'<div class="kv-val">{html.escape(str(value))}</div>'
        f'<div class="kv-label">{html.escape(label)}</div>'
        f'</div>'
    )


def _fig_card(key: str, b64_figs: dict, label: str) -> str:
    src = b64_figs.get(key, "")
    if not src:
        return ""
    return (
        f'<div class="fig-card">'
        f'<img src="{src}" alt="{html.escape(label)}" loading="lazy">'
        f'<div class="fig-label">{html.escape(label)}</div>'
        f'</div>'
    )


def _fig_full(key: str, b64_figs: dict, label: str) -> str:
    src = b64_figs.get(key, "")
    if not src:
        return ""
    return (
        f'<div class="fig-full">'
        f'<img src="{src}" alt="{html.escape(label)}" loading="lazy">'
        f'<div class="fig-label">{html.escape(label)}</div>'
        f'</div>'
    )


def _dict_table(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    headers = list(rows[0].keys())
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = ""
        for h in headers:
            val = row.get(h, "")
            css = "num" if isinstance(val, (int, float)) else ""
            tds += f'<td class="{css}">{html.escape(str(val))}</td>'
        trs += f"<tr>{tds}</tr>"
    return f'<div class="tbl-wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def _collapsible(summary_text: str, content_html: str) -> str:
    return (
        f"<details>"
        f"<summary>{html.escape(summary_text)}</summary>"
        f'<div class="details-body">{content_html}</div>'
        f"</details>"
    )


def _fmt(val: Any, decimals: int = 4) -> str:
    """Format a numeric value for table display."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}g}"
    except (TypeError, ValueError):
        return str(val)
"""
auto_insights/core.py

Orchestration layer — the only class the user needs to interact with.

Public API
----------
    from auto_insights import InsightsGenerator

    gen    = InsightsGenerator(df, dataset_name="My Dataset")
    report = gen.run()                        # full pipeline
    report.save("report.html")               # write HTML
    report.markdown()                         # quick text summary
    report.narrative.recommendations          # access any LLM section directly
    report.figures["hist_velocity"]           # access any figure directly

InsightsGenerator.run() pipeline
---------------------------------
    1.  validate_dataframe()          utils.py
    2.  profile_dataframe()           stats.py
    3.  generate_all_figures()        viz.py
    4.  compute_correlations()        corr.py
    5.  generate_correlation_figures  corr.py
    6.  run_statistical_tests()       tests.py
    7.  generate_insights()           llm.py
    8.  build_report()                report.py

Each step is timed and logged. Any step can be individually disabled via
constructor flags so users can skip LLM calls during dev / testing or skip
viz when running headless.

InsightsReport (return type)
-----------------------------
A lightweight dataclass bundling all outputs. Exposes convenience methods
so users never need to import report.py or llm.py directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .utils import (
    dataframe_metadata,
    get_logger,
    validate_dataframe,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class InsightsReport:
    """
    Container for all outputs produced by InsightsGenerator.run().

    Attributes
    ----------
    stats           : output of stats.profile_dataframe()
    correlations    : output of corr.compute_correlations()
    tests           : output of tests.run_statistical_tests()
    narrative       : InsightsNarrative from llm.py
    figures         : dict of {label: matplotlib Figure}
    html            : rendered HTML string (empty if run_report=False)
    dataset_name    : name passed at construction
    elapsed_seconds : wall-clock time for the full pipeline
    """
    stats          : dict[str, Any]         = field(default_factory=dict)
    correlations   : dict[str, Any]         = field(default_factory=dict)
    tests          : dict[str, Any]         = field(default_factory=dict)
    narrative      : Any                    = None   # InsightsNarrative
    figures        : dict[str, plt.Figure]  = field(default_factory=dict)
    html           : str                    = ""
    dataset_name   : str                    = "Dataset"
    elapsed_seconds: float                  = 0.0

    # ------------------------------------------------------------------ #
    # Convenience methods                                                  #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path = "auto_insights_report.html") -> Path:
        """
        Write the HTML report to *path*.

        Returns the resolved Path so callers can chain or log it:
            print(report.save("output/report.html"))
        """
        path = Path(path)
        if not self.html:
            raise RuntimeError(
                "No HTML content available. "
                "Re-run with run_report=True (the default)."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html, encoding="utf-8")
        logger.info("Report saved to %s", path)
        return path

    def markdown(self) -> str:
        """
        Return a Markdown text summary (no figures).

        Useful for Jupyter display or quick inspection:
            from IPython.display import Markdown, display
            display(Markdown(report.markdown()))
        """
        from .report import build_markdown_report
        return build_markdown_report(
            analysis_result=self._analysis_result(),
            narrative=self.narrative,
            dataset_name=self.dataset_name,
        )

    def show_figure(self, label: str) -> plt.Figure:
        """
        Return a specific figure by label for inline Jupyter display.

        Example
        -------
        report.show_figure("hist_velocity")
        """
        if label not in self.figures:
            available = list(self.figures.keys())
            raise KeyError(
                f"Figure '{label}' not found. "
                f"Available figures: {available}"
            )
        return self.figures[label]

    def list_figures(self) -> list[str]:
        """Return a sorted list of all figure labels."""
        return sorted(self.figures.keys())

    def _analysis_result(self) -> dict:
        """Package stats + correlations + tests into the dict format
        expected by report.py and llm.py."""
        return {
            "stats"       : self.stats,
            "correlations": self.correlations,
            "tests"       : self.tests,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InsightsGenerator:
    """
    One-stop EDA orchestrator.

    Parameters
    ----------
    df                  : pandas DataFrame to analyse
    dataset_name        : human-readable label used in the report header
    llm_model           : Anthropic model string (default: claude-sonnet-4-20250514)
    llm_max_tokens      : max tokens per LLM section call (default: 1024)
    run_llm             : set False to skip all Anthropic API calls
                          (useful for dev / testing / offline use)
    run_viz             : set False to skip all figure generation
    run_report          : set False to skip HTML assembly
    max_numeric_figs    : cap on individual numeric column figures
    max_categorical_figs: cap on individual categorical bar charts

    Usage
    -----
    Basic:
        gen    = InsightsGenerator(df)
        report = gen.run()
        report.save("report.html")

    No LLM (fast, offline):
        report = InsightsGenerator(df, run_llm=False).run()

    Custom model / token budget:
        report = InsightsGenerator(
            df,
            llm_model="claude-opus-4-20250514",
            llm_max_tokens=2048,
        ).run()

    Access individual outputs:
        report.stats["numeric"]["velocity"]["skewness"]
        report.correlations["top_pairs_pearson"][:3]
        report.narrative.recommendations
        report.figures["corr_heatmap_pearson"]
    """

    def __init__(
        self,
        df                  : pd.DataFrame,
        dataset_name        : str  = "Dataset",
        llm_model           : str  = "claude-sonnet-4-20250514",
        llm_max_tokens      : int  = 1024,
        run_llm             : bool = True,
        run_viz             : bool = True,
        run_report          : bool = True,
        max_numeric_figs    : int  = 20,
        max_categorical_figs: int  = 15,
    ) -> None:
        validate_dataframe(df)
        self.df                   = df.copy()
        self.dataset_name         = dataset_name
        self.llm_model            = llm_model
        self.llm_max_tokens       = llm_max_tokens
        self.run_llm              = run_llm
        self.run_viz              = run_viz
        self.run_report           = run_report
        self.max_numeric_figs     = max_numeric_figs
        self.max_categorical_figs = max_categorical_figs

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def run(self) -> InsightsReport:
        """
        Execute the full EDA pipeline and return an InsightsReport.

        Steps run in order:
            stats → viz → correlations → corr_figs → tests → llm → report

        Each step is independently try/except'd so a failure in one module
        (e.g. a bad column type confusing the test runner) doesn't abort
        the rest of the pipeline.  Errors are logged and attached to the
        report so the user can see what went wrong without losing the
        results from all other steps.
        """
        t0 = time.perf_counter()
        logger.info("=" * 60)
        logger.info("InsightsGenerator: starting pipeline for '%s'", self.dataset_name)
        logger.info("  shape   : %d rows × %d cols", *self.df.shape)
        logger.info("  run_llm : %s  |  run_viz: %s  |  run_report: %s",
                    self.run_llm, self.run_viz, self.run_report)
        logger.info("=" * 60)

        report = InsightsReport(dataset_name=self.dataset_name)

        # ── Step 1: descriptive stats ───────────────────────────────────
        report.stats = self._step("stats", self._run_stats)

        # ── Step 2: visualisations ──────────────────────────────────────
        if self.run_viz:
            report.figures.update(
                self._step("viz", self._run_viz, report.stats) or {}
            )

        # ── Step 3: correlations ────────────────────────────────────────
        report.correlations = self._step("correlations", self._run_correlations)

        # ── Step 4: correlation figures ─────────────────────────────────
        if self.run_viz:
            report.figures.update(
                self._step("corr_figures", self._run_corr_figures,
                           report.correlations) or {}
            )

        # ── Step 5: statistical tests ───────────────────────────────────
        report.tests = self._step("tests", self._run_tests)

        # ── Step 6: LLM narrative ───────────────────────────────────────
        if self.run_llm:
            report.narrative = self._step(
                "llm", self._run_llm,
                report.stats, report.correlations, report.tests,
            )
        else:
            from .llm import InsightsNarrative
            report.narrative = InsightsNarrative(
                overview="LLM disabled — run with run_llm=True to generate narrative."
            )

        # ── Step 7: HTML report ─────────────────────────────────────────
        if self.run_report:
            report.html = self._step(
                "report", self._run_report,
                report.stats, report.correlations, report.tests,
                report.narrative, report.figures,
            ) or ""

        report.elapsed_seconds = round(time.perf_counter() - t0, 2)
        logger.info("Pipeline complete in %.1fs.", report.elapsed_seconds)
        return report

    # ------------------------------------------------------------------ #
    # Step runner                                                          #
    # ------------------------------------------------------------------ #

    def _step(self, name: str, fn, *args) -> Any:
        """
        Run *fn(*args)*, log timing, and catch exceptions.

        Returns fn's result on success, or an empty dict on failure.
        Errors are logged at WARNING level — they don't raise so the
        pipeline continues.
        """
        t0 = time.perf_counter()
        logger.info("── step: %s", name)
        try:
            result = fn(*args)
            logger.info("   %s done  (%.2fs)", name, time.perf_counter() - t0)
            return result
        except Exception as exc:
            logger.warning(
                "   %s FAILED (%.2fs): %s",
                name, time.perf_counter() - t0, exc,
                exc_info=True,
            )
            return {}

    # ------------------------------------------------------------------ #
    # Step implementations                                                 #
    # ------------------------------------------------------------------ #

    def _run_stats(self) -> dict:
        from .stats import profile_dataframe
        return profile_dataframe(self.df)

    def _run_viz(self, stats_result: dict) -> dict[str, plt.Figure]:
        from .viz import generate_all_figures
        return generate_all_figures(
            self.df,
            stats_result        = stats_result,
            max_numeric_individual   = self.max_numeric_figs,
            max_categorical_individual = self.max_categorical_figs,
        )

    def _run_correlations(self) -> dict:
        from .corr import compute_correlations
        return compute_correlations(self.df)

    def _run_corr_figures(self, corr_result: dict) -> dict[str, plt.Figure]:
        from .corr import generate_correlation_figures
        return generate_correlation_figures(self.df, corr_result)

    def _run_tests(self) -> dict:
        from .tests import run_statistical_tests
        return run_statistical_tests(self.df)

    def _run_llm(
        self,
        stats_result: dict,
        corr_result : dict,
        test_result : dict,
    ):
        from .llm import generate_insights
        analysis_result = {
            "stats"       : stats_result,
            "correlations": corr_result,
            "tests"       : test_result,
        }
        return generate_insights(
            analysis_result = analysis_result,
            df_meta         = dataframe_metadata(self.df),
            model           = self.llm_model,
            max_tokens      = self.llm_max_tokens,
        )

    def _run_report(
        self,
        stats_result: dict,
        corr_result : dict,
        test_result : dict,
        narrative   : Any,
        figures     : dict[str, plt.Figure],
    ) -> str:
        from .report import build_report
        analysis_result = {
            "stats"       : stats_result,
            "correlations": corr_result,
            "tests"       : test_result,
        }
        return build_report(
            analysis_result = analysis_result,
            narrative       = narrative,
            figures         = figures,
            dataset_name    = self.dataset_name,
        )
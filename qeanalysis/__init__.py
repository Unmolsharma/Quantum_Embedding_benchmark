"""
qeanalysis — Post-benchmark analysis package for QEBench
=========================================================

Separate from qebench: requires only pandas/numpy/scipy/matplotlib.
No D-Wave stack or C++ binaries needed.

Typical workflow
----------------
    from qeanalysis import BenchmarkAnalysis

    an = BenchmarkAnalysis("results/batch_2026-02-24_14-30-00/")
    an.generate_report()   # runs everything; writes to analysis/<batch-name>/

Adding new analysis
-------------------
1. Write a standalone function in the appropriate sub-module
   (summary.py, statistics.py, or plots.py).
2. Add a thin wrapper method to BenchmarkAnalysis below.
3. Call it in generate_report() so it runs automatically.

Public re-exports
-----------------
The most commonly used standalone functions are re-exported here so that
users can do either:
    from qeanalysis import BenchmarkAnalysis
    from qeanalysis.summary import overall_summary   # direct access
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from qeanalysis.loader    import load_batch, infer_category
from qeanalysis.summary   import overall_summary, summary_by_category, rank_table
from qeanalysis.statistics import (
    win_rate_matrix, significance_tests, friedman_test,
    correlation_matrix, density_hardness_summary,
)
from qeanalysis.plots import (
    plot_heatmap, plot_scaling, plot_density_hardness,
    plot_pareto, plot_distributions, plot_head_to_head,
    plot_consistency, plot_topology_comparison,
    plot_problem_deep_dive, plot_chain_distribution,
)
from qeanalysis.export import df_to_latex, export_tables


__all__ = [
    'BenchmarkAnalysis',
    # Loader
    'load_batch', 'infer_category',
    # Summary
    'overall_summary', 'summary_by_category', 'rank_table',
    # Statistics
    'win_rate_matrix', 'significance_tests', 'friedman_test',
    'correlation_matrix', 'density_hardness_summary',
    # Plots
    'plot_heatmap', 'plot_scaling', 'plot_density_hardness',
    'plot_pareto', 'plot_distributions', 'plot_head_to_head',
    'plot_consistency', 'plot_topology_comparison',
    'plot_problem_deep_dive', 'plot_chain_distribution',
    # Export
    'df_to_latex', 'export_tables',
]


# ── BenchmarkAnalysis ─────────────────────────────────────────────────────────────

class BenchmarkAnalysis:
    """Main entry point for analysing a single qebench batch.

    Args:
        batch_dir:    Path to the batch directory (contains runs.csv, config.json).
        output_root:  Root directory for analysis output.
                      Results are written to output_root/<batch-name>/.
                      Defaults to "analysis/" relative to the current directory.

    Example::

        an = BenchmarkAnalysis("results/batch_2026-02-24_14-30-00/")
        an.generate_report()        # run everything
        print(an.overall_summary()) # inspect a table

    Extending
    ---------
    To add a new analysis method:
    1. Implement the logic as a standalone function in the appropriate sub-module.
    2. Add a thin wrapper here (see pattern below).
    3. Call the wrapper in generate_report().
    """

    def __init__(self, batch_dir, output_root: str = 'analysis/'):
        self._batch_dir = Path(batch_dir)
        self._df, self._config = load_batch(self._batch_dir)
        self._output_root = Path(output_root)

    # ── Properties ────────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Full runs DataFrame with derived columns."""
        return self._df

    @property
    def config(self) -> dict:
        """Parsed config.json from the batch directory."""
        return self._config

    @property
    def batch_name(self) -> str:
        """Name of the batch directory (e.g. 'batch_2026-02-24_14-30-00')."""
        return self._batch_dir.name

    @property
    def output_dir(self) -> Path:
        """Root output directory for this batch's analysis."""
        return self._output_root / self.batch_name

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / 'figures'

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / 'tables'

    # ── Summary tables ─────────────────────────────────────────────────────────────

    def overall_summary(self) -> pd.DataFrame:
        """One aggregate row per algorithm."""
        return overall_summary(self._df)

    def summary_by_category(self, metric: str = 'avg_chain_length') -> pd.DataFrame:
        """Algorithm × graph-category matrix of mean `metric`."""
        return summary_by_category(self._df, metric)

    def rank_table(self, metric: str = 'avg_chain_length',
                   lower_is_better: bool = True) -> pd.DataFrame:
        """Mean per-problem rank per algorithm."""
        return rank_table(self._df, metric, lower_is_better)

    # ── Statistical analyses ───────────────────────────────────────────────────────

    def win_rate_matrix(self, metric: str = 'avg_chain_length',
                        lower_is_better: bool = True) -> pd.DataFrame:
        """N×N pairwise win rate table."""
        return win_rate_matrix(self._df, metric, lower_is_better)

    def significance_tests(self, metric: str = 'avg_chain_length') -> pd.DataFrame:
        """Wilcoxon signed-rank p-values for all algorithm pairs."""
        return significance_tests(self._df, metric)

    def friedman_test(self, metric: str = 'avg_chain_length') -> dict:
        """Friedman test across all algorithms simultaneously."""
        return friedman_test(self._df, metric)

    def correlation_matrix(self,
                            graph_props: Optional[List[str]] = None,
                            embed_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Spearman correlation: graph properties × embedding metrics."""
        return correlation_matrix(self._df, graph_props, embed_metrics)

    # ── Plot wrappers ──────────────────────────────────────────────────────────────
    # All plot methods return the Figure and optionally save to figures_dir.

    def plot_heatmap(self, metric: str = 'avg_chain_length',
                     save: bool = True) -> plt.Figure:
        return plot_heatmap(self._df, metric,
                            output_dir=self.figures_dir, save=save)

    def plot_scaling(self, metric: str = 'embedding_time',
                     x: str = 'problem_nodes', log: bool = False,
                     save: bool = True) -> plt.Figure:
        return plot_scaling(self._df, metric, x, log,
                            output_dir=self.figures_dir, save=save)

    def plot_density_hardness(self, metric: str = 'avg_chain_length',
                               save: bool = True) -> plt.Figure:
        return plot_density_hardness(self._df, metric,
                                     output_dir=self.figures_dir, save=save)

    def plot_pareto(self, x: str = 'embedding_time',
                    y: str = 'avg_chain_length',
                    save: bool = True) -> plt.Figure:
        return plot_pareto(self._df, x, y,
                           output_dir=self.figures_dir, save=save)

    def plot_distributions(self, metric: str = 'avg_chain_length',
                            save: bool = True) -> plt.Figure:
        return plot_distributions(self._df, metric,
                                  output_dir=self.figures_dir, save=save)

    def plot_head_to_head(self, algo_a: str, algo_b: str,
                           metric: str = 'avg_chain_length',
                           save: bool = True) -> plt.Figure:
        return plot_head_to_head(self._df, algo_a, algo_b, metric,
                                 output_dir=self.figures_dir, save=save)

    def plot_consistency(self, save: bool = True) -> plt.Figure:
        return plot_consistency(self._df,
                                output_dir=self.figures_dir, save=save)

    def plot_topology_comparison(self, metric: str = 'avg_chain_length',
                                  save: bool = True) -> plt.Figure:
        return plot_topology_comparison(self._df, metric,
                                        output_dir=self.figures_dir, save=save)

    def plot_problem_deep_dive(self, problem_name: str,
                                save: bool = True) -> plt.Figure:
        return plot_problem_deep_dive(self._df, problem_name,
                                      output_dir=self.figures_dir, save=save)

    def plot_chain_distribution(self, save: bool = True) -> plt.Figure:
        return plot_chain_distribution(self._df,
                                       output_dir=self.figures_dir, save=save)

    # ── Export ─────────────────────────────────────────────────────────────────────

    def export_latex(self, output_dir=None) -> None:
        """Write all summary tables as .tex and .csv files."""
        out = Path(output_dir) if output_dir else self.tables_dir
        tables = {
            'overall_summary': (
                self.overall_summary(),
                'Algorithm performance summary',
                'tab:overall_summary',
            ),
            'rank_table_chain': (
                self.rank_table('avg_chain_length'),
                'Algorithm rank by average chain length',
                'tab:rank_chain',
            ),
            'rank_table_time': (
                self.rank_table('embedding_time'),
                'Algorithm rank by embedding time',
                'tab:rank_time',
            ),
            'category_breakdown_chain': (
                self.summary_by_category('avg_chain_length'),
                'Mean average chain length by graph category',
                'tab:category_chain',
            ),
            'category_breakdown_time': (
                self.summary_by_category('embedding_time'),
                'Mean embedding time by graph category',
                'tab:category_time',
            ),
            'win_rate_chain': (
                self.win_rate_matrix('avg_chain_length'),
                'Win rate matrix (avg chain length)',
                'tab:win_rate_chain',
            ),
            'significance_chain': (
                self.significance_tests('avg_chain_length'),
                'Wilcoxon significance tests (avg chain length)',
                'tab:significance_chain',
            ),
        }
        export_tables(tables, out)
        print(f"Tables exported to {out}/")

    # ── Full report ─────────────────────────────────────────────────────────────────

    def generate_report(self, fmt: str = 'png') -> Path:
        """Run all analyses and write outputs to analysis/<batch-name>/.

        Creates:
            figures/   — all plots as .png (or fmt)
            tables/    — all tables as .csv and .tex
            README.md  — index of everything generated

        Args:
            fmt:  Image format for figures ('png', 'pdf', 'svg').

        Returns:
            Path to the output directory.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        algos = sorted(self._df['algorithm'].unique())
        n_algos = len(algos)

        generated_figures = []
        generated_tables  = []

        # ── Plots ──────────────────────────────────────────────────────────────
        plot_tasks = [
            ('Heatmap (chain length)',   lambda: self.plot_heatmap('avg_chain_length')),
            ('Heatmap (time)',           lambda: self.plot_heatmap('embedding_time')),
            ('Scaling: time vs nodes',  lambda: self.plot_scaling('embedding_time', 'problem_nodes')),
            ('Scaling: chain vs nodes', lambda: self.plot_scaling('avg_chain_length', 'problem_nodes')),
            ('Density hardness',        lambda: self.plot_density_hardness()),
            ('Pareto frontier',         lambda: self.plot_pareto()),
            ('Distributions (chain)',   lambda: self.plot_distributions('avg_chain_length')),
            ('Distributions (time)',    lambda: self.plot_distributions('embedding_time')),
            ('Consistency (CV)',        lambda: self.plot_consistency()),
            ('Topology comparison',     lambda: self.plot_topology_comparison()),
            ('Chain distribution',      lambda: self.plot_chain_distribution()),
        ]

        # Head-to-head for all algorithm pairs
        for a, b in __import__('itertools').combinations(algos, 2):
            plot_tasks.append(
                (f'Head-to-head: {a} vs {b}',
                 lambda a=a, b=b: self.plot_head_to_head(a, b))
            )

        # Deep dive for each problem (only if ≤ 10 unique problems to avoid flooding)
        unique_problems = self._df['problem_name'].unique()
        if len(unique_problems) <= 10:
            for prob in unique_problems:
                plot_tasks.append(
                    (f'Deep dive: {prob}',
                     lambda p=prob: self.plot_problem_deep_dive(p))
                )

        for label, fn in plot_tasks:
            try:
                fn()
                generated_figures.append(label)
            except Exception as e:
                print(f"  ⚠  {label}: {e}")

        # ── Tables ─────────────────────────────────────────────────────────────
        try:
            self.export_latex()
            generated_tables = [
                'overall_summary', 'rank_table_chain', 'rank_table_time',
                'category_breakdown_chain', 'category_breakdown_time',
                'win_rate_chain', 'significance_chain',
            ]
        except Exception as e:
            print(f"  ⚠  Table export: {e}")

        # ── README ─────────────────────────────────────────────────────────────
        self._write_readme(generated_figures, generated_tables)

        # ── Summary print ──────────────────────────────────────────────────────
        print(f"\n📊 Analysis complete → {self.output_dir}/")
        print(f"   ├── figures/  ({len(generated_figures)} plots)")
        print(f"   ├── tables/   ({len(generated_tables)} tables × 2 formats)")
        print(f"   └── README.md")

        return self.output_dir

    def _write_readme(self, figures: list, tables: list) -> None:
        """Write a README.md index of all generated files."""
        lines = [
            f'# Analysis: {self.batch_name}\n',
            f'Batch note: {self._config.get("batch_note", "—")}\n',
            f'Algorithms: {", ".join(sorted(self._df["algorithm"].unique()))}\n',
            f'Problems:   {self._df["problem_name"].nunique()}\n',
            f'Topologies: {", ".join(sorted(self._df["topology_name"].dropna().unique()))}\n',
            '\n---\n',
            '## Figures\n',
        ]
        for f in figures:
            lines.append(f'- {f}')
        lines += ['\n## Tables\n']
        for t in tables:
            lines.append(f'- {t}.csv / {t}.tex')

        with open(self.output_dir / 'README.md', 'w') as fh:
            fh.write('\n'.join(lines))

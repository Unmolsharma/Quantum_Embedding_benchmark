"""
qeanalysis/plots.py
====================
All visualizations for qeanalysis.

Design principles
-----------------
- Every function accepts `df` (derived DataFrame from load_batch()) as first arg.
- All functions return the matplotlib Figure object so callers can further
  customise or test without file I/O.
- `save=False` by default; pass `save=True` and `output_dir` to write to disk.
- A single ALGO_PALETTE dict maps algorithm names → colours for consistency
  across all plots in the same report.

Adding new plots
----------------
Write a new standalone function following the same signature:

    def plot_my_analysis(df, ..., output_dir=None, save=False) -> plt.Figure:
        fig, ax = plt.subplots(...)
        # ... your code ...
        _maybe_save(fig, output_dir, "my_analysis.png", save)
        return fig

Then call it from BenchmarkAnalysis.generate_report() and add a method wrapper.
"""

import itertools
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; safe in scripts and tests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ── Palette ─────────────────────────────────────────────────────────────────────

# Seaborn deep palette — up to 10 distinct colours.
# Extended on demand; stable order guarantees consistent colours across plots.
_PALETTE_COLORS = sns.color_palette('tab10', 10)

def _algo_palette(algorithms) -> dict:
    """Return {algo_name: colour} for the given list of algorithms."""
    algos = sorted(set(algorithms))
    return {a: _PALETTE_COLORS[i % len(_PALETTE_COLORS)] for i, a in enumerate(algos)}


# ── Save helper ──────────────────────────────────────────────────────────────────

def _maybe_save(fig: plt.Figure, output_dir, filename: str, save: bool):
    if save and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── 1. Category heatmap ─────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame,
                 metric: str = 'avg_chain_length',
                 output_dir=None,
                 save: bool = False) -> plt.Figure:
    """Heatmap: algorithm (rows) × graph category (columns), cell = mean metric.

    Only successful trials are included.
    """
    from qeanalysis.summary import summary_by_category
    pivot = summary_by_category(df, metric)

    if pivot.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        _maybe_save(fig, output_dir, f'heatmap_{metric}.png', save)
        return fig

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.4),
                                    max(3, pivot.shape[0] * 0.9) + 1))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt='.2f',
        cmap='YlOrRd', linewidths=0.5, linecolor='white',
        cbar_kws={'label': metric.replace('_', ' ')}
    )
    ax.set_title(f'Mean {metric.replace("_", " ")} by algorithm and graph category')
    ax.set_xlabel('Graph category')
    ax.set_ylabel('Algorithm')
    plt.tight_layout()
    _maybe_save(fig, output_dir, f'heatmap_{metric}.png', save)
    return fig


# ── 2. Scaling plot ──────────────────────────────────────────────────────────────

def plot_scaling(df: pd.DataFrame,
                 metric: str = 'embedding_time',
                 x: str = 'problem_nodes',
                 log: bool = False,
                 output_dir=None,
                 save: bool = False) -> plt.Figure:
    """Line plot: metric vs x (aggregated across trials), one line per algorithm.

    Mean ± 1 std shaded ribbon.  Only successful trials included.
    """
    success_df = df[df['success']].copy()
    if success_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No successful trials', ha='center', va='center')
        _maybe_save(fig, output_dir, f'scaling_{metric}_vs_{x}.png', save)
        return fig

    palette = _algo_palette(success_df['algorithm'].unique())
    fig, ax = plt.subplots(figsize=(9, 5))

    for algo, grp in success_df.groupby('algorithm'):
        agg = grp.groupby(x)[metric].agg(['mean', 'std']).reset_index()
        agg['std'] = agg['std'].fillna(0)
        color = palette[algo]
        ax.plot(agg[x], agg['mean'], marker='o', label=algo, color=color, linewidth=2)
        ax.fill_between(agg[x],
                        agg['mean'] - agg['std'],
                        agg['mean'] + agg['std'],
                        alpha=0.15, color=color)

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel(x.replace('_', ' '))
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'Scaling: {metric.replace("_", " ")} vs {x.replace("_", " ")}')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, output_dir, f'scaling_{metric}_vs_{x}.png', save)
    return fig


# ── 3. Density-hardness (random graphs) ─────────────────────────────────────────

def plot_density_hardness(df: pd.DataFrame,
                          metric: str = 'avg_chain_length',
                          output_dir=None,
                          save: bool = False) -> plt.Figure:
    """Line plot: metric vs graph density for random graphs, one line per (algo, n).

    Only graphs with category=='random' are included.
    """
    rand_df = df[(df['category'] == 'random') & df['success']].copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    if rand_df.empty:
        ax.text(0.5, 0.5, 'No random graph data', ha='center', va='center')
        _maybe_save(fig, output_dir, f'density_hardness_{metric}.png', save)
        return fig

    palette = _algo_palette(rand_df['algorithm'].unique())
    linestyles = ['-', '--', '-.', ':']
    n_values = sorted(rand_df['problem_nodes'].unique())

    for algo, algo_grp in rand_df.groupby('algorithm'):
        color = palette[algo]
        for i, n in enumerate(n_values):
            n_grp = algo_grp[algo_grp['problem_nodes'] == n]
            if n_grp.empty:
                continue
            agg = n_grp.groupby('problem_density')[metric].mean().reset_index()
            ls = linestyles[i % len(linestyles)]
            label = f'{algo} (n={n})'
            ax.plot(agg['problem_density'], agg[metric],
                    marker='o', label=label, color=color, linestyle=ls)

    ax.set_xlabel('Graph density')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'Density hardness: {metric.replace("_", " ")} vs density (random graphs)')
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, output_dir, f'density_hardness_{metric}.png', save)
    return fig


# ── 4. Pareto frontier ───────────────────────────────────────────────────────────

def _pareto_front(points: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (minimise both dimensions)."""
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        dominated[i] = False
        is_pareto[dominated] = False
    return is_pareto


def plot_pareto(df: pd.DataFrame,
                x: str = 'embedding_time',
                y: str = 'avg_chain_length',
                output_dir=None,
                save: bool = False) -> plt.Figure:
    """Scatter: one point per (algorithm, problem), Pareto frontier highlighted.

    Uses per-problem mean across successful trials.
    """
    success_df = df[df['success']].copy()
    if success_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No successful trials', ha='center', va='center')
        _maybe_save(fig, output_dir, f'pareto_{x}_vs_{y}.png', save)
        return fig

    agg = success_df.groupby(['algorithm', 'problem_name'])[[x, y]].mean().reset_index()
    palette = _algo_palette(agg['algorithm'].unique())

    fig, ax = plt.subplots(figsize=(9, 6))

    for algo, grp in agg.groupby('algorithm'):
        color = palette[algo]
        ax.scatter(grp[x], grp[y], color=color, label=algo, alpha=0.65, s=40, zorder=3)

    # Pareto frontier across all points
    pts = agg[[x, y]].values
    if len(pts) >= 2:
        mask = _pareto_front(pts)
        front = agg[mask].sort_values(x)
        ax.plot(front[x], front[y], 'k--', linewidth=1.5,
                label='Pareto frontier', zorder=4)

    ax.set_xlabel(x.replace('_', ' '))
    ax.set_ylabel(y.replace('_', ' '))
    ax.set_title(f'Pareto frontier: {x.replace("_"," ")} vs {y.replace("_"," ")}')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, output_dir, f'pareto_{x}_vs_{y}.png', save)
    return fig


# ── 5. Distribution violin ───────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame,
                       metric: str = 'avg_chain_length',
                       output_dir=None,
                       save: bool = False) -> plt.Figure:
    """Violin plot of `metric` per algorithm (successful trials only)."""
    success_df = df[df['success']].copy()
    if success_df.empty or metric not in success_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        _maybe_save(fig, output_dir, f'distribution_{metric}.png', save)
        return fig

    palette = _algo_palette(success_df['algorithm'].unique())
    algos = sorted(success_df['algorithm'].unique())
    colors = [palette[a] for a in algos]

    fig, ax = plt.subplots(figsize=(max(6, len(algos) * 1.5), 5))
    success_df = success_df.copy()
    success_df['_algo_color'] = success_df['algorithm'].map(palette)
    sns.violinplot(
        data=success_df, x='algorithm', y=metric, order=algos,
        hue='algorithm', palette=palette, legend=False,
        ax=ax, inner='box', cut=0
    )
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'Distribution of {metric.replace("_", " ")} per algorithm')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    _maybe_save(fig, output_dir, f'distribution_{metric}.png', save)
    return fig


# ── 6. Head-to-head scatter ──────────────────────────────────────────────────────

def plot_head_to_head(df: pd.DataFrame,
                      algo_a: str,
                      algo_b: str,
                      metric: str = 'avg_chain_length',
                      output_dir=None,
                      save: bool = False) -> plt.Figure:
    """Scatter: per-problem mean metric for algo_a (x) vs algo_b (y).

    Points below the diagonal → algo_a wins (lower is better for most metrics).
    """
    success_df = df[df['success']].copy()
    per_problem = (
        success_df
        .groupby(['algorithm', 'problem_name'])[metric]
        .mean()
        .unstack(level='algorithm')
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    if algo_a not in per_problem.columns or algo_b not in per_problem.columns:
        ax.text(0.5, 0.5, f'Missing data for {algo_a} or {algo_b}',
                ha='center', va='center')
        _maybe_save(fig, output_dir, f'head_to_head_{algo_a}_vs_{algo_b}.png', save)
        return fig

    common = per_problem[[algo_a, algo_b]].dropna()
    if common.empty:
        ax.text(0.5, 0.5, 'No paired problems', ha='center', va='center')
        _maybe_save(fig, output_dir, f'head_to_head_{algo_a}_vs_{algo_b}.png', save)
        return fig

    ax.scatter(common[algo_a], common[algo_b], alpha=0.7, s=50,
               color='steelblue', zorder=3)

    # Diagonal reference
    lo = min(common[algo_a].min(), common[algo_b].min()) * 0.95
    hi = max(common[algo_a].max(), common[algo_b].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='Equal')

    # Win counts
    a_wins = (common[algo_a] < common[algo_b]).sum()
    b_wins = (common[algo_b] < common[algo_a]).sum()
    ax.set_title(f'{algo_a} vs {algo_b}\n{metric.replace("_"," ")}'
                 f'  ({algo_a} better: {a_wins}, {algo_b} better: {b_wins})')
    ax.set_xlabel(f'{algo_a} {metric.replace("_"," ")}')
    ax.set_ylabel(f'{algo_b} {metric.replace("_"," ")}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    _maybe_save(fig, output_dir, f'head_to_head_{algo_a}_vs_{algo_b}.png', save)
    return fig


# ── 7. Consistency (CV) ──────────────────────────────────────────────────────────

def plot_consistency(df: pd.DataFrame,
                     output_dir=None,
                     save: bool = False) -> plt.Figure:
    """Two-panel bar chart: coefficient of variation of time and chain length per algo.

    Lower CV → more consistent.  Computed per (algo, problem) pair, then averaged.
    Only problems with ≥ 2 successful trials contribute.
    """
    success_df = df[df['success']].copy()

    def _mean_cv(metric):
        cv_per_prob = (
            success_df.groupby(['algorithm', 'problem_name'])[metric]
            .agg(lambda s: s.std() / s.mean() if s.mean() != 0 and len(s) >= 2 else np.nan)
        )
        return cv_per_prob.groupby('algorithm').mean()

    cv_time  = _mean_cv('embedding_time')
    cv_chain = _mean_cv('avg_chain_length')

    algos = sorted(set(cv_time.index) | set(cv_chain.index))
    palette = _algo_palette(algos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, cv, title in [
        (ax1, cv_time,  'CV of embedding time'),
        (ax2, cv_chain, 'CV of avg chain length'),
    ]:
        vals = [cv.get(a, np.nan) for a in algos]
        colors = [palette[a] for a in algos]
        bars = ax.bar(algos, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel('Coefficient of variation')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    plt.suptitle('Algorithm consistency (lower CV = more consistent)')
    plt.tight_layout()
    _maybe_save(fig, output_dir, 'consistency_cv.png', save)
    return fig


# ── 8. Topology comparison ───────────────────────────────────────────────────────

def plot_topology_comparison(df: pd.DataFrame,
                              metric: str = 'avg_chain_length',
                              output_dir=None,
                              save: bool = False) -> plt.Figure:
    """Grouped bar chart: metric per (algorithm × topology).

    Meaningful when results span multiple topologies.
    """
    success_df = df[df['success']].copy()
    topologies = sorted(success_df['topology_name'].dropna().unique())
    algos = sorted(success_df['algorithm'].unique())
    palette = _algo_palette(algos)

    agg = (
        success_df
        .groupby(['algorithm', 'topology_name'])[metric]
        .mean()
        .reset_index()
    )

    n_topos = len(topologies)
    n_algos = len(algos)
    width = 0.8 / n_algos
    x = np.arange(n_topos)

    fig, ax = plt.subplots(figsize=(max(6, n_topos * n_algos * 0.7 + 2), 5))

    for i, algo in enumerate(algos):
        vals = []
        for topo in topologies:
            row = agg[(agg['algorithm'] == algo) & (agg['topology_name'] == topo)]
            vals.append(row[metric].values[0] if not row.empty else np.nan)
        offset = (i - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               label=algo, color=palette[algo], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=20, ha='right')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'{metric.replace("_"," ")} by topology and algorithm')
    ax.legend(framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, output_dir, f'topology_comparison_{metric}.png', save)
    return fig


# ── 9. Problem deep dive ─────────────────────────────────────────────────────────

def plot_problem_deep_dive(df: pd.DataFrame,
                            problem_name: str,
                            output_dir=None,
                            save: bool = False) -> plt.Figure:
    """Two-panel bar chart for a single problem: time and chain length per algorithm."""
    prob_df = df[df['problem_name'] == problem_name].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if prob_df.empty:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, f'No data for {problem_name}', ha='center', va='center')
        _maybe_save(fig, output_dir, f'deep_dive_{problem_name}.png', save)
        return fig

    success_df = prob_df[prob_df['success']]
    algos = sorted(prob_df['algorithm'].unique())
    palette = _algo_palette(algos)
    colors = [palette[a] for a in algos]

    for ax, metric, ylabel in [
        (ax1, 'embedding_time',   'Embedding time (s)'),
        (ax2, 'avg_chain_length', 'Avg chain length'),
    ]:
        vals = []
        errs = []
        for algo in algos:
            adf = success_df[success_df['algorithm'] == algo]
            if adf.empty:
                vals.append(np.nan)
                errs.append(0)
            else:
                vals.append(adf[metric].mean())
                errs.append(adf[metric].std() if len(adf) > 1 else 0)

        bars = ax.bar(algos, vals, color=colors, alpha=0.9,
                      yerr=errs, capsize=4)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}\n({problem_name})')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

        # Annotate bars with success/validity
        for bar, algo in zip(bars, algos):
            adf = prob_df[prob_df['algorithm'] == algo]
            n_ok = adf['success'].sum()
            n_valid = adf['is_valid'].sum()
            n_total = len(adf)
            annot = f'{n_ok}/{n_total} ✓'
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h * 1.02, annot,
                        ha='center', va='bottom', fontsize=7)

    plt.suptitle(f'Deep dive: {problem_name}')
    plt.tight_layout()
    fname = f'deep_dive_{problem_name.replace("/", "_")}.png'
    _maybe_save(fig, output_dir, fname, save)
    return fig


# ── 10. Chain length distribution ────────────────────────────────────────────────

def plot_chain_distribution(df: pd.DataFrame,
                             output_dir=None,
                             save: bool = False) -> plt.Figure:
    """Overlaid KDE of avg_chain_length per algorithm (successful trials only)."""
    success_df = df[df['success']].copy()
    if success_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No successful trials', ha='center', va='center')
        _maybe_save(fig, output_dir, 'chain_length_distribution.png', save)
        return fig

    palette = _algo_palette(success_df['algorithm'].unique())
    algos = sorted(success_df['algorithm'].unique())

    fig, ax = plt.subplots(figsize=(8, 4))
    for algo in algos:
        data = success_df[success_df['algorithm'] == algo]['avg_chain_length'].dropna()
        if data.empty or data.std() == 0:
            ax.axvline(data.mean(), label=algo, color=palette[algo], linestyle='--')
        else:
            sns.kdeplot(data, ax=ax, label=algo, color=palette[algo], fill=True, alpha=0.2)

    ax.set_xlabel('Avg chain length')
    ax.set_ylabel('Density')
    ax.set_title('Chain length distribution per algorithm')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _maybe_save(fig, output_dir, 'chain_length_distribution.png', save)
    return fig

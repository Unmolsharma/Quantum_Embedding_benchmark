"""
qeanalysis/loader.py
====================
Load a qebench batch directory into a DataFrame ready for analysis.

Adds derived columns not present in runs.csv:
  category              — graph family inferred from problem_name prefix
  qubit_overhead_ratio  — total_qubits_used / problem_nodes
  coupler_overhead_ratio — total_couplers_used / problem_edges
  max_to_avg_chain_ratio — max_chain_length / avg_chain_length
  is_timeout            — embedding_time >= 0.95 * timeout (from config)
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple


# ── Schema ─────────────────────────────────────────────────────────────────────

_REQUIRED_COLUMNS = frozenset({
    'algorithm', 'problem_name', 'topology_name', 'trial',
    'success', 'is_valid', 'embedding_time',
    'avg_chain_length', 'max_chain_length',
    'total_qubits_used', 'total_couplers_used',
    'problem_nodes', 'problem_edges', 'problem_density',
})

# Special-graph names (not prefix-detectable)
_SPECIAL_GRAPHS = frozenset({'petersen', 'dodecahedral', 'icosahedral'})


# ── Category inference ──────────────────────────────────────────────────────────

def infer_category(problem_name: str) -> str:
    """Return the graph category for a problem_name string.

    Rules (case-insensitive prefix match):
        K<digits>         → complete
        bipartite_*       → bipartite
        grid_*            → grid
        cycle_*           → cycle
        tree_*            → tree
        petersen / dodecahedral / icosahedral → special
        random_*          → random
        anything else     → other
    """
    name = problem_name.strip().lower()
    if name in _SPECIAL_GRAPHS:
        return 'special'
    if name.startswith('k') and len(name) > 1 and name[1:].isdigit():
        return 'complete'
    for prefix, category in [
        ('bipartite_', 'bipartite'),
        ('grid_',      'grid'),
        ('cycle_',     'cycle'),
        ('tree_',      'tree'),
        ('random_',    'random'),
    ]:
        if name.startswith(prefix):
            return category
    return 'other'


# ── Column derivation ───────────────────────────────────────────────────────────

def _derive_columns(df: pd.DataFrame, timeout: float = 60.0) -> pd.DataFrame:
    """Add computed columns to a runs DataFrame (modifies a copy)."""
    df = df.copy()

    # Category
    df['category'] = df['problem_name'].apply(infer_category)

    # Qubit overhead ratio
    df['qubit_overhead_ratio'] = np.where(
        df['problem_nodes'] > 0,
        df['total_qubits_used'] / df['problem_nodes'],
        np.nan
    )

    # Coupler overhead ratio
    df['coupler_overhead_ratio'] = np.where(
        df['problem_edges'] > 0,
        df['total_couplers_used'] / df['problem_edges'],
        np.nan
    )

    # Max-to-avg chain ratio
    df['max_to_avg_chain_ratio'] = np.where(
        df['avg_chain_length'] > 0,
        df['max_chain_length'] / df['avg_chain_length'],
        np.nan
    )

    # Timeout flag (allow 5% tolerance)
    df['is_timeout'] = df['embedding_time'] >= (timeout * 0.95)

    return df


# ── Validation ──────────────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"runs.csv is missing required columns: {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns.tolist())}"
        )


# ── Public API ──────────────────────────────────────────────────────────────────

def load_batch(batch_dir) -> Tuple[pd.DataFrame, Dict]:
    """Load a qebench batch directory into a DataFrame + config dict.

    Args:
        batch_dir: Path (str or Path) to a batch directory produced by
                   qebench.ResultsManager.  Must contain runs.csv and
                   optionally config.json.

    Returns:
        (df, config) where df has all runs.csv columns plus the derived
        columns (category, qubit_overhead_ratio, ...) and config is the
        parsed config.json dict (empty dict if file absent).

    Raises:
        FileNotFoundError: if batch_dir or runs.csv does not exist.
        ValueError: if required columns are missing from runs.csv.
    """
    batch_dir = Path(batch_dir)
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    runs_csv = batch_dir / 'runs.csv'
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found in {batch_dir}")

    # Load config (optional)
    config: Dict = {}
    config_json = batch_dir / 'config.json'
    if config_json.exists():
        with open(config_json, 'r') as f:
            config = json.load(f)

    # Load raw runs
    df = pd.read_csv(runs_csv)

    # Coerce boolean columns (CSV may store as strings)
    for col in ('success', 'is_valid'):
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Validate schema
    _validate_columns(df)

    # Derive computed columns
    timeout = float(config.get('timeout', 60.0))
    df = _derive_columns(df, timeout=timeout)

    return df, config

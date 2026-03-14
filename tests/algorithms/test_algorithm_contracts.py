"""Algorithm Contract Test Suite

Covers: EMBER_developer_guide.md § Algorithm Contract Test Suite
Run before merging any new or modified algorithm wrapper.

Tests every registered Python algorithm against the interface contract:
  - Returns correct dict type on success
  - Returns {'embedding': {}, ...} on failure — not None, not an exception
  - Respects timeout
  - Same seed → identical results on two consecutive runs
  - Does not modify input graphs
  - Produces no stdout
  - Algorithmic counters are non-negative ints and seed-stable
  - version property returns a string

C++ binary algorithms (ATOM, OCT) are tested only for the "not compiled"
failure path. When the binary is present they run as normal Python algorithms
would, but those tests are skipped in standard CI to avoid long runtimes.
"""

import io
import sys
import hashlib
import pytest
import networkx as nx
import dwave_networkx as dnx

from qebench.registry import ALGORITHM_REGISTRY


# ── Shared test fixtures ──────────────────────────────────────────────────────

CHIMERA = dnx.chimera_graph(4)     # 128 qubits, sufficient for small problems
K4 = nx.complete_graph(4)          # always embeds successfully

# Algorithms that invoke C++ subprocesses and are only valid when compiled.
# These are excluded from the main contract parametrization; they have their
# own lightweight test class below.
_BINARY_ALGO_NAMES = {
    'atom',
    'oct-triad', 'oct-triad-reduce',
    'oct-fast-oct', 'oct-fast-oct-reduce',
    'oct-hybrid-oct', 'oct-hybrid-oct-reduce',
    'oct_based',   # alias for oct-triad
}

# All registered algorithms that run entirely in Python
PYTHON_ALGO_NAMES = [
    name for name in ALGORITHM_REGISTRY
    if name not in _BINARY_ALGO_NAMES
]

COUNTER_KEYS = [
    'target_node_visits',
    'cost_function_evaluations',
    'embedding_state_mutations',
    'overlap_qubit_iterations',
]


def _graph_hash(G: nx.Graph) -> str:
    """Stable hash of a graph's node and edge sets."""
    h = hashlib.sha256()
    h.update(str(sorted(G.nodes())).encode())
    h.update(str(sorted(G.edges())).encode())
    return h.hexdigest()


# ── Return-format contract ────────────────────────────────────────────────────

class TestReturnFormat:
    """On success, embed() returns a dict with 'embedding' and 'time' keys."""

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_returns_dict(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        result = algo.embed(K4, CHIMERA, timeout=60.0, seed=0)
        assert isinstance(result, dict), (
            f"{algo_name} must return dict, got {type(result)}"
        )

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_has_embedding_key(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        result = algo.embed(K4, CHIMERA, timeout=60.0, seed=0)
        assert 'embedding' in result, f"{algo_name}: missing 'embedding' key"

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_has_time_key(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        result = algo.embed(K4, CHIMERA, timeout=60.0, seed=0)
        assert 'time' in result, f"{algo_name}: missing 'time' key"
        assert isinstance(result['time'], float), (
            f"{algo_name}: 'time' must be float, got {type(result['time'])}"
        )

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_embedding_chains_are_lists(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        result = algo.embed(K4, CHIMERA, timeout=60.0, seed=0)
        emb = result.get('embedding', {})
        for node, chain in emb.items():
            assert isinstance(chain, list), (
                f"{algo_name}: chain for node {node} must be list, got {type(chain)}"
            )


# ── Failure-format contract ───────────────────────────────────────────────────

class TestFailureFormat:
    """On failure, embed() returns a failure dict — never None, never raises."""

    @staticmethod
    def _impossible_embed(algo_name):
        """Embed K20 into a 2-node path graph — guaranteed to fail."""
        algo = ALGORITHM_REGISTRY[algo_name]
        tiny = nx.path_graph(2)
        return algo.embed(nx.complete_graph(20), tiny, timeout=1.0, seed=0)

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_failure_returns_dict_not_none(self, algo_name):
        result = self._impossible_embed(algo_name)
        assert result is not None, (
            f"{algo_name} returned None on failure — must return failure dict"
        )
        assert isinstance(result, dict), (
            f"{algo_name} failure result must be dict, got {type(result)}"
        )

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_failure_embedding_is_empty_dict(self, algo_name):
        result = self._impossible_embed(algo_name)
        assert 'embedding' in result, (
            f"{algo_name} failure dict missing 'embedding' key"
        )
        assert result['embedding'] == {}, (
            f"{algo_name} failure embedding must be {{}}, got {result['embedding']}"
        )

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_failure_success_flag_is_false(self, algo_name):
        result = self._impossible_embed(algo_name)
        if 'success' in result:
            assert result['success'] is False, (
                f"{algo_name} failure dict has success=True"
            )


# ── Binary algorithms: uncompiled failure path ────────────────────────────────

class TestBinaryAlgorithmFailurePath:
    """When C++ binary is absent, the wrapper must return a failure dict."""

    # charme is also untestable as a live algorithm, include it here
    _UNTESTABLE = sorted(_BINARY_ALGO_NAMES | {'charme'})

    @pytest.mark.parametrize("algo_name", _UNTESTABLE)
    def test_returns_dict_not_none(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        result = algo.embed(K4, CHIMERA, timeout=5.0)
        assert result is not None, f"{algo_name} returned None"
        assert isinstance(result, dict), (
            f"{algo_name} must return dict, got {type(result)}"
        )

    @pytest.mark.parametrize("algo_name", _UNTESTABLE)
    def test_has_embedding_key(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        result = algo.embed(K4, CHIMERA, timeout=5.0)
        assert 'embedding' in result, f"{algo_name}: missing 'embedding' key"


# ── Timeout contract ──────────────────────────────────────────────────────────

class TestTimeout:
    """Algorithms complete within a reasonable bound of the requested timeout."""

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_completes_within_grace_period(self, algo_name):
        import time
        algo = ALGORITHM_REGISTRY[algo_name]
        start = time.perf_counter()
        result = algo.embed(nx.complete_graph(15), CHIMERA, timeout=0.5, seed=0)
        elapsed = time.perf_counter() - start
        # 5× grace factor for Python startup and small graphs
        assert elapsed < 5.0, (
            f"{algo_name} took {elapsed:.2f}s with 0.5s timeout"
        )
        assert result is not None


# ── Seed reproducibility ──────────────────────────────────────────────────────

class TestSeedReproducibility:
    """Same seed → identical embedding on two consecutive calls."""

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_same_seed_same_embedding(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        r1 = algo.embed(K4, CHIMERA, timeout=60.0, seed=42)
        r2 = algo.embed(K4, CHIMERA, timeout=60.0, seed=42)
        assert r1.get('embedding') == r2.get('embedding'), (
            f"{algo_name}: different embeddings with seed=42"
        )


# ── Input immutability ────────────────────────────────────────────────────────

class TestInputImmutability:
    """embed() must not mutate source_graph or target_graph."""

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_graphs_unchanged_after_embed(self, algo_name):
        source = K4.copy()
        target = CHIMERA.copy()
        h_src = _graph_hash(source)
        h_tgt = _graph_hash(target)
        algo = ALGORITHM_REGISTRY[algo_name]
        algo.embed(source, target, timeout=60.0, seed=0)
        assert _graph_hash(source) == h_src, f"{algo_name} mutated source_graph"
        assert _graph_hash(target) == h_tgt, f"{algo_name} mutated target_graph"


# ── No stdout ─────────────────────────────────────────────────────────────────

class TestNoStdout:
    """embed() must not write to stdout."""

    def _capture(self, algo_name, source, target, **kwargs):
        algo = ALGORITHM_REGISTRY[algo_name]
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            algo.embed(source, target, **kwargs)
        finally:
            sys.stdout = old
        return buf.getvalue()

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_no_stdout_on_success(self, algo_name):
        output = self._capture(algo_name, K4, CHIMERA, timeout=60.0, seed=0)
        assert output == "", f"{algo_name} wrote to stdout: {output!r}"

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_no_stdout_on_failure(self, algo_name):
        output = self._capture(
            algo_name, nx.complete_graph(20), nx.path_graph(2),
            timeout=1.0, seed=0
        )
        assert output == "", f"{algo_name} wrote to stdout on failure: {output!r}"


# ── Algorithmic counters ──────────────────────────────────────────────────────

class TestAlgorithmicCounters:
    """Counter fields, when present, must be non-negative ints and seed-stable."""

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_counters_are_nonneg_ints(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        result = algo.embed(K4, CHIMERA, timeout=60.0, seed=0)
        for key in COUNTER_KEYS:
            val = result.get(key)
            if val is not None:
                assert isinstance(val, int) and not isinstance(val, bool), (
                    f"{algo_name}: {key}={val!r} is not a plain int"
                )
                assert val >= 0, f"{algo_name}: {key}={val} is negative"

    @pytest.mark.parametrize("algo_name", PYTHON_ALGO_NAMES)
    def test_counters_stable_across_same_seed_runs(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        r1 = algo.embed(K4, CHIMERA, timeout=60.0, seed=7)
        r2 = algo.embed(K4, CHIMERA, timeout=60.0, seed=7)
        for key in COUNTER_KEYS:
            v1, v2 = r1.get(key), r2.get(key)
            if v1 is not None or v2 is not None:
                assert v1 == v2, (
                    f"{algo_name}: {key} unstable across same-seed runs "
                    f"({v1} vs {v2})"
                )


# ── Version property ──────────────────────────────────────────────────────────

class TestVersionProperty:
    """Every registered algorithm must expose a version property returning str."""

    @pytest.mark.parametrize("algo_name", list(ALGORITHM_REGISTRY.keys()))
    def test_version_returns_string(self, algo_name):
        algo = ALGORITHM_REGISTRY[algo_name]
        v = algo.version
        assert isinstance(v, str), (
            f"{algo_name}.version must return str, got {type(v)}"
        )

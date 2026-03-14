"""
Smoke Benchmark — Algorithm Contract Compliance
================================================
Run this manually after any major architectural change to benchmark_one(),
EmbeddingResult, or algorithm wrappers.

What it verifies:
  1. All edited Python algorithms return the correct EmbeddingResult structure
     (wall_time ≥ 0, valid status string, algorithm_version set)
  2. Successful embeddings are marked is_valid=True, have chains, and status='SUCCESS'
  3. Failed embeds return status in VALID_STATUSES and embedding=None
  4. Seed reproducibility: same seed → identical embedding on two consecutive calls
  5. Binary algorithms (ATOM, OCT) return proper failure dicts when not compiled
  6. EmbeddingResult.to_dict() round-trips cleanly (all expected keys present)

Run with:
    conda run -n minor python tests/smoke_contract_compliance.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import dwave_networkx as dnx

from qebench import benchmark_one, EmbeddingResult
from qebench.registry import ALGORITHM_REGISTRY

# ── Configuration ─────────────────────────────────────────────────────────────

CHIMERA = dnx.chimera_graph(4)   # 128 qubits

GRAPHS = [
    ("K4",        nx.complete_graph(4)),
    ("K6",        nx.complete_graph(6)),
    ("K8",        nx.complete_graph(8)),
    ("ER_n10",    nx.erdos_renyi_graph(10, 0.4, seed=0)),
    ("cycle_12",  nx.cycle_graph(12)),
]

PYTHON_ALGORITHMS = [
    "minorminer",
    "minorminer-aggressive",
    "minorminer-fast",
    "minorminer-chainlength",
    "clique",
]

BINARY_ALGORITHMS = [
    "atom",
    "oct-triad",
    "oct-fast-oct-reduce",
    "charme",
]

SEEDS = [0, 1, 2]

VALID_STATUSES = {"SUCCESS", "INVALID_OUTPUT", "TIMEOUT", "CRASH", "OOM", "FAILURE"}

EXPECTED_RESULT_KEYS = {
    "algorithm", "problem_name", "topology_name", "trial",
    "success", "status", "wall_time", "cpu_time", "is_valid", "embedding",
    "chain_lengths", "max_chain_length", "avg_chain_length",
    "total_qubits_used", "total_couplers_used",
    "problem_nodes", "problem_edges", "problem_density",
    "algorithm_version", "partial", "error", "metadata",
    "target_node_visits", "cost_function_evaluations",
    "embedding_state_mutations", "overlap_qubit_iterations",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

passed = 0
failed = 0
issues = []


def check(condition: bool, label: str):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        issues.append(label)


def section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


# ── CHECK 1: EmbeddingResult structure on Python algorithms ───────────────────

section("CHECK 1: EmbeddingResult structure (Python algorithms)")

for algo in PYTHON_ALGORITHMS:
    ok_runs = 0
    success_runs = 0
    for gname, G in GRAPHS:
        r = benchmark_one(G, CHIMERA, algo, timeout=30.0,
                          problem_name=gname, trial=0, seed=0)

        check(r.wall_time >= 0.0,
              f"{algo}/{gname}: wall_time={r.wall_time} < 0")
        check(r.status in VALID_STATUSES,
              f"{algo}/{gname}: bad status={r.status!r}")
        check(isinstance(r.algorithm_version, str),
              f"{algo}/{gname}: algorithm_version={r.algorithm_version!r}")
        check(isinstance(r.cpu_time, float),
              f"{algo}/{gname}: cpu_time={r.cpu_time!r} not float")
        check(isinstance(r.partial, bool),
              f"{algo}/{gname}: partial={r.partial!r} not bool")

        if r.success:
            success_runs += 1
            check(r.status == "SUCCESS",
                  f"{algo}/{gname}: success=True but status={r.status!r}")
            check(r.is_valid,
                  f"{algo}/{gname}: success=True but is_valid=False")
            check(bool(r.embedding),
                  f"{algo}/{gname}: success=True but embedding empty")
            check(r.avg_chain_length > 0,
                  f"{algo}/{gname}: success but avg_chain_length=0")
        else:
            check(r.embedding is None or r.embedding == {},
                  f"{algo}/{gname}: failed but embedding={r.embedding!r}")

        ok_runs += 1

    print(f"  {algo:30s}  {ok_runs}/{len(GRAPHS)} graphs checked  "
          f"({success_runs} successful embeds)")

# ── CHECK 2: Seed reproducibility ────────────────────────────────────────────

section("CHECK 2: Seed reproducibility (minorminer + minorminer-fast)")

for algo in ["minorminer", "minorminer-fast"]:
    repro_ok = True
    for gname, G in GRAPHS[:3]:
        r1 = benchmark_one(G, CHIMERA, algo, timeout=30.0, seed=42)
        r2 = benchmark_one(G, CHIMERA, algo, timeout=30.0, seed=42)
        if r1.embedding != r2.embedding:
            repro_ok = False
            issues.append(f"{algo}/{gname}: seed=42 not reproducible")
            failed += 1
        else:
            passed += 1
    print(f"  {algo:30s}  {'PASS' if repro_ok else 'FAIL'}")

# ── CHECK 3: Binary algorithm structure ──────────────────────────────────────
# Verifies that binary algorithms return a correctly-structured EmbeddingResult
# whether or not the binary is compiled. When compiled they may succeed;
# when absent they must return a failure dict rather than None or raising.

section("CHECK 3: Binary algorithm EmbeddingResult structure")

for algo in BINARY_ALGORITHMS:
    r = benchmark_one(nx.complete_graph(4), CHIMERA, algo, timeout=10.0)
    check(r.status in VALID_STATUSES,
          f"{algo}: bad status={r.status!r}")
    check(isinstance(r.wall_time, float) and r.wall_time >= 0.0,
          f"{algo}: wall_time={r.wall_time!r}")
    check(isinstance(r.algorithm_version, str),
          f"{algo}: algorithm_version={r.algorithm_version!r}")
    # If failed: embedding must be absent; if succeeded: embedding must exist
    if r.success:
        check(bool(r.embedding), f"{algo}: success=True but embedding empty")
    else:
        check(r.embedding is None or r.embedding == {},
              f"{algo}: failed but has non-empty embedding")
    print(f"  {algo:30s}  status={r.status}  success={r.success}")

# ── CHECK 4: to_dict() round-trip ─────────────────────────────────────────────

section("CHECK 4: to_dict() contains all expected keys")

r = benchmark_one(nx.complete_graph(4), CHIMERA, "minorminer",
                  problem_name="K4", topology_name="chimera_4", trial=0, seed=0)
d = r.to_dict()
missing = EXPECTED_RESULT_KEYS - set(d.keys())
extra   = set(d.keys()) - EXPECTED_RESULT_KEYS

check(not missing, f"to_dict() missing keys: {missing}")
check(not extra,   f"to_dict() unexpected keys: {extra}")

if missing:
    print(f"  MISSING: {missing}")
if extra:
    print(f"  EXTRA:   {extra}")
if not missing and not extra:
    print(f"  All {len(EXPECTED_RESULT_KEYS)} expected keys present.")

if r.success:
    emb_str = d.get("embedding")
    check(isinstance(emb_str, str), "to_dict(): embedding not JSON-serialised")
    try:
        json.loads(emb_str)
        print("  Embedding JSON round-trips cleanly.")
        passed += 1
    except Exception as e:
        failed += 1
        issues.append(f"to_dict() embedding not valid JSON: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
total = passed + failed
print(f"RESULT: {passed}/{total} checks passed")

if issues:
    print(f"\nFAILURES ({len(issues)}):")
    for i in issues:
        print(f"  ✗ {i}")
else:
    print("All checks passed — contract compliance verified.")

print("=" * 70)

sys.exit(0 if failed == 0 else 1)

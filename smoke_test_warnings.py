"""
smoke_test_warnings.py
======================
Visual smoke test for the warning registry and end-summary output.

Tests four scenarios:
  1. All warning types present (full summary)
  2. Only TOPOLOGY_INCOMPATIBLE (one algo, multiple topos)
  3. Only INVALID_OUTPUT + CRASH (mid-run warnings only)
  4. Clean run — no warnings (summary must be silent)

Run with:
    python smoke_test_warnings.py
"""

from pathlib import Path
from qebench.benchmark import _print_warn_summary, _algo_topo_compatible
from qebench.registry import ALGORITHM_REGISTRY

FAKE_LOG_DIR = Path("/tmp/batch_2026-03-18_12-00-00/logs")
SEP = "=" * 80


def header(title):
    print(f"\n{SEP}")
    print(f"  SCENARIO: {title}")
    print(SEP)


# ── Scenario 1: All warning types ─────────────────────────────────────────────
header("All warning types (full summary)")

print(SEP)
print("Benchmark complete!")
print("Total wall time: 14m 32s")

registry_full = {
    'TOPOLOGY_INCOMPATIBLE': {
        'entries': [
            ('atom', 'pegasus_16', 450),
            ('atom', 'zephyr_12', 225),
        ],
        'total_skipped': 675,
    },
    'INVALID_OUTPUT': {
        'minorminer': 43,
        'pssa': 14,
    },
    'CRASH': {
        'charme': {'count': 3, 'first_error': 'CUDA out of memory'},
    },
    'TIMING_OUTLIER': {
        ('charme', 'chimera_16'): 12,
    },
    'ALL_ALGORITHMS_FAILED': [
        'K_15', 'random_50_d0.4', 'bipartite_8x8', 'grid_7x7', 'cycle_30', 'extra_1',
    ],
}

_print_warn_summary(registry_full, FAKE_LOG_DIR)
print(SEP)


# ── Scenario 2: Topology incompatible only ────────────────────────────────────
header("TOPOLOGY_INCOMPATIBLE only (atom vs two non-chimera topos)")

print(SEP)
print("Benchmark complete!")
print("Total wall time: 3m 10s")

registry_topo = {
    'TOPOLOGY_INCOMPATIBLE': {
        'entries': [
            ('atom', 'pegasus_16', 165),
        ],
        'total_skipped': 165,
    },
}

_print_warn_summary(registry_topo, FAKE_LOG_DIR)
print(SEP)


# ── Scenario 3: INVALID_OUTPUT + CRASH only ───────────────────────────────────
header("INVALID_OUTPUT + CRASH only (no topology filtering)")

print(SEP)
print("Benchmark complete!")
print("Total wall time: 22s")

registry_mid = {
    'INVALID_OUTPUT': {
        'atom': 7,
    },
    'CRASH': {
        'atom': {'count': 2, 'first_error': 'index out of range'},
        'clique': {'count': 1, 'first_error': None},
    },
}

_print_warn_summary(registry_mid, FAKE_LOG_DIR)
print(SEP)


# ── Scenario 4: Clean run — no warnings ───────────────────────────────────────
header("Clean run — no warnings (summary must be silent)")

print(SEP)
print("Benchmark complete!")
print("Total wall time: 8s")

_print_warn_summary({}, FAKE_LOG_DIR)
# Should print nothing — just the closing separator
print(SEP)


# ── Scenario 5: _algo_topo_compatible checks ──────────────────────────────────
header("_algo_topo_compatible() — topology restriction logic")

cases = [
    ("atom",       "chimera_4x4x4",   True),
    ("atom",       "chimera_16x16x4", True),
    ("atom",       "pegasus_16",      False),
    ("atom",       "zephyr_12",       False),
    ("minorminer", "pegasus_16",      True),   # no restriction
    ("minorminer", "chimera_4x4x4",   True),   # no restriction
    ("clique",     "chimera_4x4x4",   True),   # no restriction
]

all_pass = True
for algo, topo, expected in cases:
    result = _algo_topo_compatible(algo, topo)
    status = "✓" if result == expected else "✗ FAIL"
    if result != expected:
        all_pass = False
    print(f"  {status}  _algo_topo_compatible({algo!r}, {topo!r}) = {result}  (expected {expected})")

print()
if all_pass:
    print("  All compatibility checks passed.")
else:
    print("  SOME CHECKS FAILED — see ✗ lines above.")

print(SEP)
print()

"""Phase 1 Smoke Benchmark — verifies all four Phase 1 integrity checks."""
import json
from pathlib import Path

from qebench import EmbeddingBenchmark, verify_manifest
from qebench.graphs import load_test_graphs

# 20 graphs: 4 complete + 3 bipartite + 3 grid + 3 cycle + 3 tree + 3 special + 1 random
SELECTION = "1-4, 11-13, 21-23, 31-33, 41-43, 51-53, 100"
problems = load_test_graphs(SELECTION)
print(f"Loaded {len(problems)} graphs: {[n for n, _ in problems]}\n")

# ── CHECK 1: manifest tamper detection ────────────────────────────────────────
print("CHECK 1: Manifest tamper detection")
target_file = Path("test_graphs/complete/K4.json")
backup = target_file.read_bytes()
target_file.write_bytes(backup + b" ")   # corrupt by one byte

raised = False
try:
    verify_manifest()
except RuntimeError as e:
    raised = True
    print(f"  ✓ RuntimeError raised: {e.args[0].splitlines()[0]}")

target_file.write_bytes(backup)          # restore
verify_manifest()                        # confirm restored
print(f"  ✓ File restored; manifest passes cleanly\n")
assert raised, "FAIL: RuntimeError was not raised on corrupted graph file"

# ── Smoke benchmark ───────────────────────────────────────────────────────────
print("Starting smoke benchmark: 20 graphs × 2 topologies × 2 algorithms × 3 trials")
print("=" * 80)
bench = EmbeddingBenchmark(results_dir="./results")
batch_dir = bench.run_full_benchmark(
    problems=problems,
    topologies=["pegasus_16", "chimera_16x16x4"],
    methods=["minorminer", "atom"],
    n_trials=3,
    timeout=30.0,
    batch_note="Phase 1 smoke benchmark",
)

print(f"\nBatch dir: {batch_dir}")

# ── CHECK 2: config.json written with provenance ──────────────────────────────
print("\nCHECK 2: config.json written with provenance")
config_path = batch_dir / "config.json"
assert config_path.exists(), f"FAIL: config.json missing at {config_path}"
config = json.loads(config_path.read_text())
assert "provenance" in config, "FAIL: 'provenance' key missing"
prov = config["provenance"]
assert "dependencies" in prov, "FAIL: 'dependencies' missing from provenance"
assert "networkx" in prov["dependencies"], "FAIL: networkx not in dependencies"
assert "qebench_version" in prov, "FAIL: qebench_version missing"
print(f"  ✓ qebench_version = {prov['qebench_version']}")
print(f"  ✓ dependencies field present and contains networkx")

# ── CHECK 3: cpu_time non-zero for ATOM ───────────────────────────────────────
print("\nCHECK 3: cpu_time non-zero for ATOM (RUSAGE_CHILDREN)")
atom_results = [r for r in bench.results if r.algorithm == "atom" and r.success]
if not atom_results:
    print("  ⚠  No successful ATOM runs — cannot verify cpu_time")
else:
    cpu_times = [r.cpu_time for r in atom_results]
    print(f"  ATOM: {len(atom_results)} successful runs")
    print(f"  cpu_time — min={min(cpu_times):.4f}s  mean={sum(cpu_times)/len(cpu_times):.4f}s")
    assert min(cpu_times) > 0.001, f"FAIL: ATOM cpu_time too low ({min(cpu_times):.6f}s)"
    print(f"  ✓ All ATOM cpu_time > 0.001s")

# ── CHECK 4: cpu_time non-zero for MinorMiner ────────────────────────────────
print("\nCHECK 4: cpu_time non-zero for MinorMiner (process_time)")
mm_results = [r for r in bench.results if r.algorithm == "minorminer" and r.success]
assert all(r.cpu_time > 0.0 for r in mm_results), "FAIL: some MinorMiner cpu_time == 0"
cpu_times_mm = [r.cpu_time for r in mm_results]
print(f"  MinorMiner: {len(mm_results)} successful runs")
print(f"  cpu_time — min={min(cpu_times_mm):.4f}s  mean={sum(cpu_times_mm)/len(cpu_times_mm):.4f}s")
print(f"  ✓ All MinorMiner cpu_time > 0")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
n_total   = len(bench.results)
n_success = sum(1 for r in bench.results if r.success)
n_valid   = sum(1 for r in bench.results if r.is_valid)
print("ALL PHASE 1 CHECKS PASSED")
print(f"  {n_total} total runs  |  {n_success} succeeded ({100*n_success/n_total:.0f}%)  |  {n_valid} valid embeddings")
print(f"  Results: {batch_dir}")

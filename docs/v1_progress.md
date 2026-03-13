# Version 1 Progress Log

Reverse-chronological. One entry per session or logical unit of work.

---

**2026-03-13 — Phase 1 smoke benchmark**
Ran 20 graphs × 2 topologies × 2 algorithms × 3 trials (240 runs). All four Phase 1 integrity checks confirmed: manifest tamper detection fires on 1-byte corruption, config.json written before runs start, ATOM cpu_time > 0 via RUSAGE_CHILDREN, MinorMiner cpu_time > 0 via process_time. 240/240 succeeded. Script: `smoke_phase1.py`.

**2026-03-13 — Hardware-agnostic algorithmic operation counters**
Added four optional `int` fields to `EmbeddingResult`: `target_node_visits`, `cost_function_evaluations`, `embedding_state_mutations`, `overlap_qubit_iterations`. All default to `None`; algorithms populate whichever they can instrument. Each increment must be a bare `+= 1`. Documented in `EMBER_developer_guide.md` with applicability table, constraints, and example.

**2026-03-13 — Phase 1: CPU timing, SHA-256 manifest, environment provenance (1.1–1.3)**
- **1.1 CPU time:** `_uses_subprocess` flag on ATOM/OCT triggers `RUSAGE_CHILDREN` measurement; Python algorithms use `process_time`. `cpu_time` field on `EmbeddingResult`.
- **1.2 Manifest:** `generate_manifest()` hashes all graph JSONs to `test_graphs/manifest.sha256`; `verify_manifest()` raises `RuntimeError` on any mismatch; called automatically at benchmark startup.
- **1.3 Provenance:** `__version__ = "0.5.0"` added; `qebench_version` + pip freeze written to `config.json`; batch directory created before runs start so provenance survives crashes.

**2026-03-13 — Developer guide and algorithm documentation**
Added `EMBER_developer_guide.md` (team split, algorithm interface contract, vendoring policy, coding standards, testing strategy) and `docs/adding_algorithms.md` / `docs/adding_test_graphs.md` (contributor how-tos).

**2026-03-13 — PSSA integration**
Cloned PSSA D-Wave implementation into `algorithms/pssa_dwave/`; installed as editable package (`pip install -e`). Four variants registered: `pssa`, `pssa-weighted`, `pssa-fast`, `pssa-thorough`.

**2026-03-13 — README and algorithm status update**
`oct-fast-oct-reduce` marked as recommended OCT variant; `oct-triad-reduce` corrected to warning status (produces invalid embeddings on non-bipartite graphs). PSSA marked working. CHARME correctly noted as requiring a trained model and PyTorch.

**2026-03-13 — MinorMiner variants**
Added three additional registered MinorMiner variants: `minorminer-aggressive` (tries=50), `minorminer-fast` (tries=3), `minorminer-chainlength` (chainlength_patience=20).

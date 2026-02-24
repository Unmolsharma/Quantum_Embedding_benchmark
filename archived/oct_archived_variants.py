"""
Archived OCT Algorithm Variants
================================

These OCT-suite algorithm variants were removed from the active registry
because they crash (segfault) or produce invalid embeddings with the
current compiled binary.

They can be re-enabled once the C++ source is debugged and recompiled.

Status at time of archival (2026-02-23):
  - oct-oct:              Runs but produces 1-qubit-per-node mapping (fails validation)
  - fast-oct:             Segfault (returncode -11) even with -s/-r flags
  - fast-oct-reduce:      Same as fast-oct
  - hybrid-oct:           Segfault (returncode -11)
  - hybrid-oct-reduce:    Same as hybrid-oct

To re-enable any of these, move the config entry back to _OCT_CONFIGS
in qebench/registry.py:

    _OCT_CONFIGS = {
        ...
        'oct':               ([], 'OCT-Embed — basic deterministic, 1 qubit/node'),
        'fast-oct':          (['-s', '42', '-r', '100'], 'Fast-OCT — randomized with seed/repeats'),
        'fast-oct-reduce':   (['-s', '42', '-r', '100'], 'Reduced Fast-OCT'),
        'hybrid-oct':        (['-s', '42', '-r', '100'], 'Hybrid-OCT — combined approach'),
        'hybrid-oct-reduce': (['-s', '42', '-r', '100'], 'Reduced Hybrid-OCT'),
    }
"""

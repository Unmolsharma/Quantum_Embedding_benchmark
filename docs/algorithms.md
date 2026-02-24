# Algorithm Registry

QEBench uses a plugin system for embedding algorithms. All algorithms implement `EmbeddingAlgorithm` and are auto-registered via `@register_algorithm("name")`.

## Working Algorithms

### `minorminer`
**D-Wave MinorMiner** — industry-standard heuristic embedding.

- **Type:** Heuristic, randomized
- **Source:** `pip install minorminer` (D-Wave)
- **Paper:** Cai, Macready & Roy (2014), "A practical heuristic for finding graph minors"
- **Strengths:** Fast, general-purpose, works on any topology (Chimera, Pegasus, Zephyr)
- **Weaknesses:** Non-deterministic, quality varies between runs

```python
result = benchmark_one(source, target, "minorminer")
```

---

### `clique`
**D-Wave Clique Embedding** — topology-aware deterministic baseline.

- **Type:** Deterministic, topology-native
- **Source:** `minorminer.busclique.find_clique_embedding`
- **Strengths:** Very fast (sub-millisecond), deterministic, exploits known topology structure
- **Weaknesses:** Higher qubit overhead — embeds into clique substructure rather than optimizing per-problem
- **Note:** Works best on D-Wave native topologies (Chimera, Pegasus, Zephyr)

```python
result = benchmark_one(source, target, "clique")
```

---

### `oct-triad`
**TRIAD** — deterministic OCT-based embedding using biclique virtual hardware.

- **Type:** Deterministic
- **Source:** C++ binary (`algorithms/oct_based/embedding/driver`)
- **Paper:** Goodrich, Sullivan & Humble (2018), "Optimizing adiabatic quantum program compilation using a graph-theoretic framework"
- **Chimera only:** Requires Chimera topology
- **Strengths:** Deterministic, handles dense graphs, guaranteed 2 qubits/node
- **Weaknesses:** Chimera-only, typically higher qubit usage than minorminer

```python
result = benchmark_one(source, chimera_graph, "oct-triad")
```

---

### `oct-triad-reduce`
**Reduced TRIAD** — TRIAD with chain reduction post-processing.

- **Type:** Deterministic
- **Same as `oct-triad`** but applies reduction subroutines to minimize chain lengths after initial embedding
- **Typically produces same or better chains** than plain TRIAD

---

### `oct-fast-oct`
**Fast-OCT** — randomized OCT decomposition with repeated restarts.

- **Type:** Randomized (seed=42, runs=100)
- **Paper:** Goodrich, Sullivan & Humble (2018)
- **Chimera only**
- **Strengths:** Often produces the best embedding quality among OCT variants
- **Note:** Uses greedy randomized OCT approximation with 100 restarts

---

### `oct-fast-oct-reduce`
**Reduced Fast-OCT** — Fast-OCT with chain reduction.

- **Best quality** among OCT-suite algorithms — combines randomized OCT with reduction

---

## Partially Working

### `oct-hybrid-oct` / `oct-hybrid-oct-reduce`
**Hybrid-OCT** — combined deterministic + randomized approach.

- **Status:** Runs but frequently produces invalid embeddings on non-bipartite graphs
- **Works correctly** on bipartite source graphs
- **Chimera only**

### `atom`
**ATOM** — grows its own Chimera topology dynamically.

- **Status:** Binary runs but only outputs timing, not the actual chain mapping
- **Fix needed:** Modify C++ source to write embedding to file
- **Source:** C++ binary (`algorithms/atom/main`)

### `charme`
**CHARME** — reinforcement learning-based embedding.

- **Status:** Stub — `embed()` returns None
- **Fix needed:** Integrate Python modules (`charme/env.py`, `charme/models.py`, `charme/ppo.py`)
- **Paper:** RL-based minor embedding (2025)
- **Source:** `algorithms/charme/`

---

## Adding a New Algorithm

```python
from qebench.registry import register_algorithm, EmbeddingAlgorithm

@register_algorithm("my_algorithm")
class MyAlgorithm(EmbeddingAlgorithm):
    """My custom embedding algorithm."""
    
    def embed(self, source_graph, target_graph, timeout=60.0, **kwargs):
        # Your embedding logic here
        embedding = {node: [physical_qubits] for node in source_graph.nodes()}
        elapsed = ...
        return {'embedding': embedding, 'time': elapsed}
        # Return None if embedding fails
```

## Listing Algorithms

```python
from qebench import list_algorithms, ALGORITHM_REGISTRY

print(list_algorithms())
# ['atom', 'charme', 'clique', 'minorminer', 'oct-fast-oct', ...]

# Get algorithm details
algo = ALGORITHM_REGISTRY["minorminer"]
print(algo.description)
```

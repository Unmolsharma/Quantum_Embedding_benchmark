"""
Algorithm Registry for Minor Embedding Methods

Provides a plugin system for embedding algorithms:
- EmbeddingAlgorithm: Abstract base class all algorithms implement
- @register_algorithm: Decorator to register new algorithms
- ALGORITHM_REGISTRY: Dict[str, EmbeddingAlgorithm] of all registered algorithms

Adding a new algorithm:
    @register_algorithm("my_algo")
    class MyAlgorithm(EmbeddingAlgorithm):
        def embed(self, source_graph, target_graph, timeout=60.0, **kwargs):
            ...
            return {'embedding': {node: [qubits]}, 'time': elapsed}
"""

import time
import os
import subprocess
import tempfile
import networkx as nx
import dwave_networkx as dnx
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# REGISTRY
# ==============================================================================

ALGORITHM_REGISTRY: Dict[str, 'EmbeddingAlgorithm'] = {}


def register_algorithm(name: str):
    """Decorator to register an embedding algorithm.
    
    Usage:
        @register_algorithm("my_algo")
        class MyAlgorithm(EmbeddingAlgorithm):
            ...
    """
    def decorator(cls):
        instance = cls()
        ALGORITHM_REGISTRY[name] = instance
        return cls
    return decorator


def list_algorithms() -> List[str]:
    """Return list of all registered algorithm names."""
    return list(ALGORITHM_REGISTRY.keys())


def get_algorithm(name: str) -> Optional['EmbeddingAlgorithm']:
    """Get a registered algorithm by name."""
    return ALGORITHM_REGISTRY.get(name)


# ==============================================================================
# BASE CLASS
# ==============================================================================

class EmbeddingAlgorithm(ABC):
    """Abstract base class for embedding algorithms.
    
    Subclass this and implement `embed()` to add a new algorithm.
    Use `@register_algorithm("name")` to auto-register it.
    """
    
    @abstractmethod
    def embed(self, source_graph: nx.Graph, target_graph: nx.Graph,
              timeout: float = 60.0, **kwargs) -> Optional[Dict]:
        """Find a minor embedding of source_graph into target_graph.
        
        Args:
            source_graph: The logical graph to embed.
            target_graph: The hardware graph (e.g., Chimera, Pegasus).
            timeout: Maximum wall-clock seconds.
            **kwargs: Algorithm-specific parameters.
        
        Returns:
            Dict with at minimum:
                'embedding': Dict[int, List[int]] mapping logical nodes to physical chains
                'time': float elapsed seconds
            Or None if embedding fails.
        """
        pass
    
    @property
    def description(self) -> str:
        """Short human-readable description of the algorithm."""
        return self.__class__.__doc__ or self.__class__.__name__


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_embedding(embedding: Dict[int, list], source_graph: nx.Graph, 
                       target_graph: nx.Graph) -> bool:
    """Validate that an embedding is a correct minor embedding.
    
    Checks:
        1. All source nodes present in embedding keys
        2. All chains non-empty
        3. All physical qubits exist in target_graph
        4. No chain overlap (disjointness)
        5. Each chain is connected in target_graph
        6. Every source edge is represented by at least one target edge between chains
    """
    try:
        if set(embedding.keys()) != set(source_graph.nodes()):
            return False
        
        if any(not chain for chain in embedding.values()):
            return False
        
        all_target_nodes = set()
        for chain in embedding.values():
            all_target_nodes.update(chain)
        
        if not all_target_nodes.issubset(set(target_graph.nodes())):
            return False
        
        if len(all_target_nodes) != sum(len(chain) for chain in embedding.values()):
            return False
        
        for chain in embedding.values():
            if len(chain) > 1:
                chain_subgraph = target_graph.subgraph(chain)
                if not nx.is_connected(chain_subgraph):
                    return False
        
        for u, v in source_graph.edges():
            chain_u = set(embedding[u])
            chain_v = set(embedding[v])
            if not any(target_graph.has_edge(a, b) for a in chain_u for b in chain_v):
                return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


# ==============================================================================
# CHIMERA HELPERS
# ==============================================================================

def infer_chimera_dims(target_graph: nx.Graph) -> Optional[Tuple[int, int, int]]:
    """Try to infer Chimera dimensions (m, n, t) from a target graph."""
    try:
        graph_data = target_graph.graph
        if 'rows' in graph_data and 'columns' in graph_data and 'tile' in graph_data:
            return (graph_data['rows'], graph_data['columns'], graph_data['tile'])
        
        num_nodes = target_graph.number_of_nodes()
        for m in range(1, 20):
            for t in [4, 8]:
                if num_nodes == 2 * t * m * m:
                    candidate = dnx.chimera_graph(m, m, t)
                    if (candidate.number_of_nodes() == num_nodes and
                            candidate.number_of_edges() == target_graph.number_of_edges()):
                        return (m, m, t)
    except Exception:
        pass
    return None


# ==============================================================================
# BUILT-IN ALGORITHMS
# ==============================================================================

@register_algorithm("minorminer")
class MinorMinerAlgorithm(EmbeddingAlgorithm):
    """D-Wave minorminer — industry-standard heuristic embedding."""
    
    def embed(self, source_graph, target_graph, timeout=60.0, **kwargs):
        try:
            import minorminer
            
            start_time = time.time()
            embedding = minorminer.find_embedding(
                list(source_graph.edges()),
                list(target_graph.edges()),
                timeout=timeout,
                verbose=0
            )
            elapsed = time.time() - start_time
            
            if not embedding:
                return None
            
            return {'embedding': embedding, 'time': elapsed}
        except Exception as e:
            print(f"minorminer error: {e}")
            return None


@register_algorithm("clique")
class CliqueEmbeddingAlgorithm(EmbeddingAlgorithm):
    """D-Wave clique embedding — topology-aware deterministic baseline."""
    
    def embed(self, source_graph, target_graph, timeout=60.0, **kwargs):
        try:
            from minorminer.busclique import find_clique_embedding
            
            start_time = time.time()
            # find_clique_embedding takes number of nodes for a complete graph,
            # or a source graph directly
            raw = find_clique_embedding(source_graph, target_graph)
            elapsed = time.time() - start_time
            
            if not raw:
                return None
            
            # busclique returns tuples — convert to lists for consistency
            embedding = {k: list(v) for k, v in raw.items()}
            
            return {'embedding': embedding, 'time': elapsed}
        except Exception as e:
            print(f"clique embedding error: {e}")
            return None


@register_algorithm("atom")
class AtomAlgorithm(EmbeddingAlgorithm):
    """ATOM — grows its own Chimera topology dynamically."""
    
    def embed(self, source_graph, target_graph, timeout=60.0, **kwargs):
        try:
            atom_dir = Path("./algorithms/atom")
            atom_exe = atom_dir / "main"
            
            if not atom_exe.exists():
                print("⚠️  ATOM not compiled. Run: cd algorithms/atom && make")
                return None
            
            atom_exe = atom_exe.resolve()
            
            # Write source graph in ATOM format:
            # Line 1: number of nodes
            # Lines 2..n+1: node orderings (0, 1, ..., n-1)
            # Remaining lines: edges
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                n = source_graph.number_of_nodes()
                f.write(f"{n}\n")
                for node in range(n):
                    f.write(f"{node}\n")
                for u, v in source_graph.edges():
                    f.write(f"{u} {v}\n")
                source_file = f.name
            
            start_time = time.time()
            try:
                # capture_output=True captures stdout silently (no flooding)
                result = subprocess.run(
                    [str(atom_exe), '-pfile', source_file, '-test', '0'],
                    capture_output=True, text=True,
                    timeout=timeout, cwd=atom_dir
                )
                elapsed = time.time() - start_time
                
                os.unlink(source_file)
                results_txt = atom_dir / "Results.txt"
                if results_txt.exists():
                    os.unlink(results_txt)
                
                # Parse embedding from stdout
                # ATOM's print() outputs: "Embedding:" header, then
                # "x y k color" per qubit, then "Requires N qubits"
                # Each EB_Point has (x, y, k) = Chimera position, color = logical node
                embedding = {}
                topo_column = 0
                
                for line in result.stdout.strip().split('\n'):
                    line = line.strip()
                    if not line or line.startswith('Embedding') or line.startswith('Requires'):
                        continue
                    parts = line.split()
                    if len(parts) == 4:
                        try:
                            x, y, k, color = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                            topo_column = max(topo_column, y + 1)
                            if color not in embedding:
                                embedding[color] = []
                            embedding[color].append((x, y, k))
                        except ValueError:
                            continue
                
                if not embedding:
                    return None
                
                # Convert (x, y, k) tuples to linear Chimera qubit indices
                # Index = x * topo_column * 8 + y * 8 + k
                linear_embedding = {}
                for color, positions in embedding.items():
                    linear_embedding[color] = [
                        x * topo_column * 8 + y * 8 + k
                        for x, y, k in positions
                    ]
                
                return {
                    'embedding': linear_embedding,
                    'time': elapsed,
                    'method': 'ATOM',
                }
            except subprocess.TimeoutExpired:
                os.unlink(source_file)
                return None
        except Exception as e:
            print(f"ATOM error: {e}")
            return None


@register_algorithm("charme")
class CharmeAlgorithm(EmbeddingAlgorithm):
    """CHARME — Python RL framework. Not callable via subprocess."""
    
    def embed(self, source_graph, target_graph, timeout=60.0, **kwargs):
        print("⚠️  CHARME is a Python RL framework, not a standalone binary.")
        print("   Import its Python modules directly to use it.")
        return None


def _make_oct_algorithm_class(algo_name: str, extra_flags: list, desc: str):
    """Factory to create and register an OCT-suite algorithm class."""
    
    class OctVariant(EmbeddingAlgorithm):
        __doc__ = desc
        
        def embed(self, source_graph, target_graph, timeout=60.0,
                  chimera_dims=None, **kwargs):
            try:
                oct_dir = Path("./algorithms/oct_based").resolve()
                oct_exe = oct_dir / "embedding" / "driver"
                
                if not oct_exe.exists():
                    print("⚠️  OCT-Based not compiled. Run: cd algorithms/oct_based && make")
                    return None
                
                # Determine Chimera dimensions
                dims = chimera_dims or infer_chimera_dims(target_graph) or (4, 4, 4)
                c_m, c_n, c_t = dims
                chimera_graph = dnx.chimera_graph(c_m, c_n, c_t)
                
                # Write source graph in OCT format:
                # Line 1: number of nodes
                # Lines 2..n+1: node orderings (0, 1, ..., n-1)
                # Remaining lines: edges
                with tempfile.NamedTemporaryFile(mode='w', suffix='.graph',
                                                 dir=str(oct_dir), delete=False) as f:
                    n = source_graph.number_of_nodes()
                    f.write(f"{n}\n")
                    for node in range(n):
                        f.write(f"{node}\n")
                    for u, v in source_graph.edges():
                        f.write(f"{u} {v}\n")
                    source_file = f.name
                
                out_base = tempfile.mktemp(dir=str(oct_dir), prefix="oct_out_")
                
                start_time = time.time()
                try:
                    cmd = [str(oct_exe), '-a', algo_name,
                           '-pfile', source_file,
                           '-c', str(c_t), '-m', str(c_m), '-n', str(c_n),
                           '-o', out_base] + extra_flags
                    
                    subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=timeout, cwd=str(oct_dir))
                    
                    elapsed = time.time() - start_time
                    
                    emb_file = out_base + ".embedding"
                    embedding = {}
                    if os.path.exists(emb_file) and os.path.getsize(emb_file) > 0:
                        with open(emb_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if ':' in line:
                                    lp, pp = line.split(':', 1)
                                    pp = pp.strip()
                                    if pp:
                                        if ',' in pp:
                                            chain = [int(x.strip()) for x in pp.split(',') if x.strip()]
                                        else:
                                            chain = [int(x) for x in pp.split() if x]
                                        if chain:
                                            embedding[int(lp.strip())] = chain
                    
                    # Cleanup
                    for p in [source_file, emb_file, out_base + ".timing", out_base]:
                        if os.path.exists(p):
                            os.unlink(p)
                    
                    if not embedding:
                        return None
                    
                    return {
                        'embedding': embedding,
                        'time': elapsed,
                        'chimera_dims': dims,
                        'chimera_graph': chimera_graph,
                        'algorithm': algo_name
                    }
                except subprocess.TimeoutExpired:
                    for p in [source_file, out_base + ".embedding", out_base + ".timing", out_base]:
                        if os.path.exists(p):
                            os.unlink(p)
                    return None
            except Exception as e:
                print(f"OCT-{algo_name} error: {e}")
                return None
    
    OctVariant.__name__ = f"Oct_{algo_name.replace('-', '_')}"
    OctVariant.__qualname__ = OctVariant.__name__
    return OctVariant


# Register all OCT-suite algorithms
_OCT_CONFIGS = {
    'triad':             ([], 'TRIAD — deterministic, 2 qubits/node, handles dense graphs'),
    'triad-reduce':      ([], 'Reduced TRIAD — TRIAD with chain reduction'),
    'fast-oct':          (['-s', '42', '-r', '100'], 'Fast-OCT — randomized with seed/repeats'),
    'fast-oct-reduce':   (['-s', '42', '-r', '100'], 'Reduced Fast-OCT'),
    'hybrid-oct':        (['-s', '42', '-r', '100'], 'Hybrid-OCT — combined approach'),
    'hybrid-oct-reduce': (['-s', '42', '-r', '100'], 'Reduced Hybrid-OCT'),
}

for _name, (_flags, _desc) in _OCT_CONFIGS.items():
    _cls = _make_oct_algorithm_class(_name, _flags, _desc)
    register_algorithm(f"oct-{_name}")(_cls)

# Also register "oct_based" as an alias for oct-triad (the most reliable default)
ALGORITHM_REGISTRY["oct_based"] = ALGORITHM_REGISTRY["oct-triad"]


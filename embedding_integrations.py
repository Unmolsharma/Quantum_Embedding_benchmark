"""
Integration Module for Minor Embedding Methods
Integrates ATOM, CHARME, and OCT-Based (all C++ implementations called via subprocess)
"""

import sys
import time
import os
import subprocess
import tempfile
import networkx as nx
from pathlib import Path
from typing import Dict, Optional

# Add implementation directories to Python path
IMPLEMENTATIONS_DIR = Path("./implementations")
sys.path.insert(0, str(IMPLEMENTATIONS_DIR / "atom"))
sys.path.insert(0, str(IMPLEMENTATIONS_DIR / "charme"))
sys.path.insert(0, str(IMPLEMENTATIONS_DIR / "oct_based"))


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def networkx_to_adjacency_dict(G: nx.Graph) -> Dict[int, list]:
    """Convert NetworkX graph to adjacency dictionary format"""
    return {node: list(G.neighbors(node)) for node in G.nodes()}


def validate_embedding(embedding: Dict[int, list], source_graph: nx.Graph, 
                       target_graph: nx.Graph) -> bool:
    """Validate that an embedding is correct"""
    try:
        # All source nodes present
        if set(embedding.keys()) != set(source_graph.nodes()):
            return False
        
        # All chains non-empty
        if any(not chain for chain in embedding.values()):
            return False
        
        # All target nodes valid
        all_target_nodes = set()
        for chain in embedding.values():
            all_target_nodes.update(chain)
        
        if not all_target_nodes.issubset(set(target_graph.nodes())):
            return False
        
        # Chains don't overlap
        if len(all_target_nodes) != sum(len(chain) for chain in embedding.values()):
            return False
        
        # Chains are connected
        for chain in embedding.values():
            if len(chain) > 1:
                chain_subgraph = target_graph.subgraph(chain)
                if not nx.is_connected(chain_subgraph):
                    return False
        
        # Edges preserved
        for u, v in source_graph.edges():
            chain_u = set(embedding[u])
            chain_v = set(embedding[v])
            
            has_connection = any(
                target_graph.has_edge(node_u, node_v)
                for node_u in chain_u
                for node_v in chain_v
            )
            
            if not has_connection:
                return False
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


# ==============================================================================
# EMBEDDING METHODS
# ==============================================================================

def run_atom_embedding(source_graph: nx.Graph, target_graph: nx.Graph, 
                       timeout: float = 60.0) -> Optional[Dict]:
    """Run ATOM embedding (C++ executable via subprocess)"""
    try:
        atom_dir = Path("./implementations/atom")
        atom_exe = atom_dir / "atom"
        if not atom_exe.exists():
            atom_exe = atom_dir / "atom.exe"
        
        if not atom_exe.exists():
            print("⚠️  ATOM not compiled. Run: cd implementations/atom && make")
            return None
        
        # Write graphs to temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{source_graph.number_of_nodes()} {source_graph.number_of_edges()}\n")
            for u, v in source_graph.edges():
                f.write(f"{u} {v}\n")
            source_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{target_graph.number_of_nodes()} {target_graph.number_of_edges()}\n")
            for u, v in target_graph.edges():
                f.write(f"{u} {v}\n")
            target_file = f.name
        
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [str(atom_exe), source_file, target_file, output_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=atom_dir
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                return None
            
            embedding = {}
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        logical_node = int(parts[0])
                        physical_chain = [int(x) for x in parts[1:]]
                        embedding[logical_node] = physical_chain
            
            # Cleanup
            os.unlink(source_file)
            os.unlink(target_file)
            os.unlink(output_file)
            
            if not embedding or not validate_embedding(embedding, source_graph, target_graph):
                return None
            
            return {'embedding': embedding, 'time': elapsed_time}
            
        except subprocess.TimeoutExpired:
            os.unlink(source_file)
            os.unlink(target_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
            return None
        
    except Exception as e:
        print(f"ATOM error: {e}")
        return None


def run_charme_embedding(source_graph: nx.Graph, target_graph: nx.Graph,
                         timeout: float = 60.0) -> Optional[Dict]:
    """Run CHARME RL-based embedding (C++ executable via subprocess)"""
    try:
        charme_dir = Path("./implementations/charme")
        charme_exe = charme_dir / "charme"
        if not charme_exe.exists():
            charme_exe = charme_dir / "charme.exe"
        
        if not charme_exe.exists():
            print("⚠️  CHARME not compiled. Check implementations/charme for build instructions")
            return None
        
        # Write graphs to temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{source_graph.number_of_nodes()} {source_graph.number_of_edges()}\n")
            for u, v in source_graph.edges():
                f.write(f"{u} {v}\n")
            source_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"{target_graph.number_of_nodes()} {target_graph.number_of_edges()}\n")
            for u, v in target_graph.edges():
                f.write(f"{u} {v}\n")
            target_file = f.name
        
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [str(charme_exe), source_file, target_file, output_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=charme_dir
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                return None
            
            embedding = {}
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        logical_node = int(parts[0])
                        physical_chain = [int(x) for x in parts[1:]]
                        embedding[logical_node] = physical_chain
            
            # Cleanup
            os.unlink(source_file)
            os.unlink(target_file)
            os.unlink(output_file)
            
            if not embedding or not validate_embedding(embedding, source_graph, target_graph):
                return None
            
            return {'embedding': embedding, 'time': elapsed_time}
            
        except subprocess.TimeoutExpired:
            os.unlink(source_file)
            os.unlink(target_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
            return None
        
    except Exception as e:
        print(f"CHARME error: {e}")
        return None


def run_oct_embedding(source_graph: nx.Graph, target_graph: nx.Graph,
                      timeout: float = 60.0) -> Optional[Dict]:
    """Run OCT-Based virtual embedding (C++ executable via subprocess)"""
    try:
        oct_dir = Path("./implementations/oct_based")
        oct_exe = oct_dir / "embedding" / "driver"
        if not oct_exe.exists():
            oct_exe = oct_dir / "embedding" / "driver.exe"
        
        if not oct_exe.exists():
            print("⚠️  OCT-Based not compiled. Check implementations/oct_based for build instructions")
            return None
        
        # Write graphs to temp files (edge list format)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for u, v in source_graph.edges():
                f.write(f"{u} {v}\n")
            source_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for u, v in target_graph.edges():
                f.write(f"{u} {v}\n")
            target_file = f.name
        
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [str(oct_exe), '--algorithm', 'oct', '--source', source_file,
                 '--target', target_file, '--output', output_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=oct_dir
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                return None
            
            embedding = {}
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        logical_node = int(parts[0])
                        physical_chain = [int(x) for x in parts[1:]]
                        embedding[logical_node] = physical_chain
            
            # Cleanup
            os.unlink(source_file)
            os.unlink(target_file)
            os.unlink(output_file)
            
            if not embedding or not validate_embedding(embedding, source_graph, target_graph):
                return None
            
            return {'embedding': embedding, 'time': elapsed_time}
            
        except subprocess.TimeoutExpired:
            os.unlink(source_file)
            os.unlink(target_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
            return None
        
    except Exception as e:
        print(f"OCT-Based error: {e}")
        return None


# ==============================================================================
# BENCHMARK INTEGRATION
# ==============================================================================

def integrate_all_methods(benchmark_instance):
    """Integrate all methods into a benchmark instance"""
    
    def atom_wrapper(source_graph, timeout=60.0):
        return run_atom_embedding(source_graph, benchmark_instance.target_graph, timeout)
    
    def charme_wrapper(source_graph, timeout=60.0):
        return run_charme_embedding(source_graph, benchmark_instance.target_graph, timeout)
    
    def oct_wrapper(source_graph, timeout=60.0):
        return run_oct_embedding(source_graph, benchmark_instance.target_graph, timeout)
    
    benchmark_instance.run_atom = atom_wrapper
    benchmark_instance.run_charme = charme_wrapper
    benchmark_instance.run_oct_based = oct_wrapper
    
    print("✓ Integration wrappers installed")


# ==============================================================================
# TESTING
# ==============================================================================

def test_integration(method_name: str = "all"):
    """Test integration of one or all methods"""
    print("=" * 80)
    print("Testing Method Integration")
    print("=" * 80)
    
    source = nx.complete_graph(4)
    target = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    
    print(f"\nTest graphs: K4 (4 nodes, 6 edges) → 3×3 grid (9 nodes, 12 edges)")
    
    methods = {
        'atom': run_atom_embedding,
        'charme': run_charme_embedding,
        'oct_based': run_oct_embedding
    }
    
    if method_name != "all":
        methods = {method_name: methods[method_name]}
    
    print("\n" + "-" * 80)
    for name, func in methods.items():
        print(f"\nTesting {name.upper()}...")
        try:
            result = func(source, target, timeout=10.0)
            if result is None:
                print(f"  Status: Not compiled/implemented")
            elif 'embedding' in result:
                print(f"  Status: ✓ Success!")
                print(f"  Time: {result['time']:.3f}s")
                print(f"  Chains: {len(result['embedding'])}")
                
                if validate_embedding(result['embedding'], source, target):
                    print(f"  Validation: ✓ Valid")
                else:
                    print(f"  Validation: ✗ Invalid")
            else:
                print(f"  Status: ✗ Unexpected format")
        except Exception as e:
            print(f"  Status: ✗ Error - {e}")
    
    print("\n" + "=" * 80)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test embedding method integration')
    parser.add_argument('--method', choices=['atom', 'charme', 'oct_based', 'all'],
                       default='all', help='Which method to test')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run quick benchmark test')
    
    args = parser.parse_args()
    
    if args.benchmark:
        from embedding_benchmark import EmbeddingBenchmark, create_chimera_graph
        
        print("Running quick benchmark test...")
        target_graph = create_chimera_graph(3, 3, 4)
        benchmark = EmbeddingBenchmark(target_graph, results_dir="./integration_test")
        
        integrate_all_methods(benchmark)
        
        problems = [
            ("K4", nx.complete_graph(4)),
            ("grid_2x2", nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 2))),
            ("cycle_5", nx.cycle_graph(5))
        ]
        
        benchmark.run_full_benchmark(problems, timeout=30.0)
        benchmark.generate_report()
        
    else:
        test_integration(args.method)
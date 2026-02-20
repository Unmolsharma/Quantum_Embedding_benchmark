"""
Comprehensive Benchmarking Framework for Minor Embedding Algorithms
Compares: minorminer, ATOM, CHARME, and OCT-Based implementations
"""

import time
import numpy as np
import networkx as nx
import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EmbeddingResult:
    """Store results from a single embedding attempt"""
    method_name: str
    problem_name: str
    problem_size: int
    problem_density: float
    success: bool
    embedding_time: float
    chain_lengths: List[int]
    avg_chain_length: float
    max_chain_length: int
    total_qubits_used: int
    total_couplers_used: int
    error_message: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class EmbeddingBenchmark:
    """Main benchmarking framework"""
    
    def __init__(self, target_graph: nx.Graph, results_dir: str = "./results"):
        """
        Initialize benchmark
        
        Args:
            target_graph: Hardware graph (e.g., Chimera, Pegasus)
            results_dir: Directory to save results
        """
        self.target_graph = target_graph
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results: List[EmbeddingResult] = []
        
    def generate_test_problems(self, sizes: List[int], 
                               densities: List[float],
                               instances_per_config: int = 5) -> List[Tuple[str, nx.Graph]]:
        """
        Generate diverse test problem graphs
        
        Args:
            sizes: List of graph sizes (number of nodes)
            densities: List of edge densities
            instances_per_config: Number of random instances per size/density combo
            
        Returns:
            List of (name, graph) tuples
        """
        problems = []
        
        for size in sizes:
            for density in densities:
                for instance in range(instances_per_config):
                    # Random graph with specified density
                    G = nx.gnp_random_graph(size, density, seed=instance)
                    name = f"random_n{size}_d{density:.2f}_i{instance}"
                    problems.append((name, G))
        
        # Add structured problems
        problems.extend(self._generate_structured_problems())
        
        return problems
    
    def _generate_structured_problems(self) -> List[Tuple[str, nx.Graph]]:
        """Generate graphs with specific structures"""
        problems = []
        
        # Complete graphs
        for n in [4, 6, 8, 10]:
            G = nx.complete_graph(n)
            problems.append((f"complete_K{n}", G))
        
        # Grid graphs
        for m in [3, 4, 5]:
            G = nx.grid_2d_graph(m, m)
            # Convert to simple graph with integer nodes
            G = nx.convert_node_labels_to_integers(G)
            problems.append((f"grid_{m}x{m}", G))
        
        # Cycle graphs
        for n in [5, 10, 15, 20]:
            G = nx.cycle_graph(n)
            problems.append((f"cycle_n{n}", G))
        
        # Tree graphs
        for depth in [3, 4]:
            G = nx.balanced_tree(2, depth)
            problems.append((f"tree_d{depth}", G))
        
        return problems
    
    def run_minorminer(self, source_graph: nx.Graph, 
                       timeout: float = 60.0) -> Optional[Dict]:
        """Run D-Wave minorminer"""
        try:
            import minorminer
            
            # Convert NetworkX graphs to adjacency dicts
            source_adj = {node: list(source_graph.neighbors(node)) 
                         for node in source_graph.nodes()}
            target_adj = {node: list(self.target_graph.neighbors(node)) 
                         for node in self.target_graph.nodes()}
            
            start_time = time.time()
            embedding = minorminer.find_embedding(
                source_adj, 
                target_adj,
                timeout=timeout,
                verbose=0
            )
            elapsed_time = time.time() - start_time
            
            if not embedding:
                return None
                
            return {
                'embedding': embedding,
                'time': elapsed_time
            }
            
        except Exception as e:
            print(f"minorminer error: {e}")
            return None
    
    def run_atom(self, source_graph: nx.Graph, 
                 timeout: float = 60.0) -> Optional[Dict]:
        """Run ATOM embedding (placeholder - needs actual implementation)"""
        # TODO: Implement based on ATOM repository
        # This is a placeholder that you'll need to fill in based on ATOM's API
        try:
            # Import ATOM library here
            # import atom_embedding
            
            start_time = time.time()
            # Call ATOM's embedding function
            # embedding = atom_embedding.find_embedding(source_graph, self.target_graph)
            elapsed_time = time.time() - start_time
            
            # Return None for now (placeholder)
            return None
            
        except Exception as e:
            print(f"ATOM error: {e}")
            return None
    
    def run_charme(self, source_graph: nx.Graph, 
                   timeout: float = 60.0) -> Optional[Dict]:
        """Run CHARME RL-based embedding (placeholder)"""
        # TODO: Implement based on CHARME repository
        try:
            # Import CHARME library here
            # import charme_embedding
            
            start_time = time.time()
            # Call CHARME's embedding function
            # embedding = charme_embedding.find_embedding(source_graph, self.target_graph)
            elapsed_time = time.time() - start_time
            
            # Return None for now (placeholder)
            return None
            
        except Exception as e:
            print(f"CHARME error: {e}")
            return None
    
    def run_oct_based(self, source_graph: nx.Graph, 
                      timeout: float = 60.0) -> Optional[Dict]:
        """Run OCT-Based virtual embedding (placeholder)"""
        # TODO: Implement based on OCT repository
        try:
            # Import OCT library here
            # import oct_embedding
            
            start_time = time.time()
            # Call OCT's embedding function
            # embedding = oct_embedding.find_embedding(source_graph, self.target_graph)
            elapsed_time = time.time() - start_time
            
            # Return None for now (placeholder)
            return None
            
        except Exception as e:
            print(f"OCT-Based error: {e}")
            return None
    
    def compute_embedding_metrics(self, embedding: Dict[int, List[int]]) -> Dict:
        """
        Compute quality metrics for an embedding
        
        Args:
            embedding: Dict mapping source nodes to lists of target nodes (chains)
            
        Returns:
            Dict of metrics
        """
        chain_lengths = [len(chain) for chain in embedding.values()]
        
        # Count total qubits and couplers used
        all_qubits = set()
        coupler_count = 0
        
        for chain in embedding.values():
            all_qubits.update(chain)
            # Count intra-chain couplers
            for i in range(len(chain)):
                for j in range(i+1, len(chain)):
                    if self.target_graph.has_edge(chain[i], chain[j]):
                        coupler_count += 1
        
        return {
            'chain_lengths': chain_lengths,
            'avg_chain_length': np.mean(chain_lengths),
            'max_chain_length': max(chain_lengths),
            'total_qubits_used': len(all_qubits),
            'total_couplers_used': coupler_count
        }
    
    def benchmark_single(self, method_name: str, method_func, 
                        problem_name: str, source_graph: nx.Graph,
                        timeout: float = 60.0) -> EmbeddingResult:
        """
        Benchmark a single method on a single problem
        
        Args:
            method_name: Name of the embedding method
            method_func: Function to call for embedding
            problem_name: Name of the problem instance
            source_graph: Source graph to embed
            timeout: Maximum time allowed
            
        Returns:
            EmbeddingResult object
        """
        problem_size = source_graph.number_of_nodes()
        problem_density = (2 * source_graph.number_of_edges() / 
                          (problem_size * (problem_size - 1)) if problem_size > 1 else 0)
        
        try:
            result = method_func(source_graph, timeout=timeout)
            
            if result is None or 'embedding' not in result:
                return EmbeddingResult(
                    method_name=method_name,
                    problem_name=problem_name,
                    problem_size=problem_size,
                    problem_density=problem_density,
                    success=False,
                    embedding_time=timeout,
                    chain_lengths=[],
                    avg_chain_length=0.0,
                    max_chain_length=0,
                    total_qubits_used=0,
                    total_couplers_used=0,
                    error_message="No embedding found"
                )
            
            metrics = self.compute_embedding_metrics(result['embedding'])
            
            return EmbeddingResult(
                method_name=method_name,
                problem_name=problem_name,
                problem_size=problem_size,
                problem_density=problem_density,
                success=True,
                embedding_time=result['time'],
                chain_lengths=metrics['chain_lengths'],
                avg_chain_length=metrics['avg_chain_length'],
                max_chain_length=metrics['max_chain_length'],
                total_qubits_used=metrics['total_qubits_used'],
                total_couplers_used=metrics['total_couplers_used']
            )
            
        except Exception as e:
            return EmbeddingResult(
                method_name=method_name,
                problem_name=problem_name,
                problem_size=problem_size,
                problem_density=problem_density,
                success=False,
                embedding_time=timeout,
                chain_lengths=[],
                avg_chain_length=0.0,
                max_chain_length=0,
                total_qubits_used=0,
                total_couplers_used=0,
                error_message=str(e)
            )
    
    def run_full_benchmark(self, problems: List[Tuple[str, nx.Graph]], 
                          timeout: float = 60.0,
                          methods: Optional[List[str]] = None):
        """
        Run complete benchmark suite
        
        Args:
            problems: List of (name, graph) tuples
            timeout: Timeout per embedding attempt
            methods: List of method names to test (default: all)
        """
        if methods is None:
            methods = ['minorminer', 'atom', 'charme', 'oct_based']
        
        method_map = {
            'minorminer': self.run_minorminer,
            'atom': self.run_atom,
            'charme': self.run_charme,
            'oct_based': self.run_oct_based
        }
        
        total_runs = len(problems) * len(methods)
        current_run = 0
        
        print(f"Starting benchmark: {len(problems)} problems × {len(methods)} methods = {total_runs} runs")
        print("=" * 80)
        
        for problem_name, source_graph in problems:
            print(f"\nProblem: {problem_name} (n={source_graph.number_of_nodes()}, "
                  f"e={source_graph.number_of_edges()})")
            
            for method_name in methods:
                current_run += 1
                print(f"  [{current_run}/{total_runs}] Running {method_name}...", end=" ")
                
                result = self.benchmark_single(
                    method_name,
                    method_map[method_name],
                    problem_name,
                    source_graph,
                    timeout
                )
                
                self.results.append(result)
                
                if result.success:
                    print(f"✓ {result.embedding_time:.3f}s, "
                          f"avg_chain={result.avg_chain_length:.2f}, "
                          f"qubits={result.total_qubits_used}")
                else:
                    print(f"✗ Failed: {result.error_message}")
        
        print("\n" + "=" * 80)
        print("Benchmark complete!")
        self.save_results()
    
    def save_results(self):
        """Save results to JSON and CSV"""
        # Save as JSON
        json_path = self.results_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"\nResults saved to {json_path}")
        
        # Save as CSV
        csv_path = self.results_dir / "results.csv"
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    
    def generate_report(self):
        """Generate comprehensive analysis and visualizations"""
        if not self.results:
            print("No results to analyze!")
            return
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Create visualizations
        self._plot_success_rates(df)
        self._plot_embedding_times(df)
        self._plot_chain_lengths(df)
        self._plot_scalability(df)
        self._generate_summary_statistics(df)
    
    def _plot_success_rates(self, df: pd.DataFrame):
        """Plot success rates by method"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        success_rates = df.groupby('method_name')['success'].mean() * 100
        success_rates.plot(kind='bar', ax=ax, color='steelblue')
        
        ax.set_ylabel('Success Rate (%)')
        ax.set_xlabel('Method')
        ax.set_title('Embedding Success Rates by Method')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'success_rates.png', dpi=300)
        print(f"Success rate plot saved to {self.results_dir / 'success_rates.png'}")
        plt.close()
    
    def _plot_embedding_times(self, df: pd.DataFrame):
        """Plot embedding time distributions"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        successful_df = df[df['success'] == True]
        
        if len(successful_df) > 0:
            successful_df.boxplot(column='embedding_time', by='method_name', ax=ax)
            ax.set_ylabel('Embedding Time (seconds)')
            ax.set_xlabel('Method')
            ax.set_title('Embedding Time Distribution (Successful Embeddings Only)')
            plt.suptitle('')  # Remove default title
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'embedding_times.png', dpi=300)
            print(f"Embedding time plot saved to {self.results_dir / 'embedding_times.png'}")
            plt.close()
    
    def _plot_chain_lengths(self, df: pd.DataFrame):
        """Plot chain length comparisons"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        successful_df = df[df['success'] == True]
        
        if len(successful_df) > 0:
            # Average chain length
            successful_df.boxplot(column='avg_chain_length', by='method_name', ax=ax1)
            ax1.set_ylabel('Average Chain Length')
            ax1.set_xlabel('Method')
            ax1.set_title('Average Chain Length Distribution')
            
            # Max chain length
            successful_df.boxplot(column='max_chain_length', by='method_name', ax=ax2)
            ax2.set_ylabel('Maximum Chain Length')
            ax2.set_xlabel('Method')
            ax2.set_title('Maximum Chain Length Distribution')
            
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'chain_lengths.png', dpi=300)
            print(f"Chain length plot saved to {self.results_dir / 'chain_lengths.png'}")
            plt.close()
    
    def _plot_scalability(self, df: pd.DataFrame):
        """Plot scalability analysis"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        successful_df = df[df['success'] == True]
        
        if len(successful_df) > 0:
            for method in successful_df['method_name'].unique():
                method_df = successful_df[successful_df['method_name'] == method]
                grouped = method_df.groupby('problem_size')['embedding_time'].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=method, linewidth=2)
            
            ax.set_xlabel('Problem Size (number of nodes)')
            ax.set_ylabel('Average Embedding Time (seconds)')
            ax.set_title('Scalability: Embedding Time vs Problem Size')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'scalability.png', dpi=300)
            print(f"Scalability plot saved to {self.results_dir / 'scalability.png'}")
            plt.close()
    
    def _generate_summary_statistics(self, df: pd.DataFrame):
        """Generate and save summary statistics"""
        summary = []
        
        for method in df['method_name'].unique():
            method_df = df[df['method_name'] == method]
            successful_df = method_df[method_df['success'] == True]
            
            stats = {
                'Method': method,
                'Total Runs': len(method_df),
                'Successful': len(successful_df),
                'Success Rate (%)': len(successful_df) / len(method_df) * 100,
                'Avg Time (s)': successful_df['embedding_time'].mean() if len(successful_df) > 0 else None,
                'Std Time (s)': successful_df['embedding_time'].std() if len(successful_df) > 0 else None,
                'Avg Chain Length': successful_df['avg_chain_length'].mean() if len(successful_df) > 0 else None,
                'Avg Max Chain': successful_df['max_chain_length'].mean() if len(successful_df) > 0 else None,
                'Avg Qubits Used': successful_df['total_qubits_used'].mean() if len(successful_df) > 0 else None
            }
            summary.append(stats)
        
        summary_df = pd.DataFrame(summary)
        summary_path = self.results_dir / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nSummary statistics saved to {summary_path}")
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(summary_df.to_string(index=False))


def create_chimera_graph(m: int = 4, n: int = 4, t: int = 4) -> nx.Graph:
    """
    Create a Chimera graph (D-Wave topology)
    
    Args:
        m, n: Grid dimensions
        t: Number of qubits per unit cell
        
    Returns:
        NetworkX graph representing Chimera topology
    """
    try:
        import dwave_networkx as dnx
        return dnx.chimera_graph(m, n, t)
    except ImportError:
        print("Warning: dwave_networkx not installed, creating simplified Chimera")
        # Simplified version if dwave_networkx not available
        G = nx.Graph()
        # This is a simplified placeholder - install dwave_networkx for real Chimera
        G.add_nodes_from(range(m * n * t * 2))
        return G


if __name__ == "__main__":
    # Example usage
    print("Minor Embedding Benchmarking Framework")
    print("=" * 80)
    
    # Create target hardware graph (Chimera 4x4)
    target_graph = create_chimera_graph(4, 4, 4)
    print(f"Target graph: Chimera 4×4 with {target_graph.number_of_nodes()} qubits")
    
    # Initialize benchmark
    benchmark = EmbeddingBenchmark(target_graph, results_dir="./benchmark_results")
    
    # Generate test problems
    problems = benchmark.generate_test_problems(
        sizes=[4, 6, 8, 10],
        densities=[0.3, 0.5, 0.7],
        instances_per_config=3
    )
    print(f"Generated {len(problems)} test problems")
    
    # Run benchmark (only minorminer is implemented, others are placeholders)
    benchmark.run_full_benchmark(problems, timeout=30.0, methods=['minorminer'])
    
    # Generate analysis report
    benchmark.generate_report()

"""
Quick Start Example - Run Benchmark with minorminer
This example shows how to run the benchmark (currently only minorminer is fully integrated)
"""

from embedding_benchmark import EmbeddingBenchmark, create_chimera_graph
import networkx as nx


def main():
    print("=" * 80)
    print("Minor Embedding Benchmark - Quick Start")
    print("=" * 80)
    
    # Step 1: Create target hardware graph
    print("\n[1/4] Creating target hardware graph (Chimera 4×4)...")
    target_graph = create_chimera_graph(m=4, n=4, t=4)
    print(f"  Created Chimera graph with {target_graph.number_of_nodes()} qubits")
    print(f"  and {target_graph.number_of_edges()} couplers")
    
    # Step 2: Initialize benchmark
    print("\n[2/4] Initializing benchmark framework...")
    benchmark = EmbeddingBenchmark(target_graph, results_dir="./quick_start_results")
    print("  Benchmark initialized")
    
    # Step 3: Generate test problems
    print("\n[3/4] Generating test problems...")
    problems = benchmark.generate_test_problems(
        sizes=[4, 6, 8],           # Small, medium graphs
        densities=[0.3, 0.5],       # Low and medium density
        instances_per_config=2      # 2 random instances per configuration
    )
    print(f"  Generated {len(problems)} test problems:")
    print(f"    - Random graphs: varying sizes and densities")
    print(f"    - Structured graphs: complete, grid, cycle, tree")
    
    # Step 4: Run benchmark
    print("\n[4/4] Running benchmark...")
    print("  Note: Only testing minorminer (other methods need integration)")
    print()
    
    benchmark.run_full_benchmark(
        problems=problems,
        timeout=30.0,              # 30 second timeout per problem
        methods=['minorminer']     # Only test minorminer for now
    )
    
    # Step 5: Generate report
    print("\n[5/5] Generating analysis report...")
    benchmark.generate_report()
    
    print("\n" + "=" * 80)
    print("Quick start complete! Check ./quick_start_results/ for outputs:")
    print("  - results.json: Raw results in JSON format")
    print("  - results.csv: Results in CSV format")
    print("  - summary_statistics.csv: Aggregated statistics")
    print("  - *.png: Visualization plots")
    print("=" * 80)


if __name__ == "__main__":
    main()

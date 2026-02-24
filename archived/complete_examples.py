"""
Complete Example: Using the Integration Module

This shows you exactly how to integrate the methods and run benchmarks
"""

from embedding_benchmark import EmbeddingBenchmark, create_chimera_graph
from embedding_integrations import integrate_all_methods, test_integration
import networkx as nx


def example_1_test_integration():
    """
    Example 1: Test if your integrations are working
    This will call each method on a simple graph to verify the integration
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Testing Integration")
    print("="*80)
    
    # Test all methods
    test_integration(method_name='all')
    
    print("\n✓ Integration test complete!")
    print("If you see 'Not implemented', edit embedding_integrations.py")


def example_2_benchmark_single_method():
    """
    Example 2: Benchmark a single method (minorminer)
    Shows the basic workflow
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Benchmark Single Method (minorminer)")
    print("="*80)
    
    # Step 1: Create target hardware graph
    target_graph = create_chimera_graph(m=4, n=4, t=4)
    print(f"\n✓ Created Chimera 4×4 with {target_graph.number_of_nodes()} qubits")
    
    # Step 2: Initialize benchmark
    benchmark = EmbeddingBenchmark(target_graph, results_dir="./example_2_results")
    print("✓ Benchmark initialized")
    
    # Step 3: Generate test problems
    problems = benchmark.generate_test_problems(
        sizes=[4, 6, 8],
        densities=[0.3, 0.5],
        instances_per_config=2
    )
    print(f"✓ Generated {len(problems)} test problems")
    
    # Step 4: Run benchmark (only minorminer since it's already integrated)
    print("\n▶ Running benchmark...")
    benchmark.run_full_benchmark(
        problems=problems,
        timeout=30.0,
        methods=['minorminer']
    )
    
    # Step 5: Generate report
    print("\n▶ Generating analysis...")
    benchmark.generate_report()
    
    print("\n✓ Results saved to ./example_2_results/")


def example_3_benchmark_all_methods():
    """
    Example 3: Benchmark all four methods
    Only works after you've implemented ATOM, CHARME, and OCT-Based
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Benchmark All Methods")
    print("="*80)
    
    # Create target graph
    target_graph = create_chimera_graph(m=4, n=4, t=4)
    print(f"\n✓ Created Chimera 4×4 with {target_graph.number_of_nodes()} qubits")
    
    # Initialize benchmark
    benchmark = EmbeddingBenchmark(target_graph, results_dir="./example_3_results")
    
    # IMPORTANT: Integrate the other three methods
    integrate_all_methods(benchmark)
    print("✓ All methods integrated")
    
    # Generate comprehensive test suite
    problems = benchmark.generate_test_problems(
        sizes=[4, 6, 8, 10, 12],
        densities=[0.3, 0.5, 0.7],
        instances_per_config=5
    )
    print(f"✓ Generated {len(problems)} test problems")
    
    # Run all methods
    print("\n▶ Running benchmark on all methods...")
    print("  This may take a while...")
    
    benchmark.run_full_benchmark(
        problems=problems,
        timeout=60.0,
        methods=['minorminer', 'atom', 'charme', 'oct_based']
    )
    
    # Generate comprehensive report
    print("\n▶ Generating comprehensive analysis...")
    benchmark.generate_report()
    
    print("\n✓ Results saved to ./example_3_results/")
    print("  Check the CSV files and PNG plots for comparisons!")


def example_4_custom_problems():
    """
    Example 4: Test on your own custom problem graphs
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Problem Graphs")
    print("="*80)
    
    # Create target graph
    target_graph = create_chimera_graph(m=4, n=4, t=4)
    print(f"\n✓ Created Chimera 4×4")
    
    # Initialize benchmark
    benchmark = EmbeddingBenchmark(target_graph, results_dir="./example_4_results")
    integrate_all_methods(benchmark)
    
    # Define custom problems
    custom_problems = [
        # Your specific application graphs
        ("my_application_graph", nx.karate_club_graph()),
        ("custom_grid", nx.grid_2d_graph(4, 4)),
        ("custom_tree", nx.balanced_tree(3, 3)),
        ("erdos_renyi", nx.erdos_renyi_graph(12, 0.4)),
    ]
    
    # Convert any 2D grid graphs to integer labels
    converted_problems = []
    for name, graph in custom_problems:
        if not all(isinstance(n, int) for n in graph.nodes()):
            graph = nx.convert_node_labels_to_integers(graph)
        converted_problems.append((name, graph))
    
    print(f"✓ Created {len(converted_problems)} custom problems")
    
    # Run benchmark
    print("\n▶ Running benchmark...")
    benchmark.run_full_benchmark(
        problems=converted_problems,
        timeout=60.0,
        methods=['minorminer', 'atom', 'charme', 'oct_based']
    )
    
    benchmark.generate_report()
    print("\n✓ Results saved to ./example_4_results/")


def example_5_compare_hardware_graphs():
    """
    Example 5: Compare methods on different hardware topologies
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Compare Different Hardware Topologies")
    print("="*80)
    
    # Test on different hardware graphs
    hardware_configs = [
        ("Chimera_3x3", create_chimera_graph(3, 3, 4)),
        ("Chimera_4x4", create_chimera_graph(4, 4, 4)),
    ]
    
    # Simple test problems
    test_problems = [
        ("K5", nx.complete_graph(5)),
        ("grid_3x3", nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))),
        ("cycle_10", nx.cycle_graph(10)),
    ]
    
    all_results = []
    
    for hw_name, hw_graph in hardware_configs:
        print(f"\n▶ Testing on {hw_name}...")
        
        benchmark = EmbeddingBenchmark(
            hw_graph, 
            results_dir=f"./example_5_{hw_name}_results"
        )
        integrate_all_methods(benchmark)
        
        benchmark.run_full_benchmark(
            problems=test_problems,
            timeout=30.0,
            methods=['minorminer']  # Add others when ready
        )
        
        benchmark.generate_report()
        all_results.append((hw_name, benchmark.results))
    
    print("\n✓ All hardware configurations tested!")
    print("  Compare the results directories to see differences")


def main():
    """
    Main menu to run different examples
    """
    print("\n" + "="*80)
    print("EMBEDDING BENCHMARK - COMPLETE EXAMPLES")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Test integration (verify your implementations work)")
    print("  2. Benchmark single method (minorminer)")
    print("  3. Benchmark all methods (requires integration)")
    print("  4. Custom problem graphs")
    print("  5. Compare hardware topologies")
    print("  6. Run all examples")
    print()
    
    choice = input("Enter example number (1-6) or 'q' to quit: ").strip()
    
    if choice == '1':
        example_1_test_integration()
    elif choice == '2':
        example_2_benchmark_single_method()
    elif choice == '3':
        example_3_benchmark_all_methods()
    elif choice == '4':
        example_4_custom_problems()
    elif choice == '5':
        example_5_compare_hardware_graphs()
    elif choice == '6':
        print("\nRunning all examples...")
        example_1_test_integration()
        example_2_benchmark_single_method()
        # Only run 3-5 if methods are integrated
        print("\nSkipping examples 3-5 (require method integration)")
        print("After implementing ATOM, CHARME, OCT-Based, run them individually")
    elif choice.lower() == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")


if __name__ == "__main__":
    # You can either run the menu or directly call any example
    
    # Option 1: Interactive menu
    main()
    
    # Option 2: Run a specific example directly (comment out main() above)
    # example_1_test_integration()
    # example_2_benchmark_single_method()
    # example_3_benchmark_all_methods()
    # example_4_custom_problems()
    # example_5_compare_hardware_graphs()

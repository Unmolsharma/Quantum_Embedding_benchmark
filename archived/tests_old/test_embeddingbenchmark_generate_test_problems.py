import pytest
import networkx as nx
from embedding_benchmark import EmbeddingBenchmark

def test_generate_test_problems_basic():
    """
    Tests `generate_test_problems_basic` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    # sizes: [4, 5], densities: [0.5], instances: 2
    # Expect 2*1*2 = 4 random graphs, plus structured graphs
    problems = benchmark.generate_test_problems(sizes=[4, 5], densities=[0.5], instances_per_config=2)
    
    assert isinstance(problems, list)
    # Check if we generated the expected number of random problems
    random_problems = [p for p in problems if p[0].startswith("random_")]
    assert len(random_problems) == 4
    
    # Check graph properties for one random problem
    name, graph = random_problems[0]
    assert isinstance(name, str)
    assert isinstance(graph, nx.Graph)

def test_generate_test_problems_empty_input():
    """
    Tests `generate_test_problems_empty_input` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    problems = benchmark.generate_test_problems(sizes=[], densities=[0.5], instances_per_config=2)
    
    random_problems = [p for p in problems if p[0].startswith("random_")]
    assert len(random_problems) == 0
    # Should still have structured problems
    assert len(problems) > 0

def test_generate_test_problems_zero_instances():
    """
    Tests `generate_test_problems_zero_instances` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    problems = benchmark.generate_test_problems(sizes=[4], densities=[0.5], instances_per_config=0)
    
    random_problems = [p for p in problems if p[0].startswith("random_")]
    assert len(random_problems) == 0

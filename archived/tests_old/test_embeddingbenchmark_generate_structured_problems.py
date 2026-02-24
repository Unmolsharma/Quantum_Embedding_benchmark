import pytest
import networkx as nx
from embedding_benchmark import EmbeddingBenchmark

def test_generate_structured_problems():
    """
    Tests `generate_structured_problems` behavior.
    Equivalence Class: General behavior / Default path
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    problems = benchmark._generate_structured_problems()
    
    assert isinstance(problems, list)
    assert len(problems) > 0
    
    # Check complete graphs: 4, 6, 8, 10 (4 graphs)
    complete_graphs = [p for p in problems if p[0].startswith("complete_")]
    assert len(complete_graphs) == 4
    
    # Check grid graphs: 3, 4, 5 (3 graphs)
    grid_graphs = [p for p in problems if p[0].startswith("grid_")]
    assert len(grid_graphs) == 3
    
    # Check cycle graphs: 5, 10, 15, 20 (4 graphs)
    cycle_graphs = [p for p in problems if p[0].startswith("cycle_")]
    assert len(cycle_graphs) == 4
    
    # Check tree graphs: 3, 4 (2 graphs)
    tree_graphs = [p for p in problems if p[0].startswith("tree_")]
    assert len(tree_graphs) == 2

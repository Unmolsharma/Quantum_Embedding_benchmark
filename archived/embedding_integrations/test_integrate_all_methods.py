import pytest
import networkx as nx
from unittest.mock import MagicMock
from embedding_integrations import integrate_all_methods
from embedding_benchmark import EmbeddingBenchmark

def test_integrate_all_methods():
    """
    Tests `integrate_all_methods` behavior.
    Equivalence Class: General behavior / Default path
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    # Store originals
    orig_atom = benchmark.run_atom
    orig_charme = benchmark.run_charme
    orig_oct = benchmark.run_oct_based
    
    # Integrate
    integrate_all_methods(benchmark)
    
    # Check that they were overwritten
    assert benchmark.run_atom != orig_atom
    assert benchmark.run_charme != orig_charme
    assert benchmark.run_oct_based != orig_oct
    
    # Verify the wrappers are callable
    # Not testing the actual execution since they depend on external binaries
    assert callable(benchmark.run_atom)
    assert callable(benchmark.run_charme)
    assert callable(benchmark.run_oct_based)

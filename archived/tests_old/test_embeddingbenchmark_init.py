import pytest
import networkx as nx
from pathlib import Path
from embedding_benchmark import EmbeddingBenchmark

def test_embeddingbenchmark_init_default_dir(tmp_path):
    """
    Tests `embeddingbenchmark_init_default_dir` behavior.
    Equivalence Class: General behavior / Default path
    """
    # Setup
    target_graph = nx.complete_graph(4)
    
    # Change current working directory to tmp_path to test default directory creation
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Execute
        benchmark = EmbeddingBenchmark(target_graph=target_graph)
        
        # Assert
        assert benchmark.target_graph == target_graph
        assert benchmark.results_dir == Path("./results")
        assert benchmark.results_dir.exists()
        assert benchmark.results == []
    finally:
        os.chdir(original_cwd)

def test_embeddingbenchmark_init_custom_dir(tmp_path):
    """
    Tests `embeddingbenchmark_init_custom_dir` behavior.
    Equivalence Class: General behavior / Default path
    """
    target_graph = nx.star_graph(5)
    custom_dir = tmp_path / "custom_results"
    
    benchmark = EmbeddingBenchmark(target_graph=target_graph, results_dir=str(custom_dir))
    
    assert benchmark.target_graph == target_graph
    assert benchmark.results_dir == custom_dir
    assert benchmark.results_dir.exists()
    assert benchmark.results == []

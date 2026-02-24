import pytest
import networkx as nx
from unittest.mock import patch
from embedding_benchmark import EmbeddingBenchmark

@patch('embedding_benchmark.minorminer.find_embedding', create=True)
def test_run_minorminer_success(mock_find_embedding):
    """
    Tests `run_minorminer_success` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    mock_find_embedding.return_value = {0: [0, 1], 1: [2]}
    
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    source_graph = nx.complete_graph(2)
    
    result = benchmark.run_minorminer(source_graph, timeout=60.0)
    
    assert result is not None
    assert 'embedding' in result
    assert result['embedding'] == {0: [0, 1], 1: [2]}
    assert 'time' in result
    mock_find_embedding.assert_called_once()

@patch('embedding_benchmark.minorminer.find_embedding', create=True)
def test_run_minorminer_empty(mock_find_embedding):
    """
    Tests `run_minorminer_empty` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    mock_find_embedding.return_value = {} # empty means failure to embed
    
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    source_graph = nx.complete_graph(5)
    
    result = benchmark.run_minorminer(source_graph, timeout=60.0)
    
    assert result is None

@patch('embedding_benchmark.minorminer.find_embedding', create=True)
def test_run_minorminer_exception(mock_find_embedding):
    """
    Tests `run_minorminer_exception` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    mock_find_embedding.side_effect = Exception("minorminer crashed")
    
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    source_graph = nx.complete_graph(2)
    
    result = benchmark.run_minorminer(source_graph, timeout=60.0)
    
    assert result is None

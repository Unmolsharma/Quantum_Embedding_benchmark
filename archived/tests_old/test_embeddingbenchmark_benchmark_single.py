import pytest
import networkx as nx
from embedding_benchmark import EmbeddingBenchmark, EmbeddingResult

def dummy_method(sg, timeout):
    return {'embedding': {0: [0], 1: [1]}, 'time': 0.123}

def dummy_method_fail(sg, timeout):
    return None

def dummy_method_error(sg, timeout):
    raise ValueError("Something broke")

def test_benchmark_single_success():
    """
    Tests `benchmark_single_success` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    target_graph = nx.path_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    source_graph = nx.path_graph(2)
        
    result = benchmark.benchmark_single(
        method_name="dummy",
        method_func=dummy_method,
        problem_name="test_prob",
        source_graph=source_graph,
        timeout=10.0
    )
    
    assert isinstance(result, EmbeddingResult)
    assert result.success is True
    assert result.method_name == "dummy"
    assert result.embedding_time == 0.123
    assert result.total_qubits_used == 2

def test_benchmark_single_failure():
    """
    Tests `benchmark_single_failure` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    target_graph = nx.path_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    source_graph = nx.path_graph(2)
        
    result = benchmark.benchmark_single(
        method_name="dummy",
        method_func=dummy_method_fail,
        problem_name="test_prob",
        source_graph=source_graph,
        timeout=10.0
    )
    
    assert isinstance(result, EmbeddingResult)
    assert result.success is False
    assert result.error_message == "No embedding found"

def test_benchmark_single_exception():
    """
    Tests `benchmark_single_exception` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    target_graph = nx.path_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    source_graph = nx.path_graph(2)
        
    result = benchmark.benchmark_single(
        method_name="dummy",
        method_func=dummy_method_error,
        problem_name="test_prob",
        source_graph=source_graph,
        timeout=10.0
    )
    
    assert isinstance(result, EmbeddingResult)
    assert result.success is False
    assert "Something broke" in result.error_message

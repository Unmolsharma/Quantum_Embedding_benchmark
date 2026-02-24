import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from embedding_benchmark import EmbeddingBenchmark, EmbeddingResult

@patch.object(EmbeddingBenchmark, 'benchmark_single')
@patch.object(EmbeddingBenchmark, 'save_results')
def test_run_full_benchmark(mock_save_results, mock_benchmark_single):
    """
    Tests `run_full_benchmark` behavior.
    Equivalence Class: General behavior / Default path
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    # Create fake success result
    fake_result = EmbeddingResult(
        method_name="fake", problem_name="prob1", problem_size=2, problem_density=1.0, 
        success=True, embedding_time=1.0, chain_lengths=[1,1], avg_chain_length=1.0, 
        max_chain_length=1, total_qubits_used=2, total_couplers_used=1
    )
    mock_benchmark_single.return_value = fake_result
    
    problems = [
        ("prob1", nx.complete_graph(2))
    ]
    
    methods = ['minorminer', 'atom']
    
    benchmark.run_full_benchmark(problems, timeout=10.0, methods=methods)
    
    assert mock_benchmark_single.call_count == 2
    mock_save_results.assert_called_once()
    assert len(benchmark.results) == 2
    assert benchmark.results[0] == fake_result
    
@patch.object(EmbeddingBenchmark, 'benchmark_single')
@patch.object(EmbeddingBenchmark, 'save_results')
def test_run_full_benchmark_default_methods(mock_save_results, mock_benchmark_single):
    """
    Tests `run_full_benchmark_default_methods` behavior.
    Equivalence Class: General behavior / Default path
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    fake_result = EmbeddingResult(
        method_name="fake", problem_name="prob1", problem_size=2, problem_density=1.0, 
        success=False, embedding_time=1.0, chain_lengths=[], avg_chain_length=0.0, 
        max_chain_length=0, total_qubits_used=0, total_couplers_used=0, error_message="fail"
    )
    mock_benchmark_single.return_value = fake_result
    
    benchmark.run_full_benchmark([("p", nx.empty_graph(1))])
    
    # Default is 4 methods
    assert mock_benchmark_single.call_count == 4

import pytest
from unittest.mock import patch, MagicMock
from complete_examples import example_3_benchmark_all_methods

@patch('complete_examples.integrate_all_methods')
@patch('complete_examples.EmbeddingBenchmark')
@patch('complete_examples.create_chimera_graph')
def test_example_3_benchmark_all_methods(mock_create_chimera, mock_benchmark_cls, mock_integrate):
    """
    Tests `example_3_benchmark_all_methods` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 16
    mock_create_chimera.return_value = mock_graph
    
    mock_benchmark = MagicMock()
    mock_benchmark.generate_test_problems.return_value = [("prob1", "graph1")]
    mock_benchmark_cls.return_value = mock_benchmark
    
    example_3_benchmark_all_methods()
    
    mock_create_chimera.assert_called_once_with(m=4, n=4, t=4)
    mock_benchmark_cls.assert_called_once()
    mock_integrate.assert_called_once_with(mock_benchmark)
    mock_benchmark.generate_test_problems.assert_called_once()
    mock_benchmark.run_full_benchmark.assert_called_once()
    mock_benchmark.generate_report.assert_called_once()

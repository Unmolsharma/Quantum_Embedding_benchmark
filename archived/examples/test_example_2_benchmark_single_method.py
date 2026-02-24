import pytest
from unittest.mock import patch, MagicMock
from complete_examples import example_2_benchmark_single_method

@patch('complete_examples.EmbeddingBenchmark')
@patch('complete_examples.create_chimera_graph')
def test_example_2_benchmark_single_method(mock_create_chimera, mock_benchmark_cls):
    """
    Tests `example_2_benchmark_single_method` behavior.
    Equivalence Class: General behavior / Default path
    """
    # Setup mocks
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 16
    mock_create_chimera.return_value = mock_graph
    
    mock_benchmark = MagicMock()
    mock_benchmark.generate_test_problems.return_value = [("prob1", "graph1")]
    mock_benchmark_cls.return_value = mock_benchmark
    
    # Execute
    example_2_benchmark_single_method()
    
    # Assert
    mock_create_chimera.assert_called_once_with(m=4, n=4, t=4)
    mock_benchmark_cls.assert_called_once()
    mock_benchmark.generate_test_problems.assert_called_once()
    mock_benchmark.run_full_benchmark.assert_called_once()
    mock_benchmark.generate_report.assert_called_once()

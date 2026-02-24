import pytest
from unittest.mock import patch, MagicMock
from quick_start import main

@patch('quick_start.EmbeddingBenchmark')
@patch('quick_start.create_chimera_graph')
def test_quick_start_main(mock_chimera, mock_benchmark_cls):
    """
    Tests `quick_start_main` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 16
    mock_graph.number_of_edges.return_value = 32
    mock_chimera.return_value = mock_graph
    
    mock_benchmark = MagicMock()
    mock_benchmark.generate_test_problems.return_value = [("p", "g")]
    mock_benchmark_cls.return_value = mock_benchmark
    
    main()
    
    mock_chimera.assert_called_once()
    mock_benchmark_cls.assert_called_once_with(mock_graph, results_dir="./quick_start_results")
    mock_benchmark.generate_test_problems.assert_called_once()
    mock_benchmark.run_full_benchmark.assert_called_once()
    mock_benchmark.generate_report.assert_called_once()

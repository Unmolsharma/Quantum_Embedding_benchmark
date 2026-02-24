import pytest
from unittest.mock import patch, MagicMock
from complete_examples import example_5_compare_hardware_graphs

@patch('complete_examples.integrate_all_methods')
@patch('complete_examples.EmbeddingBenchmark')
@patch('complete_examples.create_chimera_graph')
def test_example_5_compare_hardware_graphs(mock_create_chimera, mock_benchmark_cls, mock_integrate):
    """
    Tests `example_5_compare_hardware_graphs` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_graph = MagicMock()
    mock_create_chimera.return_value = mock_graph
    
    mock_benchmark = MagicMock()
    mock_benchmark.results = []
    mock_benchmark_cls.return_value = mock_benchmark
    
    example_5_compare_hardware_graphs()
    
    assert mock_create_chimera.call_count == 2
    assert mock_benchmark_cls.call_count == 2
    assert mock_integrate.call_count == 2
    assert mock_benchmark.run_full_benchmark.call_count == 2
    assert mock_benchmark.generate_report.call_count == 2

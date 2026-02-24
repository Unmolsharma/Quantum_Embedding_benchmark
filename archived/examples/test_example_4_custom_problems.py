import pytest
from unittest.mock import patch, MagicMock
from complete_examples import example_4_custom_problems

@patch('complete_examples.integrate_all_methods')
@patch('complete_examples.EmbeddingBenchmark')
@patch('complete_examples.create_chimera_graph')
def test_example_4_custom_problems(mock_create_chimera, mock_benchmark_cls, mock_integrate):
    """
    Tests `example_4_custom_problems` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_graph = MagicMock()
    mock_create_chimera.return_value = mock_graph
    
    mock_benchmark = MagicMock()
    mock_benchmark_cls.return_value = mock_benchmark
    
    example_4_custom_problems()
    
    mock_create_chimera.assert_called_once_with(m=4, n=4, t=4)
    mock_benchmark_cls.assert_called_once()
    mock_integrate.assert_called_once_with(mock_benchmark)
    mock_benchmark.run_full_benchmark.assert_called_once()
    
    # Assert custom problems were passed
    args, kwargs = mock_benchmark.run_full_benchmark.call_args
    assert 'problems' in kwargs
    assert len(kwargs['problems']) == 4
    mock_benchmark.generate_report.assert_called_once()

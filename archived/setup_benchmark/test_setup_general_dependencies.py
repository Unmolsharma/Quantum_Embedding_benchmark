import pytest
from unittest.mock import patch
from setup_benchmark import setup_general_dependencies

@patch('setup_benchmark.run_command')
def test_setup_general_dependencies(mock_run):
    """
    Tests `setup_general_dependencies` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_run.return_value = True
    setup_general_dependencies()
    mock_run.assert_called_once()
    assert "pip install" in str(mock_run.call_args)

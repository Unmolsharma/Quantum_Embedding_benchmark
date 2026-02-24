import pytest
from unittest.mock import patch, MagicMock
from setup_benchmark import setup_minorminer

@patch('setup_benchmark.run_command')
def test_setup_minorminer(mock_run):
    """
    Tests `setup_minorminer` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_run.return_value = True
    setup_minorminer()
    mock_run.assert_called_once()
    assert "pip install" in str(mock_run.call_args)

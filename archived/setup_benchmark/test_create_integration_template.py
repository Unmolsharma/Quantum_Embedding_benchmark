import pytest
from unittest.mock import patch, mock_open
from setup_benchmark import create_integration_template

@patch('setup_benchmark.Path.exists')
def test_create_integration_template_exists(mock_exists):
    """
    Tests `create_integration_template_exists` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = True
    # Should not write if it exists
    with patch('builtins.open', mock_open()) as mocked_file:
        create_integration_template()
        mocked_file.assert_not_called()

@patch('setup_benchmark.Path.exists')
def test_create_integration_template_new(mock_exists):
    """
    Tests `create_integration_template_new` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = False
    with patch('builtins.open', mock_open()) as mocked_file:
        create_integration_template()
        mocked_file.assert_called_once_with('integration_template.py', 'w')
        # Check that something was written
        mocked_file().write.assert_called()

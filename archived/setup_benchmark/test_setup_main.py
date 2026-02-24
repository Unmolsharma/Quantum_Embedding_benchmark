import pytest
from unittest.mock import patch
from setup_benchmark import main

@patch('setup_benchmark.setup_minorminer')
@patch('setup_benchmark.setup_atom')
@patch('setup_benchmark.setup_charme')
@patch('setup_benchmark.setup_oct_based')
@patch('setup_benchmark.setup_general_dependencies')
@patch('setup_benchmark.create_integration_template')
def test_setup_main(mock_template, mock_deps, mock_oct, mock_charme, mock_atom, mock_minor):
    """
    Tests `setup_main` behavior.
    Equivalence Class: General behavior / Default path
    """
    main()
    
    mock_minor.assert_called_once()
    mock_atom.assert_called_once()
    mock_charme.assert_called_once()
    mock_oct.assert_called_once()
    mock_deps.assert_called_once()
    mock_template.assert_called_once()

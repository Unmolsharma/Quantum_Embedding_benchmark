import pytest
from unittest.mock import patch
from setup_benchmark import setup_atom, setup_charme, setup_oct_based

@patch('setup_benchmark.run_command')
@patch('setup_benchmark.Path.exists')
def test_setup_atom_already_exists(mock_exists, mock_run):
    """
    Tests `setup_atom_already_exists` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = True
    setup_atom()
    mock_run.assert_not_called()

@patch('setup_benchmark.run_command')
@patch('setup_benchmark.Path.exists')
@patch('setup_benchmark.Path.mkdir')
def test_setup_atom_new(mock_mkdir, mock_exists, mock_run):
    """
    Tests `setup_atom_new` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = False
    mock_run.return_value = True
    
    setup_atom()
    
    mock_mkdir.assert_called_once()
    assert mock_run.call_count >= 1

@patch('setup_benchmark.run_command')
@patch('setup_benchmark.Path.exists')
def test_setup_charme_already_exists(mock_exists, mock_run):
    """
    Tests `setup_charme_already_exists` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = True
    setup_charme()
    mock_run.assert_not_called()

@patch('setup_benchmark.run_command')
@patch('setup_benchmark.Path.exists')
def test_setup_oct_based_already_exists(mock_exists, mock_run):
    """
    Tests `setup_oct_based_already_exists` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = True
    setup_oct_based()
    mock_run.assert_not_called()

import pytest
from unittest.mock import patch, MagicMock
from embedding_integrations import test_integration

@patch('embedding_integrations.run_atom_embedding')
@patch('embedding_integrations.run_charme_embedding')
@patch('embedding_integrations.run_oct_embedding')
def test_test_integration_all(mock_oct, mock_charme, mock_atom):
    """
    Tests `test_integration_all` behavior.
    Equivalence Class: General behavior / Default path
    """
    # Setup mocks to return fake results
    mock_atom.return_value = {'embedding': {0: [0], 1: [1], 2: [2]}, 'time': 0.1}
    mock_charme.return_value = {'embedding': {0: [0], 1: [1], 2: [2]}, 'time': 0.1}
    mock_oct.return_value = {'embedding': {0: [0], 1: [1], 2: [2]}, 'time': 0.1}
    
    # Should not raise any exceptions
    test_integration(method_name="all")
    
    mock_atom.assert_called_once()
    mock_charme.assert_called_once()
    mock_oct.assert_called_once()

@patch('embedding_integrations.run_atom_embedding')
def test_test_integration_single(mock_atom):
    """
    Tests `test_integration_single` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_atom.return_value = None # None simulates failure
    
    test_integration(method_name="atom")
    
    mock_atom.assert_called_once()

def test_test_integration_invalid():
    """
    Tests `test_integration_invalid` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    # Invalid method names should just print and exit cleanly
    test_integration("nonexistent_method")

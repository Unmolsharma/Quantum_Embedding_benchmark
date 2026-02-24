import pytest
from unittest.mock import MagicMock
from integration_template import integrate_atom, integrate_charme, integrate_oct_based

def test_integrate_atom_template():
    """
    Tests `integrate_atom_template` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_benchmark = MagicMock()
    integrate_atom(mock_benchmark)
    
    assert hasattr(mock_benchmark, 'run_atom')
    assert callable(mock_benchmark.run_atom)
    
    # Executing the placeholder should return None
    result = mock_benchmark.run_atom(None)
    assert result is None

def test_integrate_charme_template():
    """
    Tests `integrate_charme_template` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_benchmark = MagicMock()
    integrate_charme(mock_benchmark)
    
    assert hasattr(mock_benchmark, 'run_charme')
    assert callable(mock_benchmark.run_charme)
    
    result = mock_benchmark.run_charme(None)
    assert result is None

def test_integrate_oct_based_template():
    """
    Tests `integrate_oct_based_template` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_benchmark = MagicMock()
    integrate_oct_based(mock_benchmark)
    
    assert hasattr(mock_benchmark, 'run_oct_based')
    assert callable(mock_benchmark.run_oct_based)
    
    result = mock_benchmark.run_oct_based(None)
    assert result is None

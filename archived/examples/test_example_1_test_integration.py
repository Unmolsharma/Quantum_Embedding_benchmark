import pytest
from unittest.mock import patch
from complete_examples import example_1_test_integration

@patch('complete_examples.test_integration')
def test_example_1_test_integration(mock_test_integration):
    """
    Tests `example_1_test_integration` behavior.
    Equivalence Class: General behavior / Default path
    """
    example_1_test_integration()
    mock_test_integration.assert_called_once_with(method_name='all')

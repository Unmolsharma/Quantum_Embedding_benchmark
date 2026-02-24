import pytest
from unittest.mock import patch
from complete_examples import main

@patch('builtins.input')
@patch('complete_examples.example_1_test_integration')
def test_examples_main_choice_1(mock_ex1, mock_input):
    """
    Tests `examples_main_choice_1` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_input.return_value = '1'
    main()
    mock_ex1.assert_called_once()

@patch('builtins.input')
@patch('complete_examples.example_2_benchmark_single_method')
def test_examples_main_choice_2(mock_ex2, mock_input):
    """
    Tests `examples_main_choice_2` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_input.return_value = '2'
    main()
    mock_ex2.assert_called_once()

@patch('builtins.input')
@patch('complete_examples.example_3_benchmark_all_methods')
def test_examples_main_choice_3(mock_ex3, mock_input):
    """
    Tests `examples_main_choice_3` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_input.return_value = '3'
    main()
    mock_ex3.assert_called_once()

@patch('builtins.input')
@patch('complete_examples.example_4_custom_problems')
def test_examples_main_choice_4(mock_ex4, mock_input):
    """
    Tests `examples_main_choice_4` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_input.return_value = '4'
    main()
    mock_ex4.assert_called_once()

@patch('builtins.input')
@patch('complete_examples.example_5_compare_hardware_graphs')
def test_examples_main_choice_5(mock_ex5, mock_input):
    """
    Tests `examples_main_choice_5` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_input.return_value = '5'
    main()
    mock_ex5.assert_called_once()

@patch('builtins.input')
@patch('complete_examples.example_1_test_integration')
@patch('complete_examples.example_2_benchmark_single_method')
def test_examples_main_choice_6_all(mock_ex2, mock_ex1, mock_input):
    """
    Tests `examples_main_choice_6_all` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_input.return_value = '6'
    main()
    mock_ex1.assert_called_once()
    mock_ex2.assert_called_once()
    
@patch('builtins.input')
def test_examples_main_quit(mock_input):
    """
    Tests `examples_main_quit` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_input.return_value = 'q'
    main()

@patch('builtins.input')
def test_examples_main_invalid(mock_input):
    """
    Tests `examples_main_invalid` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    mock_input.return_value = '9'
    main()

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from inspect_repos import analyze_repository

@patch('inspect_repos.Path.exists')
@patch('inspect_repos.find_main_files')
@patch('inspect_repos.find_python_files')
@patch('inspect_repos.find_embedding_functions')
def test_analyze_repository(mock_find_funcs, mock_find_py, mock_find_main, mock_exists):
    """
    Tests `analyze_repository` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = True
    
    mock_find_main.return_value = [Path("main.py")]
    mock_find_py.return_value = [Path("embed.py")]
    mock_find_funcs.return_value = ["my_embed"]
    
    with patch('builtins.print') as mock_print:
        analyze_repository(Path("fake_repo"), "FakeRepo")
        
        # Verify it prints analysis information
        assert mock_print.call_count > 10

@patch('inspect_repos.Path.exists')
def test_analyze_repository_not_found(mock_exists):
    """
    Tests `analyze_repository_not_found` behavior.
    Equivalence Class: General behavior / Default path
    """
    mock_exists.return_value = False
    
    with patch('builtins.print') as mock_print:
        analyze_repository(Path("fake_repo"), "FakeRepo")
        # Ensure error message is printed
        mock_print.assert_any_call("❌ Repository not found at fake_repo")

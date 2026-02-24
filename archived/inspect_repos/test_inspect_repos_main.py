import pytest
from unittest.mock import patch
from inspect_repos import main

@patch('inspect_repos.analyze_repository')
def test_inspect_repos_main(mock_analyze):
    """
    Tests `inspect_repos_main` behavior.
    Equivalence Class: General behavior / Default path
    """
    main()
    assert mock_analyze.call_count == 3

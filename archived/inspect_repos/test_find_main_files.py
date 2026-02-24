import pytest
from pathlib import Path
from inspect_repos import find_main_files

def test_find_main_files(tmp_path):
    """
    Tests `find_main_files` behavior.
    Equivalence Class: General behavior / Default path
    """
    f1 = tmp_path / "main.py"
    f1.write_text("print('hi')")
    
    d1 = tmp_path / "src"
    d1.mkdir()
    f2 = d1 / "example.py"
    f2.write_text("print('hi')")
    
    f3 = tmp_path / "other.py"
    f3.write_text("print('hi')")
    
    mains = find_main_files(tmp_path)
    
    assert len(mains) == 2
    assert f1 in mains
    assert f2 in mains
    assert f3 not in mains

def test_find_main_files_empty(tmp_path):
    """
    Tests `find_main_files_empty` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    f1 = tmp_path / "util.py"
    f1.write_text("print('hi')")
    
    mains = find_main_files(tmp_path)
    assert len(mains) == 0

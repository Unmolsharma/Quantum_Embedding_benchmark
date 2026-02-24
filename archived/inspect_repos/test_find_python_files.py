import pytest
from pathlib import Path
from inspect_repos import find_python_files

def test_find_python_files(tmp_path):
    """
    Tests `find_python_files` behavior.
    Equivalence Class: General behavior / Default path
    """
    # Setup temp directory structure
    d1 = tmp_path / "dir1"
    d1.mkdir()
    f1 = d1 / "file1.py"
    f1.write_text("print('hi')")
    
    f2 = tmp_path / "file2.py"
    f2.write_text("print('hi')")
    
    f3 = tmp_path / "file3.txt"
    f3.write_text("not python")
    
    # Hidden dir should be ignored
    d2 = tmp_path / ".hidden"
    d2.mkdir()
    f4 = d2 / "hidden.py"
    f4.write_text("print('hi')")
    
    files = find_python_files(tmp_path, max_depth=2)
    
    assert len(files) == 2
    assert f1 in files
    assert f2 in files
    assert f3 not in files
    assert f4 not in files

def test_find_python_files_max_depth(tmp_path):
    """
    Tests `find_python_files_max_depth` behavior.
    Equivalence Class: General behavior / Default path
    """
    # depth 0
    d1 = tmp_path / "dir1"
    d1.mkdir()
    # depth 1
    d2 = d1 / "dir2"
    d2.mkdir()
    # depth 2
    f1 = d2 / "deep.py"
    f1.write_text("print('hi')")
    
    assert len(find_python_files(tmp_path, max_depth=1)) == 0
    assert len(find_python_files(tmp_path, max_depth=2)) == 1

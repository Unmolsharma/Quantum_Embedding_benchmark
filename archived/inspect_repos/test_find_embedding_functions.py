import pytest
from pathlib import Path
from inspect_repos import find_embedding_functions

def test_find_embedding_functions(tmp_path):
    """
    Tests `find_embedding_functions` behavior.
    Equivalence Class: General behavior / Default path
    """
    f = tmp_path / "funcs.py"
    content = """
def test_embed(g): pass
def do_minor_mapping(g): pass
class GraphEmbedder: pass
class MinorMapper: pass
def unrelated(): pass
class UnrelatedClass: pass
"""
    f.write_text(content)
    
    functions = find_embedding_functions(f)
    
    assert len(functions) == 4
    assert set(functions) == {"test_embed", "do_minor_mapping", "GraphEmbedder", "MinorMapper"}

def test_find_embedding_functions_empty(tmp_path):
    """
    Tests `find_embedding_functions_empty` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    f = tmp_path / "empty.py"
    f.write_text("def unrelated(): pass")
    
    assert len(find_embedding_functions(f)) == 0

def test_find_embedding_functions_exception(tmp_path):
    """
    Tests `find_embedding_functions_exception` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    # Pass non-existent file
    assert len(find_embedding_functions(tmp_path / "does_not_exist.py")) == 0

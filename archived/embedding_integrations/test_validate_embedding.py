import pytest
import networkx as nx
from embedding_integrations import validate_embedding

def test_validate_embedding_valid():
    """
    Tests `validate_embedding_valid` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    source_graph = nx.cycle_graph(3) # 0-1-2-0
    target_graph = nx.cycle_graph(5) # 0-1-2-3-4-0
    
    # Valid embedding: 
    # v0 -> {0}
    # v1 -> {1, 2} (connected)
    # v2 -> {4}
    # Edge (0, 1) -> maps to (0, 1) or (0, 2). 0 adjacent to 1. Yes.
    # Edge (1, 2) -> maps to (1, 4) or (2, 4). Neither is an edge in original, wait!
    # target cycle is 0-1-2-3-4-0. Edges: (0,1), (1,2), (2,3), (3,4), (4,0).
    # Since Source graph has 1-2, Target graph needs edge between {1,2} and {4}.
    # Wait, is 1/2 adjacent to 4? Edge (4,0) exists, (3,4) exists. Not 1 or 2. 
    # Let's use a simpler known valid embedding.
    
    s = nx.path_graph(2) # 0-1
    t = nx.path_graph(3) # 0-1-2
    embedding = {0: [0, 1], 1: [2]} # Chain for 0 is connected (0-1). 1 uses 2. Edge between 0's chain and 1's chain exists? Yes, (1,2) in target.
    
    is_valid, reason = validate_embedding(embedding, s, t)
    assert is_valid is True
    assert reason == "Valid"

def test_validate_embedding_missing_node():
    """
    Tests `validate_embedding_missing_node` behavior.
    Equivalence Class: General behavior / Default path
    """
    s = nx.path_graph(3) # 0, 1, 2
    t = nx.path_graph(4)
    embedding = {0: [0], 1: [1]} # missing source node 2
    
    is_valid, reason = validate_embedding(embedding, s, t)
    assert is_valid is False
    assert "Missing" in reason

def test_validate_embedding_empty_chain():
    """
    Tests `validate_embedding_empty_chain` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    s = nx.path_graph(2)
    t = nx.path_graph(3)
    embedding = {0: [0], 1: []}
    
    is_valid, reason = validate_embedding(embedding, s, t)
    assert is_valid is False
    assert "Empty chain" in reason

def test_validate_embedding_disconnected_chain():
    """
    Tests `validate_embedding_disconnected_chain` behavior.
    Equivalence Class: General behavior / Default path
    """
    s = nx.path_graph(2)
    t = nx.path_graph(4) # 0-1-2-3
    embedding = {0: [0, 2], 1: [3]} # {0,2} is disconnected
    
    is_valid, reason = validate_embedding(embedding, s, t)
    assert is_valid is False
    assert "disconnected" in reason.lower()

def test_validate_embedding_missing_edge():
    """
    Tests `validate_embedding_missing_edge` behavior.
    Equivalence Class: General behavior / Default path
    """
    s = nx.cycle_graph(3) # 0-1-2-0
    t = nx.path_graph(4) # 0-1-2-3. No cycle exists! Cannot embed K3 into path_graph.
    # We will try a mapping
    embedding = {0: [0], 1: [1], 2: [3]}
    # Source edge (0,2) doesn't exist among target chains
    is_valid, reason = validate_embedding(embedding, s, t)
    assert is_valid is False
    assert "Edge" in reason and "not preserved" in reason

def test_validate_embedding_overlap():
    """
    Tests `validate_embedding_overlap` behavior.
    Equivalence Class: General behavior / Default path
    """
    s = nx.path_graph(2)
    t = nx.path_graph(3)
    embedding = {0: [0, 1], 1: [1, 2]} # 1 is in both chains
    
    is_valid, reason = validate_embedding(embedding, s, t)
    assert is_valid is False
    assert "Overlapping" in reason

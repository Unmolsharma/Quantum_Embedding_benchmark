import pytest
import networkx as nx
from embedding_integrations import networkx_to_adjacency_dict

def test_networkx_to_adjacency_dict():
    """
    Tests `networkx_to_adjacency_dict` behavior.
    Equivalence Class: General behavior / Default path
    """
    # Setup
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
    
    # Execute
    adj_dict = networkx_to_adjacency_dict(G)
    
    # Assert
    assert isinstance(adj_dict, dict)
    assert set(adj_dict.keys()) == {0, 1, 2, 3, 4}
    assert set(adj_dict[0]) == {1, 2}
    assert set(adj_dict[1]) == {0, 2}
    assert set(adj_dict[2]) == {1, 0}
    assert set(adj_dict[3]) == {4}
    assert set(adj_dict[4]) == {3}

def test_networkx_to_adjacency_dict_empty():
    """
    Tests `networkx_to_adjacency_dict_empty` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    G = nx.Graph()
    adj_dict = networkx_to_adjacency_dict(G)
    assert adj_dict == {}

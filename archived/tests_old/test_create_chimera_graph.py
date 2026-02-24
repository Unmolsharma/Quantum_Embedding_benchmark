import pytest
import networkx as nx
from embedding_benchmark import create_chimera_graph

def test_create_chimera_graph_default():
    """
    Tests `create_chimera_graph_default` behavior.
    Equivalence Class: General behavior / Default path
    """
    graph = create_chimera_graph()
    assert isinstance(graph, nx.Graph)
    # 4*4*4*2 = 128 nodes if dwave_networkx is used or fallback
    assert len(graph.nodes) > 0

def test_create_chimera_graph_custom_params():
    """
    Tests `create_chimera_graph_custom_params` behavior.
    Equivalence Class: General behavior / Default path
    """
    graph = create_chimera_graph(m=2, n=2, t=4)
    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes) > 0

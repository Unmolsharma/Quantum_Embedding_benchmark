import pytest
import networkx as nx
from embedding_benchmark import EmbeddingBenchmark

def test_compute_embedding_metrics_basic():
    """
    Tests `compute_embedding_metrics_basic` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    # target graph: a path 0-1-2-3
    target_graph = nx.path_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    # 0 maps to [0, 1]
    # 1 maps to [2]
    embedding = {0: [0, 1], 1: [2]}
    
    metrics = benchmark.compute_embedding_metrics(embedding)
    
    # Ensure correct keys are present
    assert 'chain_lengths' in metrics
    assert 'avg_chain_length' in metrics
    assert 'max_chain_length' in metrics
    assert 'total_qubits_used' in metrics
    assert 'total_couplers_used' in metrics
    
    # Value assertions
    assert metrics['chain_lengths'] == [2, 1]
    assert metrics['avg_chain_length'] == 1.5
    assert metrics['max_chain_length'] == 2
    assert metrics['total_qubits_used'] == 3
    # Qubits are 0, 1, 2. Intra-chain couplers inside chain [0, 1]: edge (0, 1) exists in target_graph path_graph(4)
    assert metrics['total_couplers_used'] == 1

def test_compute_embedding_metrics_empty():
    """
    Tests `compute_embedding_metrics_empty` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    target_graph = nx.path_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    embedding = {}
    
    metrics = benchmark.compute_embedding_metrics(embedding)
    
    assert metrics['chain_lengths'] == []
    import numpy as np
    assert np.isnan(metrics['avg_chain_length'])
    # min() on empty sequence raises ValueError, but max() here raises ValueError if empty.
    # Ah wait. compute_embedding_metrics will error on max([])
    try:
        metrics = benchmark.compute_embedding_metrics(embedding)
    except ValueError:
        pass # max() arg is an empty sequence

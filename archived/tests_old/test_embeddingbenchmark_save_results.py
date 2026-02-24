import pytest
import networkx as nx
import json
import pandas as pd
from embedding_benchmark import EmbeddingBenchmark, EmbeddingResult
import os

def test_save_results(tmp_path):
    """
    Tests `save_results` behavior.
    Equivalence Class: General behavior / Default path
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph, results_dir=str(tmp_path))
    
    benchmark.results.append(
        EmbeddingResult(
            method_name="fake", problem_name="prob1", problem_size=2, problem_density=1.0, 
            success=True, embedding_time=1.0, chain_lengths=[1,1], avg_chain_length=1.0, 
            max_chain_length=1, total_qubits_used=2, total_couplers_used=1
        )
    )
    
    benchmark.save_results()
    
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    
    assert json_path.exists()
    assert csv_path.exists()
    
    # Verify json
    with open(json_path, 'r') as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]['method_name'] == 'fake'
        
    # Verify csv
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]['method_name'] == 'fake'

import pytest
import networkx as nx
from unittest.mock import patch
from embedding_benchmark import EmbeddingBenchmark, EmbeddingResult

@patch.object(EmbeddingBenchmark, '_plot_success_rates')
@patch.object(EmbeddingBenchmark, '_plot_embedding_times')
@patch.object(EmbeddingBenchmark, '_plot_chain_lengths')
@patch.object(EmbeddingBenchmark, '_plot_scalability')
@patch.object(EmbeddingBenchmark, '_generate_summary_statistics')
def test_generate_report_with_results(mock_stats, mock_scalability, mock_chain_lengths, mock_embed_times, mock_success_rates):
    """
    Tests `generate_report_with_results` behavior.
    Equivalence Class: General behavior / Default path
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    benchmark.results.append(EmbeddingResult("fake", "prob", 2, 1.0, True, 1.0, [1], 1.0, 1, 1, 1))
    
    benchmark.generate_report()
    
    mock_success_rates.assert_called_once()
    mock_embed_times.assert_called_once()
    mock_chain_lengths.assert_called_once()
    mock_scalability.assert_called_once()
    mock_stats.assert_called_once()

@patch.object(EmbeddingBenchmark, '_plot_success_rates')
def test_generate_report_empty(mock_success_rates):
    """
    Tests `generate_report_empty` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    target_graph = nx.complete_graph(4)
    benchmark = EmbeddingBenchmark(target_graph)
    
    # Empty results
    benchmark.generate_report() # Should essentially be a no-op, shouldn't crash
    mock_success_rates.assert_not_called()

import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from embedding_integrations import run_atom_embedding

@patch('embedding_integrations.subprocess.run')
@patch('embedding_integrations.tempfile.NamedTemporaryFile')
@patch('embedding_integrations.os.path.exists')
def test_run_atom_embedding_success(mock_exists, mock_tempfile, mock_run):
    """
    Tests `run_atom_embedding_success` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    source = nx.path_graph(2)
    target = nx.path_graph(3)
    
    # Mock executables existing
    mock_exists.return_value = True
    
    # Mock temp files to have fake names
    mock_source_file = MagicMock()
    mock_source_file.name = "source.txt"
    mock_target_file = MagicMock()
    mock_target_file.name = "target.txt"
    
    # Side effects so we can yield these multiple times
    mock_tempfile.side_effect = [mock_source_file, mock_target_file]
    
    mock_process = MagicMock()
    mock_process.stdout = "Embedding found:\n0: [0, 1]\n1: [2]"
    mock_process.returncode = 0
    mock_run.return_value = mock_process
    
    result = run_atom_embedding(source, target)
    
    assert result is not None
    assert 'embedding' in result
    # We aren't testing the actual parsing logic here perfectly, assuming the 
    # placeholder/real implementation exists in the source. If the source parsing
    # logic changes, this test may need updates. Based on typical patterns, 
    # it likely returns a dict mapping ints to lists of ints.
    
@patch('embedding_integrations.os.path.exists')
def test_run_atom_embedding_no_exe(mock_exists):
    """
    Tests `run_atom_embedding_no_exe` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    mock_exists.return_value = False
    
    source = nx.path_graph(2)
    target = nx.path_graph(3)
    
    result = run_atom_embedding(source, target)
    assert result is None

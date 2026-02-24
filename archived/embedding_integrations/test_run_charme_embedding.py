import pytest
import networkx as nx
from unittest.mock import patch, MagicMock
from embedding_integrations import run_charme_embedding

@patch('embedding_integrations.subprocess.run')
@patch('embedding_integrations.tempfile.NamedTemporaryFile')
@patch('embedding_integrations.os.path.exists')
def test_run_charme_embedding_success(mock_exists, mock_tempfile, mock_run):
    """
    Tests `run_charme_embedding_success` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    source = nx.path_graph(2)
    target = nx.path_graph(3)
    
    mock_exists.return_value = True
    
    mock_source_file = MagicMock()
    mock_source_file.name = "source.json"
    mock_target_file = MagicMock()
    mock_target_file.name = "target.json"
    
    mock_tempfile.side_effect = [mock_source_file, mock_target_file]
    
    mock_process = MagicMock()
    mock_process.stdout = '{"0": [0], "1": [1, 2]}'
    mock_process.returncode = 0
    mock_run.return_value = mock_process
    
    result = run_charme_embedding(source, target)
    
    assert result is not None

@patch('embedding_integrations.os.path.exists')
def test_run_charme_embedding_no_exe(mock_exists):
    """
    Tests `run_charme_embedding_no_exe` behavior.
    Equivalence Class: Empty/Zero boundary conditions
    """
    mock_exists.return_value = False
    
    source = nx.path_graph(2)
    target = nx.path_graph(3)
    
    result = run_charme_embedding(source, target)
    assert result is None

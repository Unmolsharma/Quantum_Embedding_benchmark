import pytest
from unittest.mock import patch, MagicMock
from setup_benchmark import run_command

@patch('setup_benchmark.subprocess.run')
def test_run_command_success(mock_run):
    """
    Tests `run_command_success` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_run.return_value = mock_process
    
    result = run_command("echo hello")
    assert result is True
    
    mock_run.assert_called_once_with("echo hello", shell=True, check=False, cwd=None)

@patch('setup_benchmark.subprocess.run')
def test_run_command_failure(mock_run):
    """
    Tests `run_command_failure` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_run.return_value = mock_process
    
    result = run_command("false")
    assert result is False

@patch('setup_benchmark.subprocess.run')
def test_run_command_exception(mock_run):
    """
    Tests `run_command_exception` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    mock_run.side_effect = Exception("System error")
    
    result = run_command("ls")
    assert result is False

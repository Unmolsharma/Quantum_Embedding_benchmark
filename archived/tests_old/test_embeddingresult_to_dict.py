import pytest
from embedding_benchmark import EmbeddingResult

def test_embeddingresult_to_dict_basic():
    """
    Tests `embeddingresult_to_dict_basic` behavior.
    Equivalence Class: Standard valid inputs / Success
    """
    # Setup
    result = EmbeddingResult(
        method_name="test_method",
        problem_name="test_problem",
        problem_size=10,
        problem_density=0.5,
        success=True,
        embedding_time=1.23,
        chain_lengths=[1, 2, 3],
        avg_chain_length=2.0,
        max_chain_length=3,
        total_qubits_used=6,
        total_couplers_used=5,
        error_message=None
    )
    
    # Execute
    result_dict = result.to_dict()
    
    # Assert
    assert isinstance(result_dict, dict)
    assert result_dict["method_name"] == "test_method"
    assert result_dict["problem_size"] == 10
    assert result_dict["chain_lengths"] == [1, 2, 3]
    assert result_dict["success"] is True

def test_embeddingresult_to_dict_with_error():
    """
    Tests `embeddingresult_to_dict_with_error` behavior.
    Equivalence Class: Invalid inputs / Exceptions / Failures
    """
    result = EmbeddingResult(
        method_name="fail_method",
        problem_name="fail_problem",
        problem_size=5,
        problem_density=0.1,
        success=False,
        embedding_time=60.0,
        chain_lengths=[],
        avg_chain_length=0.0,
        max_chain_length=0,
        total_qubits_used=0,
        total_couplers_used=0,
        error_message="Timeout exceeded"
    )
    
    result_dict = result.to_dict()
    assert result_dict["success"] is False
    assert result_dict["error_message"] == "Timeout exceeded"

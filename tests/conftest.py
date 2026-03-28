"""
tests/conftest.py
=================
Shared pytest fixtures available to all test modules.
"""
import networkx as nx
import dwave_networkx as dnx
import pytest


@pytest.fixture(scope="session")
def chimera_session():
    """Chimera(4,4,4) — 128 qubits. Session-scoped to avoid rebuilding."""
    return dnx.chimera_graph(4, 4, 4)


@pytest.fixture
def chimera():
    return dnx.chimera_graph(4, 4, 4)


@pytest.fixture
def K4():
    return nx.complete_graph(4)


@pytest.fixture
def K8():
    return nx.complete_graph(8)


@pytest.fixture
def cycle10():
    return nx.cycle_graph(10)


@pytest.fixture
def petersen():
    return nx.petersen_graph()

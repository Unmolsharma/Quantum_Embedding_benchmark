"""
ember_qc/algorithms/charme.py
==============================
CHARME — Python RL framework stub.

Full integration is pending. The algorithm code lives in algorithms/charme/
in the repository but is not yet callable via the ember-qc package API.
"""

import logging

from ember_qc.registry import EmbeddingAlgorithm, register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("charme")
class CharmeAlgorithm(EmbeddingAlgorithm):
    """CHARME — Python RL framework. Not callable via subprocess."""

    _requires = ["torch", "karateclub"]
    _install_instruction = "CHARME package integration is pending — algorithm not yet callable from ember-qc."

    def embed(self, source_graph, target_graph, timeout=60.0, **kwargs):
        logger.warning(
            "CHARME is a Python RL framework and has not been wrapped yet. "
            "Import its Python modules directly to use it."
        )
        return {'embedding': {}, 'time': 0.0, 'success': False, 'status': 'FAILURE'}

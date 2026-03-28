"""
ember_qc/algorithms/_loader.py
================================
Dynamic loader for user-defined custom algorithms.

Scans the user algorithms directory (~/.local/share/ember-qc/algorithms/)
for .py files and imports each one. Any file that uses @register_algorithm
will self-register when imported.

Called once from algorithms/__init__.py at package import time.
"""

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_user_algorithms() -> None:
    """Import every .py file in the user algorithms directory.

    Files that fail to import are logged as warnings and skipped — a broken
    custom algorithm must not prevent the rest of the package from loading.
    """
    try:
        from ember_qc._paths import get_user_algo_dir
        algo_dir = get_user_algo_dir()
    except Exception:
        return

    if not algo_dir.exists():
        return

    for py_file in sorted(algo_dir.glob("*.py")):
        module_name = f"ember_qc.algorithms.user.{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.debug("Loaded custom algorithm file: %s", py_file.name)
        except Exception as exc:
            logger.warning(
                "Failed to load custom algorithm %s: %s — skipping.",
                py_file.name, exc,
            )

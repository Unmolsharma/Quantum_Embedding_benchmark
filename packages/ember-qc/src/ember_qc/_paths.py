"""
ember_qc/_paths.py
==================
User directory resolution for ember-qc.

All paths that point into the user data directory go through this module.
Other modules (config, registry wrappers for binaries) import from here
rather than computing paths independently.

User data root (OS-conventional, survives pip upgrades):
  Linux:   ~/.local/share/ember-qc/
  macOS:   ~/Library/Application Support/ember-qc/
  Windows: C:\\Users\\<user>\\AppData\\Local\\ember-qc\\ember-qc\\
"""

import os
from pathlib import Path


def get_user_dir() -> Path:
    """
    Return the ember-qc user data root directory.

    Uses platformdirs if available; falls back to OS conventions otherwise.
    """
    try:
        from platformdirs import user_data_dir
        return Path(user_data_dir("ember-qc", "ember-qc"))
    except ImportError:
        if os.name == "nt":
            base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return base / "ember-qc" / "ember-qc"
        elif os.uname().sysname == "Darwin":
            return Path.home() / "Library" / "Application Support" / "ember-qc"
        else:
            xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
            return Path(xdg) / "ember-qc"


def get_user_algo_dir() -> Path:
    """Return the directory for user-defined custom algorithm files."""
    return get_user_dir() / "algorithms"


def get_user_binary_dir() -> Path:
    """Return the directory where compiled C++ binaries are installed."""
    return get_user_dir() / "binaries"


def get_user_config_path() -> Path:
    """Return the path to config.json (may not exist yet)."""
    return get_user_dir() / "config.json"


def get_user_unfinished_dir() -> Path:
    """Return the staging directory for in-progress benchmark runs."""
    return get_user_dir() / "runs_unfinished"


def get_user_graphs_dir() -> Path:
    """Return the user-local graph cache directory.

    Persists across pip upgrades. Contains:
      - local_index.json  (tracks cached graphs, their hashes, and download dates)
      - <id>.json         (graph files fetched from remote or provided by the user)
    """
    return get_user_dir() / "graphs"

"""Command-line path guard shared by the Palmer recipe scripts (REFIMPL-30).

The scripts run under ``python -I``, so they put this directory on ``sys.path`` from ``__file__``.
"""

import os
from pathlib import Path


def path_arg(value: str) -> Path:
    """Resolve a command-line path, refusing a relative one that escapes the working directory.

    Same guard as benchmarks/cli_multiprocessing.py (pythonsecurity:S8707); absolute paths are
    the documented usage and pass through after normalisation.
    """
    resolved = os.path.realpath(value)
    base = os.path.realpath(os.getcwd())
    if not os.path.isabs(value) and resolved != base and not resolved.startswith(base + os.sep):
        raise SystemExit(f"{value!r} is outside the working directory")
    return Path(resolved)

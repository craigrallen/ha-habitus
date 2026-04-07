"""Habitus utility helpers.

Shared low-level utilities used across multiple modules.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

log = logging.getLogger("habitus")


def atomic_write(path: str | os.PathLike, data: Any, *, indent: int = 2) -> None:
    """Atomically write JSON data to *path*.

    Writes to a temporary file in the same directory first, then calls
    ``os.replace()`` to atomically swap it into place.  This prevents
    corrupt/partial JSON files if the process is killed mid-write.

    Args:
        path: Destination file path.
        data: JSON-serialisable value.
        indent: JSON indentation (default 2).

    Raises:
        OSError: If the directory cannot be created or the file cannot be
                 written.  The destination file is not modified on failure.
    """
    path = str(path)
    dest_dir = os.path.dirname(path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    # Write to a sibling .tmp file, then atomically replace
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=indent, default=str)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the temp file so we don't leave strays
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

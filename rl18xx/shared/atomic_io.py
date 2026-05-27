"""Atomic JSON write helpers.

Multiple components in this codebase write small status/config JSON files
that are concurrently polled by readers (loop status, self-play game
progress, dashboard config, pretraining progress, ...). Using a plain
``open(path, "w") + json.dump`` pattern truncates the destination *before*
the new bytes land, so a reader who polls during the write can observe an
empty or half-written file and crash with ``json.JSONDecodeError``.

The helper here writes to a temp file in the same directory and then uses
``os.replace`` (an atomic rename on POSIX and Windows for files on the same
filesystem) to swap it into place. Readers always see either the previous
fully-written file or the new fully-written file -- never a partial state.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _default_jsonable(obj: Any) -> Any:
    """Best-effort coercion of common non-JSON-native types.

    Several callers pass dicts that pick up numpy scalars / arrays from
    upstream computations (e.g. self-play game results stored as
    ``np.ndarray``). Without coercion ``json.dump`` raises ``TypeError`` and
    the status file ends up missing entirely. Handling these inline keeps
    every caller from needing its own ``default=`` shim.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def atomic_write_json(path: str | os.PathLike, data: Any, *, indent: int = 2, default=None) -> None:
    """Write ``data`` as JSON to ``path`` atomically (tmp + ``os.replace``).

    Concurrent readers see either the old or new contents -- never a
    partial write. The temp file is created in the destination's parent
    directory so the final rename stays within a single filesystem.

    ``default`` falls back to a coercer that handles numpy scalars / arrays.
    """
    if default is None:
        default = _default_jsonable
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent, default=default)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise

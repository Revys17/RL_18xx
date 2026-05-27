"""Tests for the atomic JSON write helper.

Truly proving the absence of partial reads requires racing threads against the
writer, which is fragile to test. These tests instead verify the observable
guarantees the helper is supposed to provide: the destination ends up with the
correct content, overwrites replace the old content cleanly, no temp files are
left behind, and parent directories are created on demand.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from rl18xx.shared.atomic_io import atomic_write_json


def test_atomic_write_json_creates_file_with_correct_content(tmp_path: Path):
    p = tmp_path / "test.json"
    atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
    with open(p) as f:
        assert json.load(f) == {"a": 1, "b": [1, 2, 3]}


def test_atomic_write_json_overwrites_cleanly(tmp_path: Path):
    p = tmp_path / "test.json"
    atomic_write_json(p, {"a": 1, "b": [1, 2, 3]})
    atomic_write_json(p, {"c": 4})
    with open(p) as f:
        assert json.load(f) == {"c": 4}


def test_atomic_write_json_leaves_no_tmp_files(tmp_path: Path):
    p = tmp_path / "test.json"
    atomic_write_json(p, {"a": 1})
    atomic_write_json(p, {"b": 2})
    # No leftover .tmp files in the destination dir.
    assert list(tmp_path.glob(".test.json.*.tmp")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "nested" / "dirs" / "test.json"
    atomic_write_json(p, {"x": 1})
    assert p.exists()
    with open(p) as f:
        assert json.load(f) == {"x": 1}


def test_atomic_write_json_accepts_string_path(tmp_path: Path):
    p = tmp_path / "test.json"
    atomic_write_json(str(p), {"k": "v"})
    with open(p) as f:
        assert json.load(f) == {"k": "v"}


def test_atomic_write_json_respects_default_param(tmp_path: Path):
    """`default=str` is used by loop status writes to coerce datetime/etc."""
    p = tmp_path / "test.json"
    # Object with no default JSON serializer
    class Foo:
        def __str__(self):
            return "foo-str"

    atomic_write_json(p, {"x": Foo()}, default=str)
    with open(p) as f:
        assert json.load(f) == {"x": "foo-str"}


def test_atomic_write_json_coerces_numpy_scalars(tmp_path: Path):
    """Numpy scalars / arrays appear in self-play status writes (per-player
    score floats etc.). The built-in default coercer should handle them so
    callers don't each need their own ``default=`` shim."""
    import numpy as np

    p = tmp_path / "np.json"
    data = {
        "scalar_f": np.float32(0.5),
        "scalar_i": np.int64(7),
        "vector": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }
    atomic_write_json(p, data)
    with open(p) as f:
        loaded = json.load(f)
    assert loaded["scalar_i"] == 7
    assert abs(loaded["scalar_f"] - 0.5) < 1e-6
    assert len(loaded["vector"]) == 3


def test_atomic_write_json_failure_cleans_up_tmp(tmp_path: Path):
    """If json.dump raises mid-write, the tmp file should be removed."""
    p = tmp_path / "test.json"
    # Write an initial valid file we can verify is untouched.
    atomic_write_json(p, {"good": True})

    # Non-serializable value (set) raises during json.dump.
    with pytest.raises(TypeError):
        atomic_write_json(p, {"bad": {1, 2, 3}})

    # Destination should still contain the original content.
    with open(p) as f:
        assert json.load(f) == {"good": True}
    # And no temp file should be leaking.
    assert list(tmp_path.glob(".test.json.*.tmp")) == []


def test_atomic_write_json_overwrite_is_atomic_replace(tmp_path: Path):
    """Verify the implementation actually goes through os.replace, not a
    write-then-rewrite. This guarantees readers never observe a truncated file.
    """
    p = tmp_path / "test.json"
    atomic_write_json(p, {"first": 1})

    with patch("rl18xx.shared.atomic_io.os.replace", wraps=os.replace) as mock_replace:
        atomic_write_json(p, {"second": 2})
        assert mock_replace.call_count == 1
        # Verify destination is the second arg.
        args, _ = mock_replace.call_args
        assert Path(args[1]) == p

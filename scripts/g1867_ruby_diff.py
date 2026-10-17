"""Rust-vs-Ruby per-action state diff for recorded 1867 games — a thin
wrapper over the title-generic scripts/ruby_diff.py (kept so the
documented 1867 commands keep working). See that script for usage.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_ruby_diff.py \\
        --corpus-dir human_games/1867 --audit logs/g1867/audit.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ruby_diff  # noqa: E402

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.argv = [sys.argv[0], "--title", "1867"] + sys.argv[1:]
    sys.exit(ruby_diff.main())

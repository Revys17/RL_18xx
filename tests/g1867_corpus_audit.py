"""1867 human-corpus outcome audit — a thin wrapper over the title-generic
tests/corpus_audit.py (kept so the documented 1867 gate command and its
JSON output stay unchanged).

Usage:
    uv run python tests/g1867_corpus_audit.py [--glob 'human_games/1867/*.json']
        [--limit N] [--json /tmp/g1867_audit.json]
        [--ruby-results tests/g1867_ruby_head_results.json]

Games with optional rules (grid_market) are skipped, as before: 1867
declares no optional rules. See tests/corpus_audit.py for the buckets.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.corpus_audit import align_recorded_keys, main  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] + ["--title", "1867", "--skip-optional-rules"], default_title="1867"))

"""Forwarder so ``python -m hukuk_rag ...`` works."""

import sys
from pathlib import Path

# Make src/ importable when running from repo root without installing.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

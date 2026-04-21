"""
Vercel FastAPI entrypoint (official pattern: root `main.py` exporting `app`).
Implementation lives in `backend/app.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent / "backend"
if _backend.is_dir() and str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app import app  # noqa: E402

__all__ = ["app"]

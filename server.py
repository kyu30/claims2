"""
Vercel FastAPI entrypoint: exposes the `app` ASGI instance from `backend/app.py`.
See https://vercel.com/docs/frameworks/backend/fastapi
"""
from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent / "backend"
if _backend.is_dir() and str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app import app  # noqa: E402

__all__ = ["app"]

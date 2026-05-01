"""
Vercel FastAPI entry: import the ASGI app as a real package so the bundler
includes ``backend/`` (dynamic ``sys.path`` hacks are not traced on deploy).
Local: ``uvicorn main:app`` or ``uvicorn backend.app:app`` from repo root.
"""
from __future__ import annotations

from backend.app import app

__all__ = ["app"]

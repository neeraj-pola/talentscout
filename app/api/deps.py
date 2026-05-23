# app/api/deps.py
"""Shared FastAPI dependencies — one-time startup logic."""
from app.storage.db import init_db


def ensure_db_initialized():
    """Called once at startup to make sure SQLite tables exist."""
    init_db()
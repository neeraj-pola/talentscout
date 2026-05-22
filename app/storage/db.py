# app/storage/db.py
"""SQLAlchemy setup. Single source of truth for the SQLite engine + session."""
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings

engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False},  # SQLite + threads (Streamlit)
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ============================================================
# ORM Models — these are the DB tables. Pydantic models stay separate.
# ============================================================

class JDRow(Base):
    __tablename__ = "jds"

    id = Column(String, primary_key=True)  # UUID as str
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    must_have_skills = Column(Text, nullable=False)   # JSON-encoded list
    nice_to_have_skills = Column(Text, default="[]")  # JSON-encoded list
    min_years_experience = Column(Integer, nullable=False)
    max_years_experience = Column(Integer, nullable=True)
    location = Column(String, nullable=False)
    remote_ok = Column(Boolean, default=False)
    employment_type = Column(String, nullable=False)
    target_hiring_date = Column(String, nullable=False)  # ISO date
    status = Column(String, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    closed_by = Column(String, nullable=True)
    closed_with_candidate_id = Column(String, nullable=True)

    # Cached blobs — convenient for the UI; not the source of truth for agents
    parsed_jd_json = Column(Text, nullable=True)
    shortlist_json = Column(Text, nullable=True)
    top_pick_json = Column(Text, nullable=True)
    outreach_json = Column(Text, nullable=True)


class AuditRow(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    jd_id = Column(String, nullable=False, index=True)
    candidate_id = Column(String, nullable=False)
    closed_by = Column(String, nullable=False)
    closed_at = Column(DateTime, default=datetime.utcnow)
    justification = Column(Text, nullable=False)
    final_ranking_snapshot = Column(Text, nullable=False)  # JSON list of UUIDs
    total_cost_usd = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    total_llm_calls = Column(Integer, default=0)


class CostRow(Base):
    """Per-LLM-call cost record. One row per LLM invocation."""
    __tablename__ = "llm_costs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    jd_id = Column(String, nullable=False, index=True)
    agent = Column(String, nullable=False)
    model = Column(String, nullable=False)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    usd = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    ts = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create all tables. Idempotent."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for a DB session with auto-commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
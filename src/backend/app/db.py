"""Database engine and the per-request session dependency."""

from collections.abc import Generator

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

engine = create_engine(settings.database_url, echo=settings.sql_echo)


def create_db_and_tables() -> None:
    """Enable pgvector if available, then create any missing tables."""
    # pgvector is Postgres-only, and other dialects don't understand
    # "CREATE EXTENSION" at all, so the step is skipped for them.
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Yield a database session that is closed after the request."""
    with Session(engine) as session:
        yield session

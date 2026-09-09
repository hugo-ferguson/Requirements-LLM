from collections.abc import Generator

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

engine = create_engine(settings.database_url, echo=settings.sql_echo)


def create_db_and_tables() -> None:
    # pgvector is Postgres-only — the test suite swaps this engine for an
    # in-memory SQLite one (see tests/conftest.py), which doesn't understand
    # "CREATE EXTENSION" at all, so this step is skipped for any other dialect.
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

"""Shared fixtures: a Postgres test database and offline models."""

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

import app.db as db
from app.config import settings
from app.db import get_session
from app.llm_config import get_models_config
from app.main import app

# Overridable, so parallel test runs don't drop each other's tables.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/app_test",
)
TEST_MODELS_CONFIG = Path(__file__).with_name("models.test.json")


@pytest.fixture(name="engine", scope="session")
def engine_fixture():
    """Create the test schema once per run and drop it afterwards."""
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(autouse=True)
def stub_generation_agents() -> Generator[None, None, None]:
    """Point every model at PydanticAI's TestModel and turn voting off.

    Keeps the tests offline.
    """
    original_path = settings.models_config_path
    original_voting = settings.generation_enable_voting

    settings.models_config_path = str(TEST_MODELS_CONFIG)
    settings.generation_enable_voting = False
    get_models_config.cache_clear()

    yield

    settings.models_config_path = original_path
    settings.generation_enable_voting = original_voting
    get_models_config.cache_clear()


@pytest.fixture(name="client")
def client_fixture(engine) -> Generator[TestClient, None, None]:
    """Yield an API client whose writes are rolled back afterwards."""
    original_engine = db.engine
    db.engine = engine

    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    def get_session_override() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_session] = get_session_override
    with TestClient(app) as client:
        yield client

    session.close()
    transaction.rollback()
    connection.close()
    app.dependency_overrides.clear()
    db.engine = original_engine

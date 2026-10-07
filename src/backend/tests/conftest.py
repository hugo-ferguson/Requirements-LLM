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

TEST_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/app_test"


@pytest.fixture(name="engine", scope="session")
def engine_fixture():
	engine = create_engine(TEST_DATABASE_URL)
	with engine.begin() as connection:
		connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
	SQLModel.metadata.create_all(engine)
	yield engine
	SQLModel.metadata.drop_all(engine)
	engine.dispose()



TEST_MODELS_CONFIG = Path(__file__).with_name("models.test.json")


@pytest.fixture(autouse=True)
def stub_generation_agents() -> Generator[None, None, None]:
    """Point the ensemble at PydanticAI's TestModel for the whole suite.

    Keeps the tests offline and free: no API keys, no Ollama, no token spend.
    Voting is disabled for the same reason - the voters need a live provider.
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

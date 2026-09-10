from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

import app.db as db
from app.db import get_session
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

import os
from collections.abc import Callable, Iterator

import psycopg
import pytest

from investigation_engine.storage.database import Database

DEFAULT_TEST_DATABASE_URL = "postgresql://investigation:investigation@127.0.0.1:5433/investigation"
CONNECT_TIMEOUT_SECONDS = 2


@pytest.fixture(scope="session")
def postgres_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)
    try:
        psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_SECONDS).close()
    except psycopg.OperationalError:
        pytest.skip("no test database; start one with: docker compose up -d db")
    return url


@pytest.fixture
def make_database(postgres_url: str) -> Iterator[Callable[[int], Database]]:
    created: list[Database] = []

    def build(dimension: int) -> Database:
        database = Database(postgres_url, dimension)
        database.drop_schema()
        database.ensure_schema()
        created.append(database)
        return database

    yield build
    for database in created:
        database.drop_schema()
        database.close()

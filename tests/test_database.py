from collections.abc import Callable

import pytest
from psycopg import Connection

from investigation_engine.storage.database import Database

EXPECTED_TABLES = {"documents", "chunks", "chunk_owners", "chunk_vectors"}


def table_names(connection: Connection) -> set[str]:
    rows = connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    return {name for (name,) in rows}


def test_ensure_schema_creates_every_table(make_database: Callable[[int], Database]) -> None:
    database = make_database(2)

    assert EXPECTED_TABLES.issubset(database.run(table_names))


def test_ensure_schema_can_run_twice(make_database: Callable[[int], Database]) -> None:
    database = make_database(2)

    database.ensure_schema()

    assert EXPECTED_TABLES.issubset(database.run(table_names))


def test_drop_schema_removes_every_table(make_database: Callable[[int], Database]) -> None:
    database = make_database(2)

    database.drop_schema()

    assert not EXPECTED_TABLES & database.run(table_names)


def test_run_returns_the_value_the_work_returns(make_database: Callable[[int], Database]) -> None:
    database = make_database(2)

    assert database.run(lambda connection: connection.execute("SELECT 41 + 1").fetchone()) == (42,)


def test_run_commits_the_work_when_it_succeeds(make_database: Callable[[int], Database]) -> None:
    database = make_database(2)

    database.run(
        lambda connection: connection.execute(
            "INSERT INTO documents (id, title, source, content_hash) VALUES ('a', 't', 's', 'h')"
        )
    )

    count = database.run(lambda connection: connection.execute("SELECT count(*) FROM documents"))
    assert count.fetchone() == (1,)


def test_run_rolls_back_the_work_when_it_raises(make_database: Callable[[int], Database]) -> None:
    database = make_database(2)

    def failing_work(connection: Connection) -> None:
        connection.execute(
            "INSERT INTO documents (id, title, source, content_hash) VALUES ('a', 't', 's', 'h')"
        )
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        database.run(failing_work)

    count = database.run(lambda connection: connection.execute("SELECT count(*) FROM documents"))
    assert count.fetchone() == (0,)


def test_run_does_not_retry_an_error_that_is_not_a_lost_connection(
    make_database: Callable[[int], Database],
) -> None:
    database = make_database(2)
    calls: list[int] = []

    def failing_work(connection: Connection) -> None:
        connection.execute("SELECT 1")
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        database.run(failing_work)

    assert len(calls) == 1


def test_run_reconnects_and_repeats_the_work_once_after_the_connection_was_lost(
    make_database: Callable[[int], Database],
) -> None:
    database = make_database(2)
    calls: list[int] = []

    def work(connection: Connection) -> tuple[int] | None:
        calls.append(1)
        if len(calls) == 1:
            connection.execute("SELECT pg_terminate_backend(pg_backend_pid())")
        return connection.execute("SELECT 41 + 1").fetchone()

    assert database.run(work) == (42,)
    assert len(calls) == 2


def test_run_gives_up_when_the_connection_is_lost_twice(
    make_database: Callable[[int], Database],
) -> None:
    database = make_database(2)
    calls: list[int] = []

    def work(connection: Connection) -> None:
        calls.append(1)
        connection.execute("SELECT pg_terminate_backend(pg_backend_pid())")
        connection.execute("SELECT 1")

    with pytest.raises(Exception, match="connection|terminat"):
        database.run(work)

    assert len(calls) == 2

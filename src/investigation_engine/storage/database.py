import threading
from collections.abc import Callable
from contextlib import suppress

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, sql

ATTEMPTS = 2
CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_EF_SEARCH = 100

STATIC_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        id text PRIMARY KEY,
        title text NOT NULL,
        source text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        content_hash text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS documents_created_at ON documents (created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id text PRIMARY KEY,
        document_id text NOT NULL,
        position integer NOT NULL,
        text text NOT NULL,
        title text NOT NULL,
        source text NOT NULL,
        fingerprint text NOT NULL UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunk_owners (
        seq bigint GENERATED ALWAYS AS IDENTITY,
        chunk_id text NOT NULL REFERENCES chunks (id) ON DELETE CASCADE ON UPDATE CASCADE,
        document_id text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        position integer NOT NULL,
        PRIMARY KEY (chunk_id, document_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS chunk_owners_document ON chunk_owners (document_id)",
)

VECTOR_TABLE = sql.SQL(
    "CREATE TABLE IF NOT EXISTS chunk_vectors "
    "(chunk_id text PRIMARY KEY, embedding halfvec({dimension}) NOT NULL)"
)
VECTOR_INDEX = (
    "CREATE INDEX IF NOT EXISTS chunk_vectors_embedding "
    "ON chunk_vectors USING hnsw (embedding halfvec_cosine_ops)"
)
DROP_TABLES = "DROP TABLE IF EXISTS chunk_vectors, chunk_owners, chunks, documents CASCADE"


class Database:
    def __init__(self, url: str, dimension: int, ef_search: int = DEFAULT_EF_SEARCH) -> None:
        if dimension <= 0 or ef_search <= 0:
            raise ValueError("dimension and ef_search must be positive")
        self._url = url
        self._dimension = dimension
        self._ef_search = ef_search
        self._lock = threading.RLock()
        self._connection: Connection | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def run[T](self, work: Callable[[Connection], T]) -> T:
        with self._lock:
            for _ in range(ATTEMPTS - 1):
                try:
                    return self._attempt(work)
                except psycopg.OperationalError:
                    self.close()
            return self._attempt(work)

    def ensure_schema(self) -> None:
        self.run(self._create_tables)

    def drop_schema(self) -> None:
        self.run(lambda connection: connection.execute(DROP_TABLES))

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                with suppress(psycopg.Error):
                    self._connection.close()
                self._connection = None

    def _create_tables(self, connection: Connection) -> None:
        for statement in STATIC_SCHEMA:
            connection.execute(statement)
        connection.execute(VECTOR_TABLE.format(dimension=sql.Literal(self._dimension)))
        connection.execute(VECTOR_INDEX)

    def _attempt[T](self, work: Callable[[Connection], T]) -> T:
        connection = self._open()
        try:
            result = work(connection)
            connection.commit()
        except Exception:
            with suppress(psycopg.Error):
                connection.rollback()
            raise
        return result

    def _open(self) -> Connection:
        existing = self._connection
        if existing is not None and not existing.closed and not existing.broken:
            return existing
        connection = psycopg.connect(
            self._url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_SECONDS
        )
        try:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(connection)
            connection.execute(
                sql.SQL("SET hnsw.ef_search = {value}").format(value=sql.Literal(self._ef_search))
            )
        except Exception:
            connection.close()
            raise
        connection.autocommit = False
        self._connection = connection
        return connection

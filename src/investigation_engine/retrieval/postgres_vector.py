from collections.abc import Sequence

import numpy as np
from psycopg import Connection

from investigation_engine.storage.database import Database

UPSERT_VECTOR = (
    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (%s, %s) "
    "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding"
)
NEAREST_NEIGHBOURS = (
    "SELECT chunk_id, score FROM ("
    "SELECT chunk_id, 1 - (embedding <=> %(vector)s) AS score FROM chunk_vectors "
    "ORDER BY embedding <=> %(vector)s LIMIT %(limit)s) nearest "
    "ORDER BY score DESC, chunk_id"
)


class PostgresVectorIndex:
    def __init__(self, database: Database) -> None:
        self._database = database

    def __len__(self) -> int:
        row = self._database.run(
            lambda connection: connection.execute("SELECT count(*) FROM chunk_vectors").fetchone()
        )
        return row[0] if row else 0

    def add(self, item_id: str, vector: Sequence[float]) -> None:
        self.add_many([(item_id, vector)])

    def add_many(self, items: Sequence[tuple[str, Sequence[float]]]) -> None:
        rows = [(item_id, self._checked(vector)) for item_id, vector in items]
        if not rows:
            return

        def write(connection: Connection) -> None:
            with connection.cursor() as cursor:
                cursor.executemany(UPSERT_VECTOR, rows)

        self._database.run(write)

    def remove(self, item_id: str) -> None:
        self._database.run(
            lambda connection: connection.execute(
                "DELETE FROM chunk_vectors WHERE chunk_id = %s", (item_id,)
            )
        )

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[str, float]]:
        query = self._checked(vector, allow_zero=True)
        if limit <= 0 or not query.any():
            return []
        rows = self._database.run(
            lambda connection: connection.execute(
                NEAREST_NEIGHBOURS, {"vector": query, "limit": limit}
            ).fetchall()
        )
        return [(chunk_id, float(score)) for chunk_id, score in rows]

    def _checked(self, vector: Sequence[float], allow_zero: bool = False) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self._database.dimension,):
            raise ValueError(f"expected a vector with dimension {self._database.dimension}")
        if not allow_zero and not array.any():
            raise ValueError("cannot index a zero vector")
        return array

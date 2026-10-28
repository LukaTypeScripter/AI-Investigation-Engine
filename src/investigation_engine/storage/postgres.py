from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass

from psycopg import Connection
from psycopg.types.json import Jsonb

from investigation_engine.domain import Chunk, Document, SourceRef
from investigation_engine.ingestion.dedup import fingerprint
from investigation_engine.ingestion.service import Deletion
from investigation_engine.storage.database import Database

DEFAULT_PAGE_SIZE = 50
DEFAULT_STREAM_BATCH = 1000

SELECT_CHUNKS_BY_ID = (
    "SELECT id, document_id, position, text, title, source FROM chunks WHERE id = ANY(%s)"
)
SELECT_CHUNK_PAGE = (
    "SELECT id, document_id, position, text, title, source FROM chunks "
    "WHERE id > %s ORDER BY id LIMIT %s"
)
LIST_DOCUMENTS = (
    "SELECT d.id, d.title, d.source, "
    "(SELECT count(*) FROM chunk_owners o WHERE o.document_id = d.id) "
    "FROM documents d WHERE (%(text)s = '' OR strpos(lower(d.title), lower(%(text)s)) > 0 "
    "OR strpos(lower(d.source), lower(%(text)s)) > 0) "
    "ORDER BY d.created_at DESC, d.id LIMIT %(limit)s OFFSET %(offset)s"
)
COUNT_DOCUMENTS = (
    "SELECT count(*) FROM documents d WHERE (%(text)s = '' "
    "OR strpos(lower(d.title), lower(%(text)s)) > 0 "
    "OR strpos(lower(d.source), lower(%(text)s)) > 0)"
)


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    id: str
    title: str
    source: str
    chunk_count: int


def to_chunk(row: tuple[str, str, int, str, str, str]) -> Chunk:
    chunk_id, document_id, index, text, title, source = row
    return Chunk(
        id=chunk_id, document_id=document_id, index=index, text=text, title=title, source=source
    )


class PostgresDocumentStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    def content_hash(self, document_id: str) -> str | None:
        row = self._database.run(
            lambda connection: connection.execute(
                "SELECT content_hash FROM documents WHERE id = %s", (document_id,)
            ).fetchone()
        )
        return row[0] if row else None

    def chunk_ids_by_fingerprint(self, fingerprints: Collection[str]) -> dict[str, str]:
        if not fingerprints:
            return {}
        rows = self._database.run(
            lambda connection: connection.execute(
                "SELECT fingerprint, id FROM chunks WHERE fingerprint = ANY(%s)",
                (list(fingerprints),),
            ).fetchall()
        )
        return dict(rows)

    def save(
        self,
        document: Document,
        content_hash: str,
        new_chunks: Sequence[Chunk],
        shared: Mapping[str, int],
    ) -> None:
        self._database.run(
            lambda connection: self._save(connection, document, content_hash, new_chunks, shared)
        )

    def delete(self, document_id: str) -> Deletion | None:
        return self._database.run(lambda connection: self._delete(connection, document_id))

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        rows = self._database.run(
            lambda connection: connection.execute(
                SELECT_CHUNKS_BY_ID, (list(chunk_ids),)
            ).fetchall()
        )
        found = {row[0]: to_chunk(row) for row in rows}
        return {chunk_id: found[chunk_id] for chunk_id in chunk_ids if chunk_id in found}

    def sources_of(self, chunk_id: str) -> list[SourceRef]:
        rows = self._database.run(
            lambda connection: connection.execute(
                "SELECT d.id, d.title, d.source FROM chunk_owners o "
                "JOIN documents d ON d.id = o.document_id WHERE o.chunk_id = %s ORDER BY o.seq",
                (chunk_id,),
            ).fetchall()
        )
        return [SourceRef(*row) for row in rows]

    def list_documents(
        self, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, containing: str = ""
    ) -> list[DocumentSummary]:
        rows = self._database.run(
            lambda connection: connection.execute(
                LIST_DOCUMENTS, {"text": containing, "limit": limit, "offset": offset}
            ).fetchall()
        )
        return [DocumentSummary(*row) for row in rows]

    def count_documents(self, containing: str = "") -> int:
        row = self._database.run(
            lambda connection: connection.execute(COUNT_DOCUMENTS, {"text": containing}).fetchone()
        )
        return row[0] if row else 0

    def count_chunks(self) -> int:
        row = self._database.run(
            lambda connection: connection.execute("SELECT count(*) FROM chunks").fetchone()
        )
        return row[0] if row else 0

    def iter_chunks(self, batch_size: int = DEFAULT_STREAM_BATCH) -> Iterator[Chunk]:
        last_id = ""
        while True:
            batch = self._chunk_page(last_id, batch_size)
            yield from batch
            if len(batch) < batch_size:
                return
            last_id = batch[-1].id

    def chunk_ids_without_vector(self) -> list[str]:
        rows = self._database.run(
            lambda connection: connection.execute(
                "SELECT c.id FROM chunks c LEFT JOIN chunk_vectors v ON v.chunk_id = c.id "
                "WHERE v.chunk_id IS NULL ORDER BY c.id"
            ).fetchall()
        )
        return [row[0] for row in rows]

    def document_ids(self) -> set[str]:
        rows = self._database.run(
            lambda connection: connection.execute("SELECT id FROM documents").fetchall()
        )
        return {row[0] for row in rows}

    def chunk_ids(self) -> set[str]:
        rows = self._database.run(
            lambda connection: connection.execute("SELECT id FROM chunks").fetchall()
        )
        return {row[0] for row in rows}

    def chunk(self, chunk_id: str) -> Chunk:
        found = self.get_chunks([chunk_id])
        if chunk_id not in found:
            raise KeyError(chunk_id)
        return found[chunk_id]

    def owners_of(self, chunk_id: str) -> dict[str, int]:
        rows = self._database.run(
            lambda connection: connection.execute(
                "SELECT document_id, position FROM chunk_owners WHERE chunk_id = %s ORDER BY seq",
                (chunk_id,),
            ).fetchall()
        )
        return dict(rows)

    def _chunk_page(self, after_id: str, size: int) -> list[Chunk]:
        rows = self._database.run(
            lambda connection: connection.execute(SELECT_CHUNK_PAGE, (after_id, size)).fetchall()
        )
        return [to_chunk(row) for row in rows]

    @staticmethod
    def _save(
        connection: Connection,
        document: Document,
        content_hash: str,
        new_chunks: Sequence[Chunk],
        shared: Mapping[str, int],
    ) -> None:
        inserted = connection.execute(
            "INSERT INTO documents (id, title, source, metadata, content_hash) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING RETURNING id",
            (document.id, document.title, document.source, Jsonb(document.metadata), content_hash),
        ).fetchone()
        if inserted is None:
            raise ValueError(f"document {document.id} is already stored")
        owners = [(chunk.id, document.id, chunk.index) for chunk in new_chunks]
        owners.extend((chunk_id, document.id, index) for chunk_id, index in shared.items())
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (id, document_id, position, text, title, source, fingerprint) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.index,
                        chunk.text,
                        chunk.title,
                        chunk.source,
                        fingerprint(chunk.text),
                    )
                    for chunk in new_chunks
                ],
            )
            cursor.executemany(
                "INSERT INTO chunk_owners (chunk_id, document_id, position) VALUES (%s, %s, %s)",
                owners,
            )

    @classmethod
    def _delete(cls, connection: Connection, document_id: str) -> Deletion | None:
        owned = [
            row[0]
            for row in connection.execute(
                "SELECT chunk_id FROM chunk_owners WHERE document_id = %s", (document_id,)
            ).fetchall()
        ]
        deleted = connection.execute(
            "DELETE FROM documents WHERE id = %s RETURNING id", (document_id,)
        ).fetchone()
        if deleted is None:
            return None
        orphans = [
            row[0]
            for row in connection.execute(
                "DELETE FROM chunks WHERE id = ANY(%s) "
                "AND NOT EXISTS (SELECT 1 FROM chunk_owners WHERE chunk_id = chunks.id) "
                "RETURNING id",
                (owned,),
            ).fetchall()
        ]
        homeless = connection.execute(
            "SELECT id FROM chunks WHERE id = ANY(%s) AND document_id = %s ORDER BY id",
            (owned, document_id),
        ).fetchall()
        rehomed = [cls._rehome(connection, row[0]) for row in homeless]
        removed = sorted([*orphans, *(row[0] for row in homeless)])
        return Deletion(removed_chunk_ids=removed, rehomed_chunks=rehomed)

    @staticmethod
    def _rehome(connection: Connection, chunk_id: str) -> Chunk:
        owner = connection.execute(
            "SELECT o.document_id, o.position, d.title, d.source FROM chunk_owners o "
            "JOIN documents d ON d.id = o.document_id WHERE o.chunk_id = %s "
            "ORDER BY o.seq LIMIT 1",
            (chunk_id,),
        ).fetchone()
        if owner is None:
            raise LookupError(f"chunk {chunk_id} has no owner to move to")
        owner_id, index, title, source = owner
        moved_id = f"{owner_id}:{index}"
        row = connection.execute(
            "UPDATE chunks SET id = %s, document_id = %s, position = %s, title = %s, source = %s "
            "WHERE id = %s RETURNING text",
            (moved_id, owner_id, index, title, source, chunk_id),
        ).fetchone()
        if row is None:
            raise LookupError(f"chunk {chunk_id} disappeared while it was being moved")
        return Chunk(
            id=moved_id, document_id=owner_id, index=index, text=row[0], title=title, source=source
        )

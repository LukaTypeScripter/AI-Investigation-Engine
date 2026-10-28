import hashlib
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from investigation_engine.domain import Chunk, Document
from investigation_engine.ingestion.chunking import Chunker
from investigation_engine.ingestion.dedup import DedupResult, fingerprint, split_duplicates


@dataclass(frozen=True, slots=True)
class Deletion:
    removed_chunk_ids: list[str]
    rehomed_chunks: list[Chunk]


class DocumentStore(Protocol):
    def content_hash(self, document_id: str) -> str | None: ...

    def chunk_ids_by_fingerprint(self, fingerprints: Collection[str]) -> dict[str, str]: ...

    def save(
        self,
        document: Document,
        content_hash: str,
        new_chunks: Sequence[Chunk],
        shared: Mapping[str, int],
    ) -> None: ...

    def delete(self, document_id: str) -> Deletion | None: ...


@dataclass(frozen=True, slots=True)
class IndexChanges:
    chunks_added: list[Chunk]
    chunks_removed: list[str]


@dataclass(frozen=True, slots=True)
class IngestReport:
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    duplicate_chunks: int
    changes: IndexChanges


class ChangeLog:
    def __init__(self) -> None:
        self._added: dict[str, Chunk] = {}
        self._removed: set[str] = set()

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self._added[chunk.id] = chunk

    def remove(self, chunk_ids: Iterable[str]) -> None:
        for chunk_id in chunk_ids:
            self._added.pop(chunk_id, None)
            self._removed.add(chunk_id)

    def record_deletion(self, deletion: Deletion) -> None:
        self.remove(deletion.removed_chunk_ids)
        self.add(deletion.rehomed_chunks)

    def build(self) -> IndexChanges:
        return IndexChanges(
            chunks_added=list(self._added.values()),
            chunks_removed=sorted(self._removed - self._added.keys()),
        )


def document_hash(document: Document) -> str:
    payload = json.dumps([document.title, document.text, document.metadata], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def shared_placements(chunks: Sequence[Chunk], result: DedupResult) -> dict[str, int]:
    new_chunk_ids = {chunk.id for chunk in result.unique}
    shared: dict[str, int] = {}
    for chunk in chunks:
        canonical_id = result.duplicate_of.get(chunk.id)
        if canonical_id is not None and canonical_id not in new_chunk_ids:
            shared.setdefault(canonical_id, chunk.index)
    return shared


class IngestionService:
    def __init__(self, store: DocumentStore, chunker: Chunker) -> None:
        self._store = store
        self._chunker = chunker

    def ingest(self, documents: Iterable[Document]) -> IngestReport:
        log = ChangeLog()
        added: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        duplicate_chunks = 0
        for document in documents:
            digest = document_hash(document)
            previous = self._store.content_hash(document.id)
            if previous == digest:
                unchanged.append(document.id)
                continue
            if previous is None:
                added.append(document.id)
            else:
                updated.append(document.id)
                self._remove(document.id, log)
            duplicate_chunks += self._store_document(document, digest, log)
        return IngestReport(
            added=added,
            updated=updated,
            unchanged=unchanged,
            duplicate_chunks=duplicate_chunks,
            changes=log.build(),
        )

    def delete(self, document_id: str) -> IndexChanges | None:
        log = ChangeLog()
        if not self._remove(document_id, log):
            return None
        return log.build()

    def _remove(self, document_id: str, log: ChangeLog) -> bool:
        deletion = self._store.delete(document_id)
        if deletion is None:
            return False
        log.record_deletion(deletion)
        return True

    def _store_document(self, document: Document, digest: str, log: ChangeLog) -> int:
        chunks = self._chunker.chunk(document)
        known = self._store.chunk_ids_by_fingerprint({fingerprint(chunk.text) for chunk in chunks})
        result = split_duplicates(chunks, known)
        self._store.save(document, digest, result.unique, shared_placements(chunks, result))
        log.add(result.unique)
        return len(result.duplicate_of)

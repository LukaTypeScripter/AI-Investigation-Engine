from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace

from investigation_engine.domain import Chunk, Document
from investigation_engine.ingestion.dedup import fingerprint
from investigation_engine.ingestion.service import Deletion


@dataclass(slots=True)
class StoredDocument:
    document: Document
    content_hash: str
    chunk_ids: set[str]


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._documents: dict[str, StoredDocument] = {}
        self._chunks: dict[str, Chunk] = {}
        self._chunk_ids_by_fingerprint: dict[str, str] = {}
        self._owners: dict[str, dict[str, int]] = {}

    def content_hash(self, document_id: str) -> str | None:
        stored = self._documents.get(document_id)
        return stored.content_hash if stored else None

    def chunk_ids_by_fingerprint(self, fingerprints: Collection[str]) -> dict[str, str]:
        known = self._chunk_ids_by_fingerprint
        return {key: known[key] for key in fingerprints if key in known}

    def save(
        self,
        document: Document,
        content_hash: str,
        new_chunks: Sequence[Chunk],
        shared: Mapping[str, int],
    ) -> None:
        if document.id in self._documents:
            raise ValueError(f"document {document.id} is already stored")
        chunk_ids = set(shared)
        for chunk in new_chunks:
            self._chunks[chunk.id] = chunk
            self._chunk_ids_by_fingerprint[fingerprint(chunk.text)] = chunk.id
            self._owners[chunk.id] = {document.id: chunk.index}
            chunk_ids.add(chunk.id)
        for chunk_id, index in shared.items():
            self._owners[chunk_id][document.id] = index
        self._documents[document.id] = StoredDocument(document, content_hash, chunk_ids)

    def delete(self, document_id: str) -> Deletion | None:
        stored = self._documents.pop(document_id, None)
        if stored is None:
            return None
        removed: list[str] = []
        rehomed: list[Chunk] = []
        for chunk_id in sorted(stored.chunk_ids):
            owners = self._owners[chunk_id]
            del owners[document_id]
            if not owners:
                self._drop_chunk(chunk_id)
                removed.append(chunk_id)
            elif self._chunks[chunk_id].document_id == document_id:
                removed.append(chunk_id)
                rehomed.append(self._rehome(chunk_id))
        return Deletion(removed_chunk_ids=removed, rehomed_chunks=rehomed)

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        return {
            chunk_id: self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks
        }

    def document_ids(self) -> set[str]:
        return set(self._documents)

    def chunk_ids(self) -> set[str]:
        return set(self._chunks)

    def chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def owners_of(self, chunk_id: str) -> dict[str, int]:
        return dict(self._owners[chunk_id])

    def _drop_chunk(self, chunk_id: str) -> None:
        chunk = self._chunks.pop(chunk_id)
        del self._chunk_ids_by_fingerprint[fingerprint(chunk.text)]
        del self._owners[chunk_id]

    def _rehome(self, chunk_id: str) -> Chunk:
        chunk = self._chunks.pop(chunk_id)
        owners = self._owners.pop(chunk_id)
        owner_id, index = next(iter(owners.items()))
        owner = self._documents[owner_id].document
        moved = replace(
            chunk,
            id=f"{owner_id}:{index}",
            document_id=owner_id,
            index=index,
            title=owner.title,
            source=owner.source,
        )
        self._chunks[moved.id] = moved
        self._owners[moved.id] = owners
        self._chunk_ids_by_fingerprint[fingerprint(moved.text)] = moved.id
        for owner_document_id in owners:
            owned_chunk_ids = self._documents[owner_document_id].chunk_ids
            owned_chunk_ids.discard(chunk_id)
            owned_chunk_ids.add(moved.id)
        return moved

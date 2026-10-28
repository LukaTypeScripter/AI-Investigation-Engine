from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from investigation_engine.domain import Chunk

DEFAULT_PAGE_SIZE = 50
DEFAULT_STREAM_BATCH = 1000


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    id: str
    title: str
    source: str
    chunk_count: int


class DocumentCatalog(Protocol):
    def list_documents(
        self, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0, containing: str = ""
    ) -> list[DocumentSummary]: ...

    def count_documents(self, containing: str = "") -> int: ...

    def count_chunks(self) -> int: ...

    def origins_of(self, document_ids: Sequence[str]) -> dict[str, str]: ...

    def iter_chunks(self, batch_size: int = DEFAULT_STREAM_BATCH) -> Iterator[Chunk]: ...

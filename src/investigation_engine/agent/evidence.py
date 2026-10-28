from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from investigation_engine.agent.models import Citation
from investigation_engine.domain import Chunk, SourceRef
from investigation_engine.retrieval.hybrid import Hit

ID_PUNCTUATION = "[]() ,.;:"
DEFAULT_MAX_CHARS = 1200
TRIMMED_SUFFIX = "..."


class SourceLookup(Protocol):
    def sources_of(self, chunk_id: str) -> list[SourceRef]: ...


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    chunk: Chunk
    sources: tuple[SourceRef, ...]


def normalize_evidence_id(raw: str) -> str:
    return raw.strip().strip(ID_PUNCTUATION).upper()


class EvidenceBoard:
    def __init__(self, sources: SourceLookup, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self._sources = sources
        self._max_chars = max_chars
        self._items: dict[str, EvidenceItem] = {}
        self._items_by_chunk: dict[str, EvidenceItem] = {}

    def add(self, hits: Sequence[Hit]) -> list[EvidenceItem]:
        return [self._item_for(hit.chunk) for hit in hits]

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._items.get(normalize_evidence_id(evidence_id))

    def unknown(self, evidence_ids: Iterable[str]) -> list[str]:
        normalized = (normalize_evidence_id(evidence_id) for evidence_id in evidence_ids)
        return [evidence_id for evidence_id in normalized if evidence_id not in self._items]

    def citation(self, evidence_id: str) -> Citation:
        item = self._items[normalize_evidence_id(evidence_id)]
        return Citation(
            evidence_id=item.evidence_id,
            chunk_id=item.chunk.id,
            title=item.chunk.title,
            text=item.chunk.text,
            sources=item.sources,
        )

    def render(self, query: str, items: Sequence[EvidenceItem]) -> str:
        if not items:
            return f"No results found for: {query}"
        blocks = [self._render_item(item) for item in items]
        return "\n\n".join([f"Search results for: {query}", *blocks])

    def _item_for(self, chunk: Chunk) -> EvidenceItem:
        existing = self._items_by_chunk.get(chunk.id)
        if existing is not None:
            return existing
        sources = tuple(self._sources.sources_of(chunk.id)) or (
            SourceRef(chunk.document_id, chunk.title, chunk.source),
        )
        item = EvidenceItem(f"E{len(self._items) + 1}", chunk, sources)
        self._items[item.evidence_id] = item
        self._items_by_chunk[chunk.id] = item
        return item

    def _render_item(self, item: EvidenceItem) -> str:
        title = item.chunk.title or "untitled"
        sources = ", ".join(ref.source or ref.document_id for ref in item.sources)
        return f'[{item.evidence_id}] "{title}" (sources: {sources})\n{self._trim(item.chunk.text)}'

    def _trim(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars].rstrip() + TRIMMED_SUFFIX

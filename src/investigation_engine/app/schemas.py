from pydantic import BaseModel, Field

from investigation_engine.agent.models import Citation, Conflict, Investigation
from investigation_engine.app.engine import DocumentPage, UploadResult
from investigation_engine.domain import SourceRef
from investigation_engine.retrieval.hybrid import Hit

MAX_SEARCH_RESULTS = 20
DEFAULT_SEARCH_RESULTS = 5


class SourceOut(BaseModel):
    document_id: str
    title: str
    source: str


class CitationOut(BaseModel):
    evidence_id: str
    chunk_id: str
    title: str
    text: str
    sources: list[SourceOut]


class ConflictOut(BaseModel):
    description: str
    evidence_ids: list[str]


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class AnswerOut(BaseModel):
    question: str
    status: str
    answer: str
    citations: list[CitationOut]
    conflicts: list[ConflictOut]
    searches: list[str]
    usage: UsageOut
    llm_calls: int
    abstain_reason: str | None
    elapsed_ms: int


class HitOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    sources: list[SourceOut]


class SearchOut(BaseModel):
    hits: list[HitOut]


class DocumentOut(BaseModel):
    id: str
    title: str
    source: str
    chunk_count: int


class DocumentPageOut(BaseModel):
    items: list[DocumentOut]
    total: int


class UploadOut(BaseModel):
    filename: str
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    duplicate_chunks: int
    chunks_added: int
    chunks_removed: int


class HealthOut(BaseModel):
    status: str
    documents: int
    chunks: int
    can_answer: bool


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=DEFAULT_SEARCH_RESULTS, ge=1, le=MAX_SEARCH_RESULTS)


class AskIn(BaseModel):
    question: str


def source_out(source: SourceRef) -> SourceOut:
    return SourceOut(document_id=source.document_id, title=source.title, source=source.source)


def citation_out(citation: Citation) -> CitationOut:
    return CitationOut(
        evidence_id=citation.evidence_id,
        chunk_id=citation.chunk_id,
        title=citation.title,
        text=citation.text,
        sources=[source_out(source) for source in citation.sources],
    )


def conflict_out(conflict: Conflict) -> ConflictOut:
    return ConflictOut(description=conflict.description, evidence_ids=list(conflict.evidence_ids))


def answer_out(investigation: Investigation, elapsed_ms: int) -> AnswerOut:
    return AnswerOut(
        question=investigation.question,
        status=investigation.status,
        answer=investigation.answer,
        citations=[citation_out(citation) for citation in investigation.citations],
        conflicts=[conflict_out(conflict) for conflict in investigation.conflicts],
        searches=list(investigation.searches),
        usage=UsageOut(
            prompt_tokens=investigation.usage.prompt_tokens,
            completion_tokens=investigation.usage.completion_tokens,
        ),
        llm_calls=investigation.llm_calls,
        abstain_reason=investigation.abstain_reason,
        elapsed_ms=elapsed_ms,
    )


def hit_out(hit: Hit, sources: list[SourceRef]) -> HitOut:
    return HitOut(
        chunk_id=hit.chunk.id,
        document_id=hit.chunk.document_id,
        title=hit.chunk.title,
        source=hit.chunk.source,
        text=hit.chunk.text,
        score=hit.score,
        vector_rank=hit.vector_rank,
        keyword_rank=hit.keyword_rank,
        sources=[source_out(source) for source in sources],
    )


def page_out(page: DocumentPage) -> DocumentPageOut:
    return DocumentPageOut(
        items=[
            DocumentOut(
                id=item.id, title=item.title, source=item.source, chunk_count=item.chunk_count
            )
            for item in page.items
        ],
        total=page.total,
    )


def upload_out(result: UploadResult) -> UploadOut:
    return UploadOut(
        filename=result.filename,
        added=result.added,
        updated=result.updated,
        unchanged=result.unchanged,
        duplicate_chunks=result.duplicate_chunks,
        chunks_added=result.chunks_added,
        chunks_removed=result.chunks_removed,
    )

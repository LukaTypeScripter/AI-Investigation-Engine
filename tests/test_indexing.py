from collections.abc import Sequence

from fakes import HashingEmbedder
from investigation_engine.domain import Chunk, Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IndexChanges, IngestionService
from investigation_engine.retrieval.bm25 import Bm25Index
from investigation_engine.retrieval.indexing import SearchIndexer, passage_text
from investigation_engine.retrieval.vector import InMemoryVectorIndex
from investigation_engine.storage.memory import InMemoryDocumentStore


def make_chunk(chunk_id: str, text: str, title: str = "") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=chunk_id.split(":")[0],
        index=0,
        text=text,
        title=title,
        source="",
    )


def make_indexer(
    batch_size: int = 64,
) -> tuple[SearchIndexer, Bm25Index, InMemoryVectorIndex, HashingEmbedder]:
    embedder = HashingEmbedder()
    keywords = Bm25Index()
    vectors = InMemoryVectorIndex(dimension=embedder.dimension)
    return SearchIndexer(embedder, vectors, keywords, batch_size), keywords, vectors, embedder


def added(*chunks: Chunk) -> IndexChanges:
    return IndexChanges(chunks_added=list(chunks), chunks_removed=[])


def test_passage_text_puts_the_title_before_the_text() -> None:
    assert passage_text(make_chunk("a:0", "Body text.", "Title")) == "Title\nBody text."


def test_passage_text_without_a_title_is_just_the_text() -> None:
    assert passage_text(make_chunk("a:0", "Body text.")) == "Body text."


def test_apply_makes_added_chunks_searchable_by_keyword_and_by_vector() -> None:
    indexer, keywords, vectors, embedder = make_indexer()

    indexer.apply(added(make_chunk("a:0", "payments failed in march")))

    assert [item_id for item_id, _ in keywords.search("payments", 5)] == ["a:0"]
    query = embedder.embed_query("payments failed")
    assert [item_id for item_id, _ in vectors.search(query, 5)] == ["a:0"]


def test_apply_removes_chunks_from_both_indexes() -> None:
    indexer, keywords, vectors, _ = make_indexer()
    indexer.apply(added(make_chunk("a:0", "payments failed")))

    indexer.apply(IndexChanges(chunks_added=[], chunks_removed=["a:0"]))

    assert len(keywords) == 0
    assert len(vectors) == 0


def test_apply_replaces_a_chunk_that_is_removed_and_added_in_the_same_change() -> None:
    indexer, keywords, _, _ = make_indexer()
    indexer.apply(added(make_chunk("a:0", "old words")))

    indexer.apply(IndexChanges([make_chunk("a:0", "new words")], ["a:0"]))

    assert keywords.search("old", 5) == []
    assert [item_id for item_id, _ in keywords.search("new", 5)] == ["a:0"]


def test_apply_embeds_chunks_in_batches() -> None:
    indexer, _, _, embedder = make_indexer(batch_size=2)

    indexer.apply(added(*(make_chunk(f"a:{n}", f"text number {n}") for n in range(5))))

    assert [len(batch) for batch in embedder.document_batches] == [2, 2, 1]


class BatchRecordingVectorIndex(InMemoryVectorIndex):
    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.batch_sizes: list[int] = []

    def add_many(self, items: Sequence[tuple[str, Sequence[float]]]) -> None:
        self.batch_sizes.append(len(items))
        super().add_many(items)


def test_apply_writes_each_embedding_batch_to_the_vector_index_in_one_call() -> None:
    embedder = HashingEmbedder()
    vectors = BatchRecordingVectorIndex(embedder.dimension)
    indexer = SearchIndexer(embedder, vectors, Bm25Index(), batch_size=2)

    indexer.apply(added(*(make_chunk(f"a:{n}", f"text number {n}") for n in range(5))))

    assert vectors.batch_sizes == [2, 2, 1]
    assert len(vectors) == 5


def test_apply_without_changes_does_not_call_the_embedder() -> None:
    indexer, _, _, embedder = make_indexer()

    indexer.apply(IndexChanges(chunks_added=[], chunks_removed=[]))

    assert embedder.document_batches == []


def test_the_title_is_searchable_by_keyword() -> None:
    indexer, keywords, _, _ = make_indexer()

    indexer.apply(added(make_chunk("a:0", "American director.", "Scott Derrickson")))

    assert [item_id for item_id, _ in keywords.search("Derrickson", 5)] == ["a:0"]


def test_index_adds_chunks_without_a_change_report() -> None:
    indexer, keywords, vectors, _ = make_indexer()

    indexer.index([make_chunk("a:0", "one"), make_chunk("b:0", "two")])

    assert (len(keywords), len(vectors)) == (2, 2)


def test_indexes_follow_ingestion_updates_and_deletions() -> None:
    indexer, keywords, vectors, _ = make_indexer()
    service = IngestionService(InMemoryDocumentStore(), SentenceChunker(max_words=20))
    document = Document(id="a", text="Payments failed in March.", title="Outage", source="a.txt")

    indexer.apply(service.ingest([document]).changes)
    assert [item_id for item_id, _ in keywords.search("payments", 5)] == ["a:0"]

    changed = Document(id="a", text="Refunds were late.", title="Outage", source="a.txt")
    indexer.apply(service.ingest([changed]).changes)
    assert keywords.search("payments", 5) == []
    assert [item_id for item_id, _ in keywords.search("refunds", 5)] == ["a:0"]

    deletion = service.delete("a")
    assert deletion is not None
    indexer.apply(deletion)
    assert (len(keywords), len(vectors)) == (0, 0)

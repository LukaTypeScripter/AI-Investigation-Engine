from collections.abc import Callable
from pathlib import Path

import pytest

from fakes import HashingEmbedder
from investigation_engine.app.main import build_engine, create_default_app
from investigation_engine.domain import Document
from investigation_engine.ingestion.chunking import SentenceChunker
from investigation_engine.ingestion.service import IngestionService
from investigation_engine.llm.factory import ConfigurationError
from investigation_engine.retrieval.postgres_vector import PostgresVectorIndex
from investigation_engine.settings import Settings
from investigation_engine.storage.database import Database
from investigation_engine.storage.postgres import PostgresDocumentStore

DIMENSION = HashingEmbedder().dimension
OUTAGE = b"Payments failed on 5 March because of a bad deploy."


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("GEMINI_API_KEY", "GROQ_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(variable, raising=False)


def make_settings(url: str, tmp_path: Path, gemini_api_key: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        database_url=url,
        embedding_dimension=DIMENSION,
        llm_cache_path=tmp_path / "cache.sqlite",
        gemini_api_key=gemini_api_key,
    )


def test_building_the_engine_without_a_database_url_explains_what_is_missing(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, llm_cache_path=tmp_path / "cache.sqlite")

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        build_engine(settings, HashingEmbedder())


def test_the_built_engine_creates_the_schema_and_serves_uploads(
    make_database: Callable[[int], Database], postgres_url: str, tmp_path: Path
) -> None:
    make_database(DIMENSION).drop_schema()

    engine = build_engine(make_settings(postgres_url, tmp_path), HashingEmbedder())
    engine.upload("outage.txt", OUTAGE)

    assert [hit.chunk.document_id for hit in engine.search("bad deploy", 3)] == ["outage.txt"]


def test_documents_stored_before_a_restart_are_searchable_by_keyword_and_vector(
    make_database: Callable[[int], Database], postgres_url: str, tmp_path: Path
) -> None:
    make_database(DIMENSION)
    settings = make_settings(postgres_url, tmp_path)
    build_engine(settings, HashingEmbedder()).upload("outage.txt", OUTAGE)

    restarted = build_engine(settings, HashingEmbedder())

    hit = restarted.search("bad deploy", 3)[0]
    assert hit.chunk.document_id == "outage.txt"
    assert (hit.keyword_rank, hit.vector_rank) == (1, 1)


def test_chunks_stored_without_a_vector_are_embedded_when_the_engine_starts(
    make_database: Callable[[int], Database], postgres_url: str, tmp_path: Path
) -> None:
    database = make_database(DIMENSION)
    IngestionService(PostgresDocumentStore(database), SentenceChunker(max_words=30)).ingest(
        [Document(id="a", text="Alpha gateway outage.", title="a", source="a.txt")]
    )

    build_engine(make_settings(postgres_url, tmp_path), HashingEmbedder())

    assert len(PostgresVectorIndex(database)) == 1


def test_the_engine_answers_only_when_a_language_model_key_is_configured(
    make_database: Callable[[int], Database], postgres_url: str, tmp_path: Path
) -> None:
    make_database(DIMENSION)

    without_key = build_engine(make_settings(postgres_url, tmp_path), HashingEmbedder())
    with_key = build_engine(
        make_settings(postgres_url, tmp_path, gemini_api_key="test-key"), HashingEmbedder()
    )

    assert (without_key.can_answer, with_key.can_answer) == (False, True)


def test_the_default_app_is_built_without_touching_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/none")

    app = create_default_app()

    assert app.title == "Investigation Engine"

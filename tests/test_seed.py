import json
from collections.abc import Callable
from pathlib import Path

import pytest

from fakes import HashingEmbedder
from investigation_engine.app.seed import main
from investigation_engine.evaluation.questions import EvalQuestion, write_questions
from investigation_engine.retrieval.postgres_vector import PostgresVectorIndex
from investigation_engine.settings import Settings
from investigation_engine.storage.database import Database
from investigation_engine.storage.postgres import PostgresDocumentStore

DIMENSION = HashingEmbedder().dimension


def hotpot_row(row_id: str, titles: tuple[str, ...], supporting: tuple[str, ...]) -> str:
    row = {
        "id": row_id,
        "question": f"Question {row_id}?",
        "answer": "yes",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": list(supporting), "sent_id": [0] * len(supporting)},
        "context": {
            "title": list(titles),
            "sentences": [[f"{title} is a thing.", " It has facts."] for title in titles],
        },
    }
    return json.dumps(row)


@pytest.fixture
def settings(postgres_url: str, tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=postgres_url,
        embedding_dimension=DIMENSION,
        llm_cache_path=tmp_path / "cache.sqlite",
    )


@pytest.fixture
def samples(tmp_path: Path) -> Path:
    directory = tmp_path / "samples"
    directory.mkdir()
    (directory / "outage.txt").write_text("Payments failed after a bad deploy.", encoding="utf-8")
    (directory / "refunds.md").write_text("# Refunds\n\nRefunds were late.", encoding="utf-8")
    return directory


def stored(settings: Settings) -> tuple[PostgresDocumentStore, PostgresVectorIndex]:
    assert settings.database_url is not None
    database = Database(settings.database_url.get_secret_value(), DIMENSION)
    return PostgresDocumentStore(database), PostgresVectorIndex(database)


def test_seeding_the_sample_files_stores_every_document_and_embeds_every_chunk(
    make_database: Callable[[int], Database],
    settings: Settings,
    samples: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_database(DIMENSION)

    code = main(["--samples", str(samples)], HashingEmbedder(), settings)

    store, vectors = stored(settings)
    assert code == 0
    assert (store.count_documents(), store.count_chunks(), len(vectors)) == (2, 2, 2)
    assert "2 documents" in capsys.readouterr().out


def test_seeding_the_same_files_again_changes_nothing_and_embeds_nothing(
    make_database: Callable[[int], Database], settings: Settings, samples: Path
) -> None:
    make_database(DIMENSION)
    main(["--samples", str(samples)], HashingEmbedder(), settings)
    second_embedder = HashingEmbedder()

    code = main(["--samples", str(samples)], second_embedder, settings)

    store, vectors = stored(settings)
    assert code == 0
    assert (store.count_documents(), len(vectors)) == (2, 2)
    assert second_embedder.document_batches == []


def test_seeding_reads_hotpot_paragraphs_as_one_document_per_title(
    make_database: Callable[[int], Database], settings: Settings, tmp_path: Path
) -> None:
    make_database(DIMENSION)
    hotpot = tmp_path / "hotpot.jsonl"
    hotpot.write_text(
        hotpot_row("q1", ("Alpha", "Beta"), ("Alpha",))
        + "\n"
        + hotpot_row("q2", ("Beta", "Gamma"), ("Gamma",)),
        encoding="utf-8",
    )

    main(["--hotpot", str(hotpot)], HashingEmbedder(), settings)

    store, _ = stored(settings)
    assert store.document_ids() == {"hotpot:Alpha", "hotpot:Beta", "hotpot:Gamma"}


def test_a_document_limit_keeps_every_document_the_questions_need(
    make_database: Callable[[int], Database], settings: Settings, tmp_path: Path
) -> None:
    make_database(DIMENSION)
    hotpot = tmp_path / "hotpot.jsonl"
    titles = tuple(f"Title{number}" for number in range(10))
    hotpot.write_text(hotpot_row("q1", titles, ("Title3", "Title7")), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    write_questions(
        questions,
        [
            EvalQuestion(
                id="q1",
                text="Question q1?",
                category="multi_step",
                gold_answers=("yes",),
                gold_document_ids=("hotpot:Title3", "hotpot:Title7"),
                expected_status="answered",
                origin="hotpotqa",
            )
        ],
    )

    main(
        ["--hotpot", str(hotpot), "--questions", str(questions), "--max-documents", "4"],
        HashingEmbedder(),
        settings,
    )

    store, _ = stored(settings)
    ids = store.document_ids()
    assert len(ids) == 4
    assert {"hotpot:Title3", "hotpot:Title7"} <= ids


def test_seeding_without_a_database_url_explains_what_is_missing(
    samples: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--samples", str(samples)], HashingEmbedder(), Settings(_env_file=None))

    assert code == 2
    assert "DATABASE_URL" in capsys.readouterr().err


def test_seeding_without_any_source_explains_what_to_pass(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([], HashingEmbedder(), settings)

    assert code == 2
    assert "--samples" in capsys.readouterr().err

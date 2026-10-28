import json
from pathlib import Path

import httpx
import pytest

from investigation_engine.evaluation.hotpot import (
    DownloadError,
    HotpotExample,
    build_documents,
    download_validation,
    gold_document_ids,
    load_examples,
    parse_row,
)


def make_row(
    row_id: str = "q1",
    titles: tuple[str, ...] = ("Alpha", "Beta", "Gamma"),
    supporting: tuple[str, ...] = ("Alpha", "Beta"),
    kind: str = "bridge",
) -> dict[str, object]:
    return {
        "id": row_id,
        "question": f"Question {row_id}?",
        "answer": "yes",
        "type": kind,
        "level": "hard",
        "supporting_facts": {"title": list(supporting), "sent_id": [0] * len(supporting)},
        "context": {
            "title": list(titles),
            "sentences": [[f"{title} is a thing.", " It has facts."] for title in titles],
        },
    }


def test_parse_row_reads_question_answer_type_and_supporting_titles() -> None:
    example = parse_row(make_row())

    assert (example.id, example.question, example.answer) == ("q1", "Question q1?", "yes")
    assert (example.type, example.level) == ("bridge", "hard")
    assert example.supporting_titles == ("Alpha", "Beta")


def test_parse_row_joins_sentences_into_one_paragraph_per_title() -> None:
    example = parse_row(make_row(titles=("Alpha",), supporting=("Alpha",)))

    assert example.paragraphs == (("Alpha", "Alpha is a thing. It has facts."),)


def test_parse_row_keeps_each_supporting_title_once() -> None:
    row = make_row(supporting=("Alpha", "Alpha", "Beta"))

    assert parse_row(row).supporting_titles == ("Alpha", "Beta")


@pytest.mark.parametrize("missing", ["question", "answer", "context", "supporting_facts"])
def test_parse_row_rejects_rows_with_missing_fields(missing: str) -> None:
    row = make_row()
    del row[missing]

    with pytest.raises(ValueError, match=missing):
        parse_row(row)


def test_load_examples_reads_one_example_per_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps(make_row(row_id)) for row_id in ("q1", "q2")) + "\n",
        encoding="utf-8",
    )

    assert [example.id for example in load_examples(path)] == ["q1", "q2"]


def make_example(
    row_id: str, titles: tuple[str, ...], supporting: tuple[str, ...]
) -> HotpotExample:
    return parse_row(make_row(row_id, titles, supporting))


def test_build_documents_creates_one_document_per_distinct_title() -> None:
    examples = [
        make_example("q1", ("Alpha", "Beta"), ("Alpha", "Beta")),
        make_example("q2", ("Beta", "Gamma"), ("Beta", "Gamma")),
    ]

    documents = build_documents(examples)

    assert [document.id for document in documents] == [
        "hotpot:Alpha",
        "hotpot:Beta",
        "hotpot:Gamma",
    ]
    assert documents[0].title == "Alpha"
    assert documents[0].text == "Alpha is a thing. It has facts."
    assert documents[0].metadata["format"] == "hotpot"


def test_build_documents_keeps_the_first_text_seen_for_a_title() -> None:
    first = make_example("q1", ("Alpha",), ("Alpha",))
    other_row = make_row("q2", ("Alpha",), ("Alpha",))
    other_row["context"] = {"title": ["Alpha"], "sentences": [["A different text."]]}

    documents = build_documents([first, parse_row(other_row)])

    assert [document.text for document in documents] == ["Alpha is a thing. It has facts."]


def test_gold_document_ids_follow_the_supporting_titles() -> None:
    example = make_example("q1", ("Alpha", "Beta", "Gamma"), ("Beta", "Alpha"))

    assert gold_document_ids(example) == ("hotpot:Beta", "hotpot:Alpha")


def page_handler(total: int, requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params["offset"])
        length = int(request.url.params["length"])
        rows = [
            {"row_idx": n, "row": make_row(f"q{n}")}
            for n in range(offset, min(offset + length, total))
        ]
        return httpx.Response(200, json={"rows": rows, "num_rows_total": total})

    return handler


def test_download_validation_pages_through_every_row(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    client = httpx.Client(transport=httpx.MockTransport(page_handler(250, requests)))
    destination = tmp_path / "out" / "rows.jsonl"

    count = download_validation(client, destination, page_size=100)

    assert count == 250
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 250
    assert [int(r.url.params["offset"]) for r in requests] == [0, 100, 200]
    assert requests[0].url.params["dataset"] == "hotpotqa/hotpot_qa"
    assert requests[0].url.params["config"] == "distractor"
    assert requests[0].url.params["split"] == "validation"


def test_download_validation_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(page_handler(10, [])))
    destination = tmp_path / "rows.jsonl"

    download_validation(client, destination, page_size=100)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["rows.jsonl"]


def test_download_validation_retries_a_page_that_fails_temporarily(tmp_path: Path) -> None:
    attempts: list[int] = []
    sleeps: list[float] = []
    good = page_handler(5, [])

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503)
        return good(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    count = download_validation(client, tmp_path / "rows.jsonl", sleep=sleeps.append)

    assert count == 5
    assert len(attempts) == 2
    assert sleeps == [1.0]


def test_download_validation_gives_up_after_the_allowed_attempts(tmp_path: Path) -> None:
    sleeps: list[float] = []

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request)

    client = httpx.Client(transport=httpx.MockTransport(failing))
    destination = tmp_path / "rows.jsonl"

    with pytest.raises(DownloadError, match="500"):
        download_validation(client, destination, attempts=3, sleep=sleeps.append)

    assert sleeps == [1.0, 2.0]
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []

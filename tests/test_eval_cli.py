import json
from pathlib import Path

import pytest
from investigation_engine.evaluation.cli import CHUNKERS, main

from fakes import HashingEmbedder
from investigation_engine.evaluation.questions import EvalQuestion, read_questions, write_questions


def hotpot_row(
    row_id: str, kind: str, titles: tuple[str, ...], answer: str = "yes"
) -> dict[str, object]:
    return {
        "id": row_id,
        "question": f"Question about {titles[0]} and {titles[1]}?",
        "answer": answer,
        "type": kind,
        "level": "hard",
        "supporting_facts": {"title": list(titles[:2]), "sent_id": [0, 0]},
        "context": {
            "title": list(titles),
            "sentences": [[f"{title} is described here with some words."] for title in titles],
        },
    }


def write_hotpot(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


@pytest.fixture
def hotpot_file(tmp_path: Path) -> Path:
    rows = [
        hotpot_row(f"b{n:02d}", "bridge", (f"Bridge{n}A", f"Bridge{n}B", f"Filler{n}"))
        for n in range(8)
    ] + [
        hotpot_row(f"c{n:02d}", "comparison", (f"Compare{n}A", f"Compare{n}B", f"Extra{n}"))
        for n in range(4)
    ]
    path = tmp_path / "hotpot.jsonl"
    write_hotpot(path, rows)
    return path


def test_build_questions_writes_a_reproducible_sample(tmp_path: Path, hotpot_file: Path) -> None:
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    arguments = ["build-questions", "--hotpot", str(hotpot_file), "--count", "6", "--seed", "1"]

    assert main([*arguments, "--out", str(first)]) == 0
    assert main([*arguments, "--out", str(second)]) == 0

    questions = read_questions(first)
    assert len(questions) == 6
    assert [question.category for question in questions].count("multi_step") == 4
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


@pytest.fixture
def experiment_inputs(tmp_path: Path, hotpot_file: Path) -> dict[str, Path]:
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "runbook.txt").write_text(
        "Restart the gateway with the restart command.", encoding="utf-8"
    )
    (samples / "notes.md").write_text(
        "# Notes\n\nThe ledger backlog was drained.", encoding="utf-8"
    )
    questions = tmp_path / "questions.jsonl"
    write_questions(
        questions,
        [
            EvalQuestion(
                "b00",
                "Question about Bridge0A and Bridge0B?",
                "multi_step",
                ("yes",),
                ("hotpot:Bridge0A", "hotpot:Bridge0B"),
                "answered",
                "hotpotqa",
            ),
            EvalQuestion(
                "inc-1",
                "How is the gateway restarted?",
                "simple",
                ("restart",),
                ("runbook.txt",),
                "answered",
                "incidents",
            ),
        ],
    )
    return {"hotpot": hotpot_file, "samples": samples, "questions": questions}


def experiment_arguments(command: str, inputs: dict[str, Path], out: Path) -> list[str]:
    return [
        command,
        "--hotpot",
        str(inputs["hotpot"]),
        "--samples",
        str(inputs["samples"]),
        "--questions",
        str(inputs["questions"]),
        "--out",
        str(out),
    ]


def test_retrieval_command_reports_every_configuration_and_prints_tables(
    tmp_path: Path, experiment_inputs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.json"

    code = main(experiment_arguments("retrieval", experiment_inputs, out), HashingEmbedder())

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert [report["config"] for report in payload["reports"]] == [
        "bm25_only",
        "vector_only",
        "hybrid_rrf",
        "hybrid_rerank",
    ]
    assert payload["reports"][0]["documents"] == 36 + 2
    assert "| config |" in capsys.readouterr().out


def test_the_document_limit_still_keeps_every_gold_document(
    tmp_path: Path, experiment_inputs: dict[str, Path]
) -> None:
    out = tmp_path / "report.json"
    arguments = [*experiment_arguments("retrieval", experiment_inputs, out), "--max-documents", "5"]

    main(arguments, HashingEmbedder())

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["reports"][0]["documents"] == 5 + 2
    assert payload["reports"][2]["metrics"]["all"]["recall@10"] == 1.0


def test_chunking_command_compares_every_chunker(
    tmp_path: Path, experiment_inputs: dict[str, Path]
) -> None:
    out = tmp_path / "chunking.json"

    code = main(experiment_arguments("chunking", experiment_inputs, out), HashingEmbedder())

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert {report["chunker"] for report in payload["reports"]} == set(CHUNKERS)
    assert all(report["config"].startswith(report["chunker"]) for report in payload["reports"])


def test_an_unknown_chunker_is_rejected(tmp_path: Path, experiment_inputs: dict[str, Path]) -> None:
    arguments = [*experiment_arguments("retrieval", experiment_inputs, tmp_path / "x.json")]

    with pytest.raises(SystemExit):
        main([*arguments, "--chunker", "nonsense"], HashingEmbedder())

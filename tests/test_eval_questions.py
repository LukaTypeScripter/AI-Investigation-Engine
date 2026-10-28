from pathlib import Path

import pytest

from investigation_engine.evaluation.hotpot import HotpotExample
from investigation_engine.evaluation.questions import (
    EvalQuestion,
    read_questions,
    sample_hotpot_questions,
    write_questions,
)


def make_example(number: int, kind: str) -> HotpotExample:
    title_a, title_b = f"A{number}", f"B{number}"
    return HotpotExample(
        id=f"q{number:03d}",
        question=f"Question {number}?",
        answer=f"answer {number}",
        type=kind,
        level="hard",
        supporting_titles=(title_a, title_b),
        paragraphs=((title_a, "text a"), (title_b, "text b"), ("Other", "text c")),
    )


def make_examples() -> list[HotpotExample]:
    bridges = [make_example(number, "bridge") for number in range(40)]
    comparisons = [make_example(number, "comparison") for number in range(40, 60)]
    return bridges + comparisons


def test_sampling_returns_the_requested_number_with_a_two_to_one_mix_of_bridge_questions() -> None:
    questions = sample_hotpot_questions(make_examples(), count=30, seed=7)

    assert len(questions) == 30
    categories = [question.category for question in questions]
    assert categories.count("multi_step") == 20
    assert categories.count("multi_document") == 10


def test_sampling_is_reproducible_for_the_same_seed_and_differs_for_another_seed() -> None:
    examples = make_examples()

    first = sample_hotpot_questions(examples, count=12, seed=1)
    again = sample_hotpot_questions(list(reversed(examples)), count=12, seed=1)
    other = sample_hotpot_questions(examples, count=12, seed=2)

    assert [q.id for q in first] == [q.id for q in again]
    assert [q.id for q in first] != [q.id for q in other]


def test_sampled_questions_carry_the_gold_answer_and_gold_documents() -> None:
    question = sample_hotpot_questions(make_examples(), count=3, seed=0)[0]

    assert question.origin == "hotpotqa"
    assert question.expected_status == "answered"
    assert question.gold_answers == (f"answer {int(question.id[1:])}",)
    assert question.gold_document_ids == (
        f"hotpot:A{int(question.id[1:])}",
        f"hotpot:B{int(question.id[1:])}",
    )


def test_sampling_more_than_available_is_rejected() -> None:
    with pytest.raises(ValueError, match="only"):
        sample_hotpot_questions(make_examples()[:3], count=30, seed=0)


def test_questions_round_trip_through_a_jsonl_file(tmp_path: Path) -> None:
    questions = [
        EvalQuestion(
            id="inc-1",
            text="What happened?",
            category="conflicting",
            gold_answers=(),
            gold_document_ids=("a.md", "b.md"),
            expected_status="conflicting_evidence",
            origin="incidents",
        ),
        EvalQuestion("inc-2", "Who?", "unanswerable", (), (), "insufficient_evidence", "incidents"),
    ]
    path = tmp_path / "questions.jsonl"

    write_questions(path, questions)

    assert read_questions(path) == questions


def test_reading_a_question_with_an_unknown_category_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id": "x", "text": "t", "category": "weird", "gold_answers": [], '
        '"gold_document_ids": [], "expected_status": "answered", "origin": "o"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="category"):
        read_questions(path)

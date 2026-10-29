import re
import string
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = frozenset(string.punctuation)
TYPOGRAPHIC_TO_ASCII = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)
YES_NO_ANSWERS = frozenset({"yes", "no"})


@dataclass(frozen=True, slots=True)
class CitationScores:
    precision: float
    recall: float


def normalize_answer(text: str) -> str:
    lowered = text.translate(TYPOGRAPHIC_TO_ASCII).lower()
    without_punctuation = "".join(
        character for character in lowered if character not in PUNCTUATION
    )
    return " ".join(ARTICLES.sub(" ", without_punctuation).split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    predicted_text = normalize_answer(prediction)
    gold_text = normalize_answer(gold)
    if predicted_text in YES_NO_ANSWERS or gold_text in YES_NO_ANSWERS:
        return float(predicted_text == gold_text)
    predicted = predicted_text.split()
    expected = gold_text.split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, gold: str) -> bool:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not expected:
        return False
    return any(
        predicted[start : start + len(expected)] == expected
        for start in range(len(predicted) - len(expected) + 1)
    )


def recall_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(ranked[:k]) & gold_set) / len(gold_set)


def full_recall_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    gold_set = set(gold)
    return float(bool(gold_set) and gold_set <= set(ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], gold: Collection[str]) -> float:
    gold_set = set(gold)
    for position, item in enumerate(ranked, start=1):
        if item in gold_set:
            return 1 / position
    return 0.0


def citation_scores(cited: Collection[str], gold: Collection[str]) -> CitationScores:
    cited_set, gold_set = set(cited), set(gold)
    correct = len(cited_set & gold_set)
    return CitationScores(
        precision=correct / len(cited_set) if cited_set else 0.0,
        recall=correct / len(gold_set) if gold_set else 0.0,
    )


def mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0

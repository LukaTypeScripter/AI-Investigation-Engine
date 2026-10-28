import heapq
import math
import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        self._k1 = k1
        self._b = b
        self._postings: dict[str, dict[str, int]] = {}
        self._term_counts: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, item_id: str, text: str) -> None:
        self.remove(item_id)
        counts = Counter(tokenize(text))
        length = sum(counts.values())
        self._term_counts[item_id] = counts
        self._lengths[item_id] = length
        self._total_length += length
        for term, count in counts.items():
            self._postings.setdefault(term, {})[item_id] = count

    def remove(self, item_id: str) -> None:
        counts = self._term_counts.pop(item_id, None)
        if counts is None:
            return
        self._total_length -= self._lengths.pop(item_id)
        for term in counts:
            postings = self._postings[term]
            del postings[item_id]
            if not postings:
                del self._postings[term]

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if not self._lengths:
            return []
        average_length = self._total_length / len(self._lengths)
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._inverse_document_frequency(len(postings))
            for item_id, frequency in postings.items():
                weight = self._term_weight(frequency, self._lengths[item_id], average_length)
                scores[item_id] = scores.get(item_id, 0.0) + idf * weight
        return heapq.nsmallest(limit, scores.items(), key=lambda pair: (-pair[1], pair[0]))

    def _inverse_document_frequency(self, document_frequency: int) -> float:
        total = len(self._lengths)
        return math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))

    def _term_weight(self, frequency: int, length: int, average_length: float) -> float:
        length_penalty = 1 - self._b + self._b * length / average_length
        return frequency * (self._k1 + 1) / (frequency + self._k1 * length_penalty)

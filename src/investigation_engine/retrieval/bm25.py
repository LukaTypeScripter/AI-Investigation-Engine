import heapq
import math
import re
import sys
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
        self._terms: dict[str, tuple[str, ...]] = {}
        self._lengths: dict[str, int] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, item_id: str, text: str) -> None:
        self.remove(item_id)
        counts = Counter(sys.intern(token) for token in tokenize(text))
        length = sum(counts.values())
        self._terms[item_id] = tuple(counts)
        self._lengths[item_id] = length
        self._total_length += length
        for term, count in counts.items():
            self._postings.setdefault(term, {})[item_id] = count

    def remove(self, item_id: str) -> None:
        terms = self._terms.pop(item_id, None)
        if terms is None:
            return
        self._total_length -= self._lengths.pop(item_id)
        for term in terms:
            postings = self._postings[term]
            del postings[item_id]
            if not postings:
                del self._postings[term]

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if not self._lengths or limit <= 0:
            return []
        weighted_terms = self._weighted_terms(query)
        average_length = self._total_length / len(self._lengths)
        length_base = self._k1 * (1 - self._b)
        length_slope = self._k1 * self._b / average_length
        remaining_bounds = self._remaining_bounds(weighted_terms)
        scores: dict[str, float] = {}
        for (weight, postings), remaining_bound in zip(
            weighted_terms, remaining_bounds, strict=True
        ):
            if self._is_settled(scores, limit, remaining_bound):
                self._update_candidates(scores, weight, postings, length_base, length_slope)
            else:
                self._update_all(scores, weight, postings, length_base, length_slope)
        return heapq.nsmallest(limit, scores.items(), key=lambda pair: (-pair[1], pair[0]))

    def _weighted_terms(self, query: str) -> list[tuple[float, dict[str, int]]]:
        weighted: list[tuple[float, str, dict[str, int]]] = []
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if postings:
                weighted.append((self._weight(len(postings)), term, postings))
        weighted.sort(key=lambda entry: (-entry[0], entry[1]))
        return [(weight, postings) for weight, _, postings in weighted]

    def _weight(self, document_frequency: int) -> float:
        total = len(self._lengths)
        inverse_document_frequency = math.log(
            1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        return inverse_document_frequency * (self._k1 + 1)

    @staticmethod
    def _remaining_bounds(weighted_terms: list[tuple[float, dict[str, int]]]) -> list[float]:
        bounds: list[float] = []
        running = 0.0
        for weight, _ in reversed(weighted_terms):
            running += weight
            bounds.append(running)
        bounds.reverse()
        return bounds

    @staticmethod
    def _is_settled(scores: dict[str, float], limit: int, remaining_bound: float) -> bool:
        if len(scores) < limit:
            return False
        return heapq.nlargest(limit, scores.values())[-1] > remaining_bound

    def _update_all(
        self,
        scores: dict[str, float],
        weight: float,
        postings: dict[str, int],
        length_base: float,
        length_slope: float,
    ) -> None:
        lengths = self._lengths
        for item_id, frequency in postings.items():
            contribution = (
                weight * frequency / (frequency + length_base + length_slope * lengths[item_id])
            )
            scores[item_id] = scores.get(item_id, 0.0) + contribution

    def _update_candidates(
        self,
        scores: dict[str, float],
        weight: float,
        postings: dict[str, int],
        length_base: float,
        length_slope: float,
    ) -> None:
        lengths = self._lengths
        if len(scores) <= len(postings):
            matches = ((item_id, postings.get(item_id)) for item_id in scores)
        else:
            matches = (
                (item_id, frequency) for item_id, frequency in postings.items() if item_id in scores
            )
        for item_id, frequency in matches:
            if frequency:
                scores[item_id] += (
                    weight * frequency / (frequency + length_base + length_slope * lengths[item_id])
                )

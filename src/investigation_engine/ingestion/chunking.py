import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from investigation_engine.domain import Chunk, Document

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


def build_chunks(document: Document, texts: Iterable[str]) -> list[Chunk]:
    return [
        Chunk(
            id=f"{document.id}:{index}",
            document_id=document.id,
            index=index,
            text=text,
            title=document.title,
            source=document.source,
        )
        for index, text in enumerate(texts)
    ]


def split_into_windows(words: list[str], size: int, stride: int) -> Iterator[str]:
    for start in range(0, len(words), stride):
        yield " ".join(words[start : start + size])
        if start + size >= len(words):
            return


@dataclass(frozen=True, slots=True)
class FixedWindowChunker:
    size: int
    overlap: int

    def __post_init__(self) -> None:
        if self.size <= 0 or not 0 <= self.overlap < self.size:
            raise ValueError("window size must be positive and overlap must be in [0, size)")

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        windows = split_into_windows(words, self.size, self.size - self.overlap)
        return build_chunks(document, windows)


@dataclass(frozen=True, slots=True)
class SentenceChunker:
    max_words: int

    def __post_init__(self) -> None:
        if self.max_words <= 0:
            raise ValueError("max_words must be positive")

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = [
            " ".join(part.split())
            for part in SENTENCE_BOUNDARY.split(document.text)
            if part.strip()
        ]
        return build_chunks(document, self._pack(sentences))

    def _pack(self, sentences: list[str]) -> Iterator[str]:
        buffer: list[str] = []
        buffer_words = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if buffer and buffer_words + sentence_words > self.max_words:
                yield " ".join(buffer)
                buffer, buffer_words = [], 0
            if sentence_words > self.max_words:
                yield from split_into_windows(sentence.split(), self.max_words, self.max_words)
                continue
            buffer.append(sentence)
            buffer_words += sentence_words
        if buffer:
            yield " ".join(buffer)

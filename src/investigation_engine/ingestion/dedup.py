import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from investigation_engine.domain import Chunk

WHITESPACE = re.compile(r"\s+")


def fingerprint(text: str) -> str:
    normalized = WHITESPACE.sub(" ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DedupResult:
    unique: list[Chunk]
    duplicate_of: dict[str, str]


def split_duplicates(
    chunks: Iterable[Chunk], known: Mapping[str, str] | None = None
) -> DedupResult:
    canonical = dict(known or {})
    unique: list[Chunk] = []
    duplicate_of: dict[str, str] = {}
    for chunk in chunks:
        key = fingerprint(chunk.text)
        if key in canonical:
            duplicate_of[chunk.id] = canonical[key]
        else:
            canonical[key] = chunk.id
            unique.append(chunk)
    return DedupResult(unique=unique, duplicate_of=duplicate_of)

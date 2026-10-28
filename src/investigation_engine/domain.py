from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    title: str = ""
    source: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    index: int
    text: str
    title: str
    source: str

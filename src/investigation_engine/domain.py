from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceRef:
    document_id: str
    title: str
    source: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    index: int
    text: str
    title: str
    source: str

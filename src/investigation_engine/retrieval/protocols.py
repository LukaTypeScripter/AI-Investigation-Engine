from collections.abc import Sequence
from typing import Protocol


class KeywordSearch(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...


class VectorSearch(Protocol):
    def search(self, vector: Sequence[float], limit: int) -> list[tuple[str, float]]: ...


class KeywordIndex(KeywordSearch, Protocol):
    def add(self, item_id: str, text: str) -> None: ...

    def remove(self, item_id: str) -> None: ...


class VectorIndex(VectorSearch, Protocol):
    def add(self, item_id: str, vector: Sequence[float]) -> None: ...

    def add_many(self, items: Sequence[tuple[str, Sequence[float]]]) -> None: ...

    def remove(self, item_id: str) -> None: ...

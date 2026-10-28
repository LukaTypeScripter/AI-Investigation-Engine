import threading

from investigation_engine.retrieval.protocols import KeywordIndex


class LockedKeywordIndex:
    def __init__(self, inner: KeywordIndex) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def add(self, item_id: str, text: str) -> None:
        with self._lock:
            self._inner.add(item_id, text)

    def remove(self, item_id: str) -> None:
        with self._lock:
            self._inner.remove(item_id)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        with self._lock:
            return self._inner.search(query, limit)

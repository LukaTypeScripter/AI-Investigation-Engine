import threading

from investigation_engine.retrieval.bm25 import Bm25Index
from investigation_engine.retrieval.locked import LockedKeywordIndex

WAIT_SECONDS = 5.0
SHORT_WAIT_SECONDS = 0.2


class BlockingKeywordIndex:
    def __init__(self) -> None:
        self.search_started = threading.Event()
        self.release_search = threading.Event()
        self.add_finished = threading.Event()
        self.added: list[tuple[str, str]] = []
        self.removed: list[str] = []

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        self.search_started.set()
        self.release_search.wait(WAIT_SECONDS)
        return [(f"{query}:{limit}", 1.0)]

    def add(self, item_id: str, text: str) -> None:
        self.added.append((item_id, text))
        self.add_finished.set()

    def remove(self, item_id: str) -> None:
        self.removed.append(item_id)
        self.add_finished.set()


def test_the_locked_index_behaves_like_the_index_it_wraps() -> None:
    index = LockedKeywordIndex(Bm25Index())

    index.add("a", "payments failed")
    index.add("b", "refunds late")
    index.remove("b")

    assert [item_id for item_id, _ in index.search("payments refunds", 5)] == ["a"]


def test_a_write_waits_until_a_running_search_has_finished() -> None:
    inner = BlockingKeywordIndex()
    index = LockedKeywordIndex(inner)
    searcher = threading.Thread(target=index.search, args=("q", 1))
    writer = threading.Thread(target=index.add, args=("a", "text"))
    searcher.start()
    assert inner.search_started.wait(WAIT_SECONDS)

    writer.start()
    wrote_while_searching = inner.add_finished.wait(SHORT_WAIT_SECONDS)
    inner.release_search.set()
    searcher.join(WAIT_SECONDS)
    writer.join(WAIT_SECONDS)

    assert not wrote_while_searching
    assert inner.added == [("a", "text")]


def test_a_remove_waits_until_a_running_search_has_finished() -> None:
    inner = BlockingKeywordIndex()
    index = LockedKeywordIndex(inner)
    searcher = threading.Thread(target=index.search, args=("q", 1))
    remover = threading.Thread(target=index.remove, args=("a",))
    searcher.start()
    assert inner.search_started.wait(WAIT_SECONDS)

    remover.start()
    removed_while_searching = inner.add_finished.wait(SHORT_WAIT_SECONDS)
    inner.release_search.set()
    searcher.join(WAIT_SECONDS)
    remover.join(WAIT_SECONDS)

    assert not removed_while_searching
    assert inner.removed == ["a"]

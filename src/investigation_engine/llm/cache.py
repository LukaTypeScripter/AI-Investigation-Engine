import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from investigation_engine.llm.client import ChatClient
from investigation_engine.llm.types import Completion, Message, ToolSpec
from investigation_engine.llm.wire import build_request_body, completion_to_wire, parse_completion


class ResponseCache(Protocol):
    def get(self, key: str) -> Completion | None: ...

    def put(self, key: str, completion: Completion) -> None: ...


def cache_key(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    temperature: float,
    max_tokens: int | None,
) -> str:
    body = build_request_body(model, messages, tools, temperature, max_tokens)
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


class InMemoryResponseCache:
    def __init__(self) -> None:
        self._entries: dict[str, Completion] = {}

    def get(self, key: str) -> Completion | None:
        return self._entries.get(key)

    def put(self, key: str, completion: Completion) -> None:
        self._entries[key] = completion


class SqliteResponseCache:
    def __init__(self, path: Path | str) -> None:
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return parse_completion(json.loads(row[0])) if row else None

    def put(self, key: str, completion: Completion) -> None:
        payload = json.dumps(completion_to_wire(completion))
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO responses (key, payload) VALUES (?, ?)", (key, payload)
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachedChatClient:
    def __init__(self, inner: ChatClient, cache: ResponseCache) -> None:
        self.model = inner.model
        self.hits = 0
        self.misses = 0
        self._inner = inner
        self._cache = cache

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Completion:
        key = cache_key(self.model, messages, tools, temperature, max_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        completion = self._inner.complete(messages, tools, temperature, max_tokens)
        self._cache.put(key, completion)
        return completion

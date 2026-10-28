from collections.abc import Sequence

import numpy as np


class InMemoryVectorIndex:
    def __init__(self, dimension: int, initial_capacity: int = 1024) -> None:
        if dimension <= 0 or initial_capacity <= 0:
            raise ValueError("dimension and initial_capacity must be positive")
        self._dimension = dimension
        self._matrix = np.zeros((initial_capacity, dimension), dtype=np.float32)
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, item_id: str, vector: Sequence[float]) -> None:
        unit = self._unit(vector)
        if unit is None:
            raise ValueError("cannot index a zero vector")
        position = self._positions.get(item_id)
        if position is None:
            position = len(self._ids)
            if position == self._matrix.shape[0]:
                self._grow()
            self._ids.append(item_id)
            self._positions[item_id] = position
        self._matrix[position] = unit

    def remove(self, item_id: str) -> None:
        position = self._positions.pop(item_id, None)
        if position is None:
            return
        last = len(self._ids) - 1
        if position != last:
            moved_id = self._ids[last]
            self._matrix[position] = self._matrix[last]
            self._ids[position] = moved_id
            self._positions[moved_id] = position
        self._ids.pop()

    def search(self, vector: Sequence[float], limit: int) -> list[tuple[str, float]]:
        query = self._unit(vector)
        if limit <= 0 or not self._ids or query is None:
            return []
        scores = self._matrix[: len(self._ids)] @ query
        best = np.argsort(-scores, kind="stable")[:limit]
        ranked = [(self._ids[int(position)], float(scores[position])) for position in best]
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))

    def _unit(self, vector: Sequence[float]) -> np.ndarray | None:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self._dimension,):
            raise ValueError(f"expected a vector with dimension {self._dimension}")
        norm = float(np.linalg.norm(array))
        return array / norm if norm > 0 else None

    def _grow(self) -> None:
        self._matrix = np.vstack([self._matrix, np.zeros_like(self._matrix)])

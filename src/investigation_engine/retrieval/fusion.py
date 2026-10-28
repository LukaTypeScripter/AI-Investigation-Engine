from collections.abc import Sequence

DEFAULT_RANK_CONSTANT = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = DEFAULT_RANK_CONSTANT
) -> list[tuple[str, float]]:
    if k < 0:
        raise ValueError("rank constant must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))

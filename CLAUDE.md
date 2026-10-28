# Investigation Engine

Multi-document investigation engine: hybrid retrieval (semantic and keyword), multi-step RAG with citations, abstention and contradiction flags, and a reproducible evaluation harness.

## Principles

- Keep the pipeline simple, readable and hand-written. No heavyweight RAG frameworks. Every design choice is documented in the README.
- Python has no comments and no docstrings. Follow `.claude/skills/clean-python`.
- Configuration goes through typed settings. Never commit `.env` or secrets.
- Write the failing test first. Run the full gate below before calling anything done.

## Commits

Follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<optional scope>): <description>`.

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- Scopes name the feature or module, for example `chunking`, `dedup`, `bm25`, `fusion`, `rag`, `eval`, `api`.
- Each commit contains only the files of one feature, with its tests.
- The description is lowercase, imperative, has no trailing period, and is 72 characters or fewer.
- The body explains why, not what. Mark breaking changes with `!` after the type or a `BREAKING CHANGE:` footer.
- One logical change per commit, together with its tests.

## Tooling

Python 3.12 with `uv` only (no pip, no bare `python`).

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run python tools/check_no_comments.py
```

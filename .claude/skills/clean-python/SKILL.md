---
name: clean-python
description: House style for this repository. Python with no comments and no docstrings, where names, types and small functions carry the meaning. Use whenever writing, editing or reviewing any Python in this project, and whenever another installed skill shows examples with docstrings, comments, pip or mypy.
---

# Clean Python (no comments)

## Rules

- No `#` comments and no docstrings anywhere in `src/`, `tests/` or `tools/`. `tools/check_no_comments.py` enforces it, and a test runs it on the whole repo.
- Tool pragmas (`noqa`, `ty: ignore[rule]`, `type:`) are the only allowed comments, and only when the user asks. Fix the cause instead of suppressing it.
- Put the reason into a descriptive name, a named constant, a small function, a type, or a test name. Longer rationale belongs in markdown (README, `docs/`), never in code.
- Annotate every function. Use `X | None`, never `Optional[X]`. Avoid `Any`.
- One job per function, early returns instead of nesting, no boolean flag arguments.
- Full-word names, no abbreviations. Functions are verbs, booleans read as `is_` or `has_`.
- Replace magic values with named constants or settings.
- FastAPI: describe endpoints with `summary=` and `description=` in the decorator, not docstrings.
- Errors: specific exception types, chained with `raise ... from error`. No bare `except`, no silent `pass`.
- Configuration comes from typed settings (pydantic-settings). Business code never reads `os.environ`, and secrets are never committed.
- Tests are named `test_<unit>_<scenario>_<expected>` and separate arrange, act and assert with blank lines, not comments.

## When another skill disagrees

Other skills and online examples often show docstrings, comments, `pip`, `mypy` and LangChain. Take their ideas, not those habits. In this project use `uv`, `ruff`, `ty`, `pytest` and a hand-written retrieval pipeline.

## Before finishing any change

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run python tools/check_no_comments.py && uv run pytest -q
```

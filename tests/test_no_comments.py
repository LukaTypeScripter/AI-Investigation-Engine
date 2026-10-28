from pathlib import Path

import pytest

from check_no_comments import find_violations, scan_paths

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_sample(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_clean_module_has_no_violations(tmp_path: Path) -> None:
    path = write_sample(tmp_path, "def add(a: int, b: int) -> int:\n    return a + b\n")

    assert find_violations(path) == []


def test_full_line_comment_is_reported_with_its_line_number(tmp_path: Path) -> None:
    path = write_sample(tmp_path, "value = 1\n# explains nothing\nother = 2\n")

    violations = find_violations(path)

    assert [(v.line, v.kind) for v in violations] == [(2, "comment")]


def test_inline_comment_is_reported(tmp_path: Path) -> None:
    path = write_sample(tmp_path, "value = 1  # one\n")

    assert [(v.line, v.kind) for v in find_violations(path)] == [(1, "comment")]


def test_hash_inside_a_string_is_not_a_comment(tmp_path: Path) -> None:
    path = write_sample(tmp_path, 'text = "# not a comment"\n')

    assert find_violations(path) == []


@pytest.mark.parametrize(
    "line",
    [
        "import os  # noqa: F401",
        "value = compute()  # ty: ignore[invalid-argument-type]",
        "value = compute()  # type: ignore",
    ],
)
def test_tool_pragmas_are_allowed(tmp_path: Path, line: str) -> None:
    path = write_sample(tmp_path, line + "\n")

    assert find_violations(path) == []


def test_module_docstring_is_reported(tmp_path: Path) -> None:
    path = write_sample(tmp_path, '"""About this module."""\n\nvalue = 1\n')

    assert [(v.line, v.kind) for v in find_violations(path)] == [(1, "docstring")]


def test_function_docstring_is_reported_at_the_docstring_line(tmp_path: Path) -> None:
    path = write_sample(tmp_path, 'def one() -> int:\n    """Return one."""\n    return 1\n')

    assert [(v.line, v.kind) for v in find_violations(path)] == [(2, "docstring")]


def test_async_function_and_class_docstrings_are_reported(tmp_path: Path) -> None:
    body = (
        "class Box:\n"
        '    """A box."""\n'
        "\n"
        "    async def open(self) -> None:\n"
        '        """Open it."""\n'
    )
    path = write_sample(tmp_path, body)

    assert [(v.line, v.kind) for v in find_violations(path)] == [
        (2, "docstring"),
        (5, "docstring"),
    ]


def test_violation_renders_as_path_line_kind(tmp_path: Path) -> None:
    path = write_sample(tmp_path, "value = 1  # one\n")

    assert str(find_violations(path)[0]) == f"{path}:1: comment"


def test_scan_walks_nested_directories(tmp_path: Path) -> None:
    nested = tmp_path / "pkg" / "inner"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("value = 1  # one\n", encoding="utf-8")

    violations = scan_paths([tmp_path])

    assert [(v.path.name, v.line) for v in violations] == [("module.py", 1)]


def test_scan_skips_excluded_directories(tmp_path: Path) -> None:
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "vendored.py").write_text("value = 1  # one\n", encoding="utf-8")

    assert scan_paths([tmp_path]) == []


def test_project_source_has_no_comments_or_docstrings() -> None:
    roots = [REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "tools"]

    assert scan_paths(roots) == []

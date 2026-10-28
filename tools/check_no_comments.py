import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

ALLOWED_PRAGMA = re.compile(r"#\s*(noqa|type:|ty:|pragma|fmt:)")
EXCLUDED_DIRS = frozenset(
    {".venv", ".git", ".claude", "__pycache__", "node_modules", ".ruff_cache", ".pytest_cache"}
)
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
DEFAULT_ROOTS = ["src", "tests", "tools"]


class Violation(NamedTuple):
    path: Path
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}"


def comment_violations(path: Path, source: str) -> Iterator[Violation]:
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and not ALLOWED_PRAGMA.match(token.string):
            yield Violation(path, token.start[0], "comment")


def docstring_violations(path: Path, source: str) -> Iterator[Violation]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, DOCSTRING_OWNERS) and ast.get_docstring(node, clean=False) is not None:
            yield Violation(path, node.body[0].lineno, "docstring")


def find_violations(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    found = [*comment_violations(path, source), *docstring_violations(path, source)]
    return sorted(found, key=lambda violation: violation.line)


def python_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*.py")):
        if not EXCLUDED_DIRS.intersection(path.relative_to(root).parts):
            yield path


def scan_paths(roots: Iterable[Path]) -> list[Violation]:
    return [
        violation
        for root in roots
        for path in python_files(root)
        for violation in find_violations(path)
    ]


def main(arguments: list[str]) -> int:
    violations = scan_paths(Path(argument) for argument in arguments)
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_ROOTS))

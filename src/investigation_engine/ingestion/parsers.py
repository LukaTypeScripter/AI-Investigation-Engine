import csv
import io
import json
import re
from collections.abc import Callable, Mapping
from html.parser import HTMLParser
from pathlib import Path

from investigation_engine.domain import Document

Parser = Callable[[str, str], list[Document]]

FIELD_ID = "id"
FIELD_TITLE = "title"
FIELD_TEXT = "text"
RESERVED_FIELDS = frozenset({FIELD_ID, FIELD_TITLE, FIELD_TEXT})

MARKDOWN_TITLE = re.compile(r"^# +(.+?)\s*$", re.MULTILINE)
HTML_BLOCK_TAGS = frozenset(
    [
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    ]
)
HTML_IGNORED_TAGS = frozenset({"script", "style"})
HTML_CAPTURED_TAGS = frozenset({"title", "h1"})


class ParseError(ValueError):
    pass


class UnsupportedFormatError(ValueError):
    pass


def decode(content: bytes) -> str:
    text = content.decode("utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def single_document(source: str, text: str, title: str, metadata: dict[str, str]) -> Document:
    return Document(id=source, text=text.strip(), title=title, source=source, metadata=metadata)


def parse_plain_text(source: str, text: str) -> list[Document]:
    return [single_document(source, text, Path(source).stem, {"format": "txt"})]


def parse_markdown(source: str, text: str) -> list[Document]:
    heading = MARKDOWN_TITLE.search(text)
    title = heading.group(1) if heading else Path(source).stem
    return [single_document(source, text, title, {"format": "md"})]


class HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.heading = ""
        self.meta: dict[str, str] = {}
        self._parts: list[str] = []
        self._ignored_depth = 0
        self._captured_tag: str | None = None
        self._heading_closed = False

    @property
    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).split("\n"))
        return "\n".join(line for line in lines if line)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in HTML_IGNORED_TAGS:
            self._ignored_depth += 1
        elif tag == "meta":
            self._record_meta(dict(attrs))
        elif tag in HTML_BLOCK_TAGS:
            self._parts.append("\n")
        if tag in HTML_CAPTURED_TAGS:
            self._captured_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in HTML_IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in HTML_BLOCK_TAGS:
            self._parts.append("\n")
        if tag == self._captured_tag:
            self._captured_tag = None
        if tag == "h1":
            self._heading_closed = True

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._captured_tag == "title":
            self.title += data
            return
        if self._captured_tag == "h1" and not self._heading_closed:
            self.heading += data
        self._parts.append(data)

    def _record_meta(self, attributes: dict[str, str | None]) -> None:
        name = attributes.get("name")
        content = attributes.get("content")
        if name and content:
            self.meta[name.casefold()] = content


def parse_html(source: str, text: str) -> list[Document]:
    extractor = HtmlTextExtractor()
    extractor.feed(text)
    extractor.close()
    title = " ".join(extractor.title.split()) or " ".join(extractor.heading.split())
    metadata = {**extractor.meta, "format": "html"}
    return [single_document(source, extractor.text, title or Path(source).stem, metadata)]


def scalar_to_text(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, bool | int | float):
        return json.dumps(value)
    return None


def record_metadata(record: Mapping[str, object], file_format: str) -> dict[str, str]:
    extras: dict[str, str] = {}
    for key, value in record.items():
        text = scalar_to_text(value)
        if key not in RESERVED_FIELDS and text is not None:
            extras[key] = text
    return {**extras, "format": file_format}


def record_to_document(source: str, file_format: str, index: int, record: object) -> Document:
    if not isinstance(record, Mapping):
        raise ParseError(f"{source}: record {index} is not an object")
    text = record.get(FIELD_TEXT)
    if not isinstance(text, str):
        raise ParseError(f"{source}: record {index} has no '{FIELD_TEXT}' field")
    record_id = scalar_to_text(record.get(FIELD_ID)) or str(index)
    title = scalar_to_text(record.get(FIELD_TITLE)) or Path(source).stem
    return Document(
        id=f"{source}#{record_id}",
        text=text.strip(),
        title=title,
        source=source,
        metadata=record_metadata(record, file_format),
    )


def parse_json(source: str, text: str) -> list[Document]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ParseError(f"{source}: invalid JSON ({error.msg}, line {error.lineno})") from error
    records = payload if isinstance(payload, list) else [payload]
    return [
        record_to_document(source, "json", index, record) for index, record in enumerate(records)
    ]


def parse_json_lines(source: str, text: str) -> list[Document]:
    documents: list[Document] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ParseError(f"{source}: line {line_number} is not valid JSON") from error
        documents.append(record_to_document(source, "jsonl", line_number - 1, record))
    return documents


def parse_csv(source: str, text: str) -> list[Document]:
    reader = csv.DictReader(io.StringIO(text, newline=""), restkey="_overflow", restval="")
    if reader.fieldnames is None or FIELD_TEXT not in reader.fieldnames:
        raise ParseError(f"{source}: CSV needs a '{FIELD_TEXT}' column")
    return [record_to_document(source, "csv", index, row) for index, row in enumerate(reader)]


PARSERS: dict[str, Parser] = {
    ".txt": parse_plain_text,
    ".md": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".json": parse_json,
    ".jsonl": parse_json_lines,
    ".csv": parse_csv,
}
SUPPORTED_EXTENSIONS = tuple(sorted(PARSERS))


def parse_bytes(filename: str, content: bytes) -> list[Document]:
    source = Path(filename).name
    extension = Path(source).suffix.casefold()
    parser = PARSERS.get(extension)
    if parser is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise UnsupportedFormatError(f"unsupported file type '{extension}', supported: {supported}")
    return [document for document in parser(source, decode(content)) if document.text]


def parse_path(path: Path) -> list[Document]:
    return parse_bytes(path.name, path.read_bytes())

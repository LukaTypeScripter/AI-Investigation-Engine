import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from investigation_engine.domain import Document

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"
DEFAULT_PAGE_SIZE = 100
DEFAULT_ATTEMPTS = 6
PAUSE_BETWEEN_PAGES = 0.5
MAX_WAIT_SECONDS = 60.0
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
DOCUMENT_ID_PREFIX = "hotpot:"
DOCUMENT_SOURCE = "hotpotqa/validation"


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HotpotExample:
    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_titles: tuple[str, ...]
    paragraphs: tuple[tuple[str, str], ...]


def require(row: Mapping[str, object], key: str) -> object:
    if key not in row:
        raise ValueError(f"row is missing '{key}'")
    return row[key]


def require_text(row: Mapping[str, object], key: str) -> str:
    value = require(row, key)
    if not isinstance(value, str):
        raise ValueError(f"row field '{key}' must be text")
    return value


def require_mapping(row: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = require(row, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"row field '{key}' must be an object")
    return value


def require_list(row: Mapping[str, object], key: str) -> list[object]:
    value = require(row, key)
    if not isinstance(value, list):
        raise ValueError(f"row field '{key}' must be a list")
    return value


def join_sentences(sentences: object) -> str:
    if not isinstance(sentences, list):
        raise ValueError("context sentences must be a list of lists")
    return " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())


def parse_row(row: Mapping[str, object]) -> HotpotExample:
    context = require_mapping(row, "context")
    titles = require_list(context, "title")
    sentences = require_list(context, "sentences")
    supporting = require_list(require_mapping(row, "supporting_facts"), "title")
    return HotpotExample(
        id=require_text(row, "id"),
        question=require_text(row, "question"),
        answer=require_text(row, "answer"),
        type=str(row.get("type", "")),
        level=str(row.get("level", "")),
        supporting_titles=tuple(dict.fromkeys(str(title) for title in supporting)),
        paragraphs=tuple(
            (str(title), join_sentences(paragraph))
            for title, paragraph in zip(titles, sentences, strict=True)
        ),
    )


def load_examples(path: Path) -> list[HotpotExample]:
    with path.open(encoding="utf-8") as handle:
        return [parse_row(json.loads(line)) for line in handle if line.strip()]


def document_id_for(title: str) -> str:
    return f"{DOCUMENT_ID_PREFIX}{title}"


def gold_document_ids(example: HotpotExample) -> tuple[str, ...]:
    return tuple(document_id_for(title) for title in example.supporting_titles)


def build_documents(examples: Iterable[HotpotExample]) -> list[Document]:
    documents: dict[str, Document] = {}
    for example in examples:
        for title, text in example.paragraphs:
            document_id = document_id_for(title)
            if document_id not in documents:
                documents[document_id] = Document(
                    id=document_id,
                    text=text,
                    title=title,
                    source=DOCUMENT_SOURCE,
                    metadata={"format": "hotpot"},
                )
    return list(documents.values())


def fetch_page(
    client: httpx.Client,
    offset: int,
    page_size: int,
    attempts: int,
    sleep: Callable[[float], None],
) -> Mapping[str, object]:
    parameters: dict[str, str | int] = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": page_size,
    }
    failure = "no attempt was made"
    for attempt in range(1, attempts + 1):
        advised_wait: float | None = None
        try:
            response = client.get(ROWS_URL, params=parameters)
        except httpx.TransportError as error:
            failure = f"{type(error).__name__} while contacting the dataset server"
        else:
            if response.is_success:
                return response.json()
            failure = f"HTTP {response.status_code} from the dataset server"
            if response.status_code not in RETRYABLE_STATUSES:
                break
            advised_wait = parse_retry_after(response)
        if attempt < attempts:
            sleep(retry_delay(attempt, advised_wait))
    raise DownloadError(f"could not download rows at offset {offset}: {failure}")


def parse_retry_after(response: httpx.Response) -> float | None:
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def retry_delay(attempt: int, advised_wait: float | None) -> float:
    if advised_wait is not None:
        return min(advised_wait, MAX_WAIT_SECONDS)
    return min(float(2 ** (attempt - 1)), MAX_WAIT_SECONDS)


def resume_point(partial: Path) -> int:
    if not partial.exists():
        return 0
    data = partial.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    partial.write_bytes(complete)
    return complete.count(b"\n")


def download_validation(
    client: httpx.Client,
    destination: Path,
    page_size: int = DEFAULT_PAGE_SIZE,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    pause: float = PAUSE_BETWEEN_PAGES,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    count = resume_point(partial)
    total = count + 1
    try:
        with partial.open("a", encoding="utf-8") as handle:
            while count < total:
                if count > 0:
                    sleep(pause)
                payload = fetch_page(client, count, page_size, attempts, sleep)
                rows = payload["rows"]
                total = int(str(payload["num_rows_total"]))
                if not isinstance(rows, list) or not rows:
                    break
                for entry in rows:
                    handle.write(json.dumps(entry["row"], ensure_ascii=False) + "\n")
                handle.flush()
                count += len(rows)
    except DownloadError:
        if count == 0:
            partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    return count

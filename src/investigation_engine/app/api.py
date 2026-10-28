import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from investigation_engine.app.engine import Engine, NothingToIndexError
from investigation_engine.app.schemas import (
    AnswerOut,
    AskIn,
    DocumentPageOut,
    HealthOut,
    SearchIn,
    SearchOut,
    UploadOut,
    answer_out,
    hit_out,
    page_out,
    upload_out,
)
from investigation_engine.ingestion.parsers import ParseError, UnsupportedFormatError
from investigation_engine.llm.errors import LlmError, LlmRateLimited, LlmUnavailable
from investigation_engine.llm.factory import ConfigurationError

logger = logging.getLogger(__name__)

STATIC_DIRECTORY = Path(__file__).parent / "static"
MILLISECONDS = 1000
MAX_FILENAME_LENGTH = 255
MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50

RATE_LIMITED_MESSAGE = "The language model is rate limited. Wait a moment and try again."
UNAVAILABLE_MESSAGE = "The language model is unavailable. Try again shortly."
BAD_RESPONSE_MESSAGE = "The language model returned an unusable response."


@dataclass(frozen=True, slots=True)
class ApiLimits:
    max_upload_bytes: int
    max_question_chars: int


def get_engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


EngineDependency = Annotated[Engine, Depends(get_engine)]


async def read_limited_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"The file is larger than {limit} bytes.")
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > limit:
            raise HTTPException(413, f"The file is larger than {limit} bytes.")
    return bytes(body)


def create_app(build_engine: Callable[[], Engine], limits: ApiLimits) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.engine = await run_in_threadpool(build_engine)
        yield

    app = FastAPI(
        title="Investigation Engine",
        summary="Search and ask questions across many documents, with cited answers.",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get(
        "/api/health",
        response_model=HealthOut,
        summary="Service status",
        description="Counts of stored documents and chunks, and whether answers are available.",
    )
    def health(engine: EngineDependency) -> HealthOut:
        stats = engine.stats()
        return HealthOut(
            status="ok",
            documents=stats.documents,
            chunks=stats.chunks,
            can_answer=engine.can_answer,
        )

    @app.get(
        "/api/documents",
        response_model=DocumentPageOut,
        summary="List documents",
        description="Newest first. Filter with q, which matches the title or the file name.",
    )
    def list_documents(
        engine: EngineDependency,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        offset: Annotated[int, Query(ge=0)] = 0,
        q: Annotated[str, Query(max_length=MAX_FILENAME_LENGTH)] = "",
    ) -> DocumentPageOut:
        return page_out(engine.documents(limit, offset, q))

    @app.post(
        "/api/documents",
        status_code=201,
        response_model=UploadOut,
        summary="Upload a document",
        description=(
            "Send the raw file as the request body and its name in the filename query "
            "parameter. Supported: txt, md, html, json, jsonl and csv."
        ),
    )
    async def upload_document(
        request: Request,
        engine: EngineDependency,
        filename: Annotated[str, Query(min_length=1, max_length=MAX_FILENAME_LENGTH)],
    ) -> UploadOut:
        content = await read_limited_body(request, limits.max_upload_bytes)
        try:
            result = await run_in_threadpool(engine.upload, filename, content)
        except UnsupportedFormatError as error:
            raise HTTPException(415, str(error)) from error
        except (ParseError, NothingToIndexError) as error:
            raise HTTPException(422, str(error)) from error
        return upload_out(result)

    @app.delete(
        "/api/documents/{document_id:path}",
        status_code=204,
        summary="Delete a document",
        description="Removes the document and the chunks that belong only to it.",
    )
    def delete_document(document_id: str, engine: EngineDependency) -> Response:
        if not engine.delete_document(document_id):
            raise HTTPException(404, f"No document with id {document_id}.")
        return Response(status_code=204)

    @app.post(
        "/api/search",
        response_model=SearchOut,
        summary="Search without a language model",
        description="Hybrid keyword and vector search that returns the best matching chunks.",
    )
    def search(body: SearchIn, engine: EngineDependency) -> SearchOut:
        hits = engine.search(body.query, body.limit)
        sources = engine.sources_of_many([hit.chunk.id for hit in hits])
        return SearchOut(hits=[hit_out(hit, sources.get(hit.chunk.id, [])) for hit in hits])

    @app.post(
        "/api/ask",
        response_model=AnswerOut,
        summary="Ask a question",
        description=(
            "The model searches the documents, then answers with citations, reports "
            "conflicting sources, or says the evidence is insufficient."
        ),
    )
    def ask(body: AskIn, engine: EngineDependency) -> AnswerOut:
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "The question is empty.")
        if len(question) > limits.max_question_chars:
            raise HTTPException(
                422, f"The question is longer than {limits.max_question_chars} characters."
            )
        started = time.perf_counter()
        try:
            investigation = engine.ask(question)
        except ConfigurationError as error:
            raise HTTPException(503, str(error)) from error
        except LlmRateLimited as error:
            logger.warning("language model rate limited: %s", error)
            raise HTTPException(429, RATE_LIMITED_MESSAGE) from error
        except LlmUnavailable as error:
            logger.warning("language model unavailable: %s", error)
            raise HTTPException(503, UNAVAILABLE_MESSAGE) from error
        except LlmError as error:
            logger.warning("language model failed: %s", error)
            raise HTTPException(502, BAD_RESPONSE_MESSAGE) from error
        elapsed_ms = round((time.perf_counter() - started) * MILLISECONDS)
        return answer_out(investigation, elapsed_ms)

    return app

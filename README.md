---
title: Investigation Engine
sdk: docker
app_port: 7860
pinned: false
---

# Investigation Engine

Ask questions across tens of thousands of documents and get answers that cite their sources. The engine searches by meaning and by keyword, follows a question across several documents, and says so when the evidence is missing or the sources disagree.

- **Live demo:** `<add the public URL after deploying>`
- **Evaluation:** [reports/evaluation-report.md](reports/evaluation-report.md)
- **AI usage:** [reports/ai-usage-report.md](reports/ai-usage-report.md)
- **Interview notes:** [docs/walkthrough.md](docs/walkthrough.md)

The front matter at the top of this file is required by Hugging Face Spaces and can be ignored elsewhere.

## What it does

| Area | What is built |
| --- | --- |
| Ingestion | Parses `txt`, `md`, `html`, `json`, `jsonl` and `csv` with their metadata. Two chunkers (sentence-aware and fixed window) were compared. Duplicate text is stored once and credited to every document that contains it. Documents can be added, updated (detected by content hash) and deleted, and the indexes follow. |
| Search | Postgres with pgvector (HNSW, cosine) for meaning, a hand-written BM25 index for keywords, and a hand-written reciprocal rank fusion to combine them. An optional hand-written re-ranker is available. |
| Answers | A hand-written agent loop with two tools: `search` and `submit_answer`. It searches several times, cites evidence ids, reports conflicts between sources, and abstains when the evidence is not enough. Model failures are retried, cached and fall back from Gemini to Groq. |
| Evaluation | 56 questions (simple, multi-step, multi-document, unanswerable, conflicting), Recall@K, full recall, MRR, answer accuracy, F1, citation precision and recall, and a hallucination rate. Retrieval configurations and chunkers are compared on the full corpus. |
| Product | FastAPI backend and a single-page web interface for uploading, listing, deleting, searching and asking. |
| Data | 66,607 documents (66,581 HotpotQA paragraphs plus 26 synthetic incident documents) that become 81,243 chunks. |

## How a question is answered

```mermaid
flowchart LR
    Q[Question] --> A[Agent loop]
    A -- search --> R[Hybrid retriever]
    R --> V[pgvector HNSW]
    R --> B[BM25 index]
    V --> F[Rank fusion]
    B --> F
    F --> E[Numbered evidence E1, E2, ...]
    E --> A
    A -- submit_answer --> G[Guards]
    G --> O[Answer, citations, conflicts or abstention]
```

1. The model may call `search` up to five times with focused queries. Each call returns numbered evidence items.
2. The model finishes with `submit_answer`, naming the evidence ids it relied on.
3. The code, not the model, enforces the rules: an answer needs a citation, citations must be ids that were actually returned, a conflict needs two evidence ids, and a plain text reply is refused. When the step limit or the invalid-submission limit is reached the result is a reported abstention.
4. Evidence is passed to the model as quoted data, and the system prompt tells it to ignore instructions found inside documents.

## Deployment architecture

```mermaid
flowchart LR
    U[Browser] --> S[Hugging Face Space<br/>Docker, FastAPI, ONNX embeddings, BM25 in memory]
    S --> N[(Neon Postgres<br/>pgvector: documents, chunks, vectors)]
    S --> L[Gemini API, then Groq as fallback]
```

- The Space holds no data. Everything that must survive a restart lives in Postgres.
- At startup the Space creates any missing table, embeds any chunk that has no vector yet, and rebuilds the BM25 index from the stored chunks (about 5 s for 81,243 chunks on a laptop).
- The embedding model is baked into the image, so a cold start does not download it.
- Secrets (`DATABASE_URL`, `GEMINI_API_KEY`, `GROQ_API_KEY`) are Space secrets. They are read through typed settings, held as `SecretStr` and never printed. `.env` is git-ignored.

## Run it locally

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync
docker compose up -d db
```

Create a local `.env` (it is git-ignored). Only `DATABASE_URL` is required; add a key to enable answers:

```bash
DATABASE_URL=postgresql://investigation:investigation@127.0.0.1:5433/investigation
GEMINI_API_KEY=your-key
GROQ_API_KEY=your-key
```

Load the sample incident documents and start the server:

```bash
uv run python -m investigation_engine.app.seed --samples samples/incidents
uv run uvicorn investigation_engine.app.main:create_default_app --factory --port 8000
```

Open <http://127.0.0.1:8000>. Without a language model key the page still uploads, lists, deletes and searches; the Ask button reports that no model is configured.

Use `127.0.0.1` rather than `localhost` on Windows: `localhost` first tries IPv6 and can wait for a timeout.

### Load the full 66,607 document corpus

```bash
uv run python -m investigation_engine.evaluation.cli download
uv run python -m investigation_engine.app.seed --samples samples/incidents --hotpot data/hotpot/validation.jsonl
```

Embedding 81,243 chunks on a laptop CPU takes about 45 to 60 minutes the first time, at roughly 30 passages per second. Vectors are cached in `data/embeddings.sqlite`, so a run with a warm cache takes about 25 minutes (roughly 10 for the database writes and 15 for reading the cache). The seed command is safe to repeat and safe to interrupt: it skips unchanged documents and embeds only the chunks that still have no vector.

## Configuration

Every setting is read from the environment or `.env` through `pydantic-settings`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | none (required) | Postgres with the pgvector extension |
| `GEMINI_API_KEY` | none | Enables Gemini |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model name. Check the current list at ai.google.dev/gemini-api/docs/models; `gemini-3.5-flash-lite` is a lighter alternative |
| `GROQ_API_KEY` | none | Enables Groq, used as the fallback |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GEMINI_BASE_URL`, `GROQ_BASE_URL` | the providers' OpenAI-compatible endpoints | Override for testing |
| `LLM_TIMEOUT_SECONDS` | `30` | Per model request |
| `LLM_CACHE_PATH` | `data/llm_cache.sqlite` | Cache of model responses |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local ONNX embedding model |
| `EMBEDDING_DIMENSION` | `384` | Must match the model and the database |
| `RETRIEVAL_CANDIDATES` | `50` | Candidates taken from each search before fusion |
| `RERANK_RESULTS` | `false` | Turns on the re-ranker (see the evaluation report) |
| `CHUNK_MAX_WORDS` | `120` | Chunk size for uploads |
| `PROTECT_BUILT_IN_DOCUMENTS` | `true` | Only files uploaded through the page can be replaced or deleted. Documents loaded by the seed command are read-only (HTTP 403) |
| `MAX_UPLOAD_BYTES` | `5000000` | Upload size limit |
| `MAX_QUESTION_CHARS` | `1000` | Question length limit |
| `FASTEMBED_CACHE_PATH` | system temp folder | Where the embedding model is stored |

## HTTP API

Interactive documentation is served at `/docs`.

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Counts, and whether answers are available |
| `GET /api/documents?limit=&offset=&q=` | Newest first, filtered by title or file name |
| `POST /api/documents?filename=name.md` | Upload a file as the raw request body. Returns 415 for an unsupported type, 422 for an unreadable or empty file, 413 when it is too large |
| `DELETE /api/documents/{id}` | Delete a document and the chunks only it owns |
| `POST /api/search` | `{"query", "limit"}`: ranked chunks with scores, ranks and sources, no model needed |
| `POST /api/ask` | `{"question"}`: status, answer, citations, conflicts, the searches made, token use and time. Returns 429, 502 or 503 for model failures |

Files are sent as the raw request body, which avoids the multipart dependency.

## Tests and quality gates

```bash
docker compose up -d db
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run python tools/check_no_comments.py
uv run pytest -q
```

- Tests that need Postgres run against the `investigation_test` database that `docker compose` creates, and are skipped with a clear message if it is not running.
- The storage and vector index tests run the same contract against the in-memory and the Postgres implementation.
- `tools/check_no_comments.py` enforces the project rule that Python has no comments and no docstrings. Names, types and tests carry the meaning; design reasons live in this README and in `docs/`.

## Project layout

```
src/investigation_engine/
  domain.py          Document, Chunk, SourceRef
  settings.py        typed settings
  embeddings.py      local ONNX embeddings and a cache
  ingestion/         parsers, chunkers, dedup, ingestion service
  storage/           in-memory and Postgres stores, database connection
  retrieval/         BM25, vector indexes, rank fusion, re-ranker, hybrid retriever
  llm/               OpenAI-compatible client, retries, cache, fallback
  agent/             tools, evidence board, prompts, investigator loop
  app/               engine, HTTP API, web page, seed command
  evaluation/        metrics, HotpotQA loader, question sets, evaluation commands
samples/incidents/   13 synthetic files in six formats
eval/                56 evaluation questions
reports/             evaluation results and analysis
```

## Evaluation commands

```bash
uv run python -m investigation_engine.evaluation.cli retrieval --samples samples/incidents --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl --out reports/retrieval.json
uv run python -m investigation_engine.evaluation.cli answers --provider gemini --samples samples/incidents --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl --max-documents 5000 --pause 4 --out reports/answers.json
```

The `answers` command needs a model key. It prints each question as it finishes, and because model responses are cached, an interrupted run resumes without repeating requests.

## Deploy

1. **Database.** Create a free [Neon](https://neon.tech) project and copy its connection string (with `?sslmode=require`).
2. **Load the data once from your machine.** Seed a local database, then copy it to Neon:

   ```bash
   docker exec investigation-db sh -c "pg_dump -Fc -U investigation investigation_full > /tmp/full.dump"
   docker exec investigation-db pg_restore --no-owner --no-acl -d "NEON_CONNECTION_STRING" /tmp/full.dump
   ```

   The dump is about 160 MB and the full corpus needs about 280 MB, which fits the 512 MB free tier. HNSW is rebuilt by the restore (26 s locally). In Git Bash on Windows, prefix the commands with `MSYS_NO_PATHCONV=1` so that `/tmp/full.dump` is not rewritten as a Windows path.
3. **Space.** Create a Hugging Face Space with the Docker SDK, add `DATABASE_URL`, `GEMINI_API_KEY` and `GROQ_API_KEY` as Secrets, and push this repository to it:

   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
   git push space main
   ```

4. Open the Space URL and confirm `/api/health` reports the document count.

Free Spaces sleep after a period without traffic and wake on the next request. Neon suspends an idle database the same way; the first query after a pause reconnects automatically.

## Known limits

- The answer-quality evaluation depends on the language model and on free-tier rate limits. See the evaluation report for what was measured and what was not.
- One server process holds the BM25 index in memory, so the service runs a single worker. Searches take a lock on the keyword index, so concurrent searches are serialised.
- Anyone who can reach the page can upload files and delete the files that were uploaded through it. The documents loaded by the seed command are protected from changes, but there is no authentication and no per-user separation. A real deployment would add both.
- The embedding model is English. The tokenizer handles other scripts, but retrieval quality was only measured on English text.

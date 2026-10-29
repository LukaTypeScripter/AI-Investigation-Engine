# Investigation Engine

Ask questions across tens of thousands of documents and get answers that cite their sources. The engine searches by meaning and by keyword, follows a question across several documents, and says so when the evidence is missing or the sources disagree.

- **Live demo:** `<add the public URL after deploying>`
- **Evaluation:** [reports/evaluation-report.md](reports/evaluation-report.md)
- **AI usage:** [reports/ai-usage-report.md](reports/ai-usage-report.md)
- **Interview notes:** [docs/walkthrough.md](docs/walkthrough.md)

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
    U[Browser] --> S[Render web service, free plan<br/>Docker, FastAPI, ONNX embeddings, BM25 in memory]
    S --> N[(Neon Postgres<br/>pgvector: documents, chunks, vectors)]
    S --> L[Groq API, or Gemini first with Groq as fallback]
```

- The web service holds no data. Everything that must survive a restart lives in Postgres.
- At startup the service creates any missing table, embeds any chunk that has no vector yet, and rebuilds the BM25 index from the stored chunks (about 5 s for 81,243 chunks on a laptop with a local database).
- The embedding model is baked into the image, so a cold start does not download it.
- The service is described in [render.yaml](render.yaml): Docker runtime, free plan, the Ohio region (next to a Neon project in AWS us-east-2) and a health check on `/api/health`.
- Secrets (`DATABASE_URL`, `GROQ_API_KEY`, and `GEMINI_API_KEY` if used) are Render environment variables. In `render.yaml` they are declared with `sync: false`, so their values are typed into the dashboard and never stored in the repository. They are read through typed settings, held as `SecretStr` and never printed. `.env` is git-ignored.

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
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name. `llama-3.3-70b-versatile` no longer exists for new keys (HTTP 404 on 2026-10-29). List the models your key can use with `GET https://api.groq.com/openai/v1/models` |
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
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

## Logs

The service writes one line per event to standard error, in the form `time level logger: message`. Nothing is written to a file, so the logs live wherever the process runs:

| Where it runs | Where to read the logs |
| --- | --- |
| `uvicorn` in a terminal | That terminal. Add `2>&1 \| tee server.log` to keep a copy |
| `docker compose up` | `docker compose logs -f app` |
| Render | The **Logs** tab of the service in the Render dashboard |
| `evaluation.cli answers` | The terminal. Each question prints one line, and a failed question prints its reason. Redirect with `2> answers.log` |

What is logged:

```
INFO    engine ready: documents=66607 chunks=81243 can_answer=True
INFO    uploaded notes.md: added=1 updated=0 unchanged=0 chunks_added=1 chunks_removed=0
WARNING upload of tool.exe rejected (415): unsupported file type '.exe', supported: ...
INFO    ask 'Which service was modified by change CHG-1041?': status=answered searches=1 llm_calls=2 tokens=1587 elapsed_ms=3989
WARNING attempt 1 of 4 failed, retrying in 30.0s: <model>: rate limited: ...
WARNING language model failed on '<question>': <model>: HTTP 400: ...
```

Questions are shortened to 80 characters. API keys are never logged; provider error messages can name the model and the provider account, so do not publish raw logs. Uvicorn's own access lines (`POST /api/ask ... 200 OK`) appear alongside.

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

1. **Database.** Create a free [Neon](https://neon.tech) project and copy its **direct** connection string: the host has no `-pooler` in it, and the string ends in `?sslmode=require`. The pooled string is meant for many short connections and can break prepared statements, which this app uses.
2. **Load the data once from your machine.** Seed a local database, then copy it to Neon. The connection string goes through an environment variable so that it stays out of the command line and the shell history:

   ```bash
   export NEON_URL="the direct connection string"
   docker exec investigation-db sh -c "pg_dump -Fc -U investigation investigation_full > /tmp/full.dump"
   docker exec -e NEON_URL -e PGOPTIONS="-c maintenance_work_mem=256MB" investigation-db sh -c 'pg_restore --no-owner --no-acl -d "$NEON_URL" /tmp/full.dump'
   ```

   Measured: the dump is about 150 MB, the restore took about 5 minutes (the HNSW index about 30 s of that), and the database takes 271 MB in Neon, which fits the 512 MB free tier. Neon's default `maintenance_work_mem` is 64 MB, too little for the vector index, so the restore raises it for its own session. In Git Bash on Windows, prefix the `docker exec` commands with `MSYS_NO_PATHCONV=1` so that `/tmp/full.dump` is not rewritten as a Windows path. Check the copy by comparing `select count(*) from documents` in both databases (66,607).
3. **Render.** Sign in to [Render](https://render.com) with GitHub, choose **New, Blueprint**, select this repository and apply it. `render.yaml` sets the Docker runtime, the `free` plan, the Ohio region, the health check and `PORT=7860`. Render then asks for the two values marked `sync: false`: `DATABASE_URL` (the direct Neon string) and `GROQ_API_KEY`. Pick the Render region closest to your Neon region and change `region` in `render.yaml` if it differs. Add `GEMINI_API_KEY` only if that key really works: a key the provider rejects (for example a project without a free tier) makes every question fail, because the fallback only covers temporary errors.
4. The first deploy builds the image, which takes several minutes. Open the service URL and confirm `/api/health` reports the document count.

Hugging Face Docker Spaces were the first choice but need a paid plan, so they are not used.

**Free plan limits.** At the time of writing, a Render free web service has 512 MB of memory and 0.1 CPU, and it spins down after 15 minutes without traffic; check Render's current terms. The container was run under the same limits (`docker run --memory 512m --memory-swap 512m --cpus 0.1`) against Neon: it was ready after 144 s, and memory peaked at 440 MiB of 512 when the first search loaded the embedding model. That first search took 77 s and later searches took 11 to 17 s. The container was not killed for running out of memory. The image sets `MALLOC_ARENA_MAX=2`, which stops the memory allocator from keeping a separate arena per thread; without it the same test peaked at 486 MiB (a margin of about 5%) and the first search took 140 s. Each setting was measured once, so treat the numbers as indicative. After a sleep, the next visitor pays the whole startup again, so expect a wait of several minutes. Neon suspends an idle database in the same way; the first query after a pause reconnects automatically.

For comparison, from a laptop with full CPU the service took 101 s to start (the keyword index is rebuilt by streaming all 81,243 chunks over the internet; it takes about 5 s from a local database), and a search took 1.4 to 1.6 s.

## Known limits

- The answer-quality evaluation depends on the language model and on free-tier rate limits. See the evaluation report for what was measured and what was not.
- One server process holds the BM25 index in memory, so the service runs a single worker. Searches take a lock on the keyword index, so concurrent searches are serialised.
- Anyone who can reach the page can upload files and delete the files that were uploaded through it. The documents loaded by the seed command are protected from changes, but there is no authentication and no per-user separation. A real deployment would add both.
- The embedding model is English. The tokenizer handles other scripts, but retrieval quality was only measured on English text.

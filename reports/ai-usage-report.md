# AI Usage Report

The task allows AI tools and makes the candidate fully responsible for the code. This report says which tools were used, what they did, how their output was checked, and what the candidate decided and changed personally.

**Sections marked "Candidate to complete" can only be written by the candidate.** They are left open on purpose. A report that credits the candidate with work they did not do would be false, and the interview would expose it.

## 1. Tools used to build the project

| Tool | Used for |
| --- | --- |
| Claude Code (Anthropic's command-line coding agent), with Claude models | Planning, writing the code and tests, running the quality gates, analysing evaluation results, writing the documentation |
| Plugins and skills loaded into Claude Code (listed in section 4) | Style rules, testing discipline, Python tooling guidance |
| A browser automation tool (Playwright) driven by Claude Code | Trying the web page in a real browser: upload, search, delete. Also filling in the Render blueprint form and checking the deploy. The candidate signed in and typed the two secrets; the AI never saw them |
| Docker, Postgres with pgvector, `uv`, `ruff`, `ty`, `pytest` | Ordinary development tooling, not AI |

Models that run inside the product, as opposed to building it:

| Model | Role |
| --- | --- |
| `BAAI/bge-small-en-v1.5` through FastEmbed (ONNX, local) | Embeddings for semantic search |
| Gemini (default `gemini-3.8-flash`) with Groq (default `openai/gpt-oss-120b`) as a fallback | Language models for the agent. Both are configured by environment variables. The first Groq default, `llama-3.3-70b-versatile`, was read from documentation and returned HTTP 404 on the first real request (2026-10-29), so the default was changed to a model that the key's own model list returned. Model availability changes, so list the models before running |

No paid AI service is needed to run the project.

## 2. How the work was done

The candidate directed the work in a terminal session and approved each step ("lets plan together", then one approval per feature). Claude Code wrote the code. The working rules, set by the candidate at the start, were:

- Prefer a design the candidate can fully explain over a clever one. No heavyweight RAG framework; the retrieval, ranking, agent loop and evaluation are hand-written.
- No comments and no docstrings in Python. A test fails the build if one appears.
- Write the failing test first, then the code, then run the full gate (`ruff`, `ruff format`, `ty`, the comment checker, `pytest`).
- One Conventional Commit per feature, containing only that feature and its tests. No AI attribution lines in commits. AI use is disclosed here instead.
- Ask before adding a runtime dependency. The candidate approved the list (FastEmbed, httpx, pydantic, FastAPI, uvicorn, psycopg, pgvector). `numpy` was added without a separate approval because FastEmbed requires it, and this was reported at the time.

## 3. What the AI did, by area

| Area | What Claude Code produced |
| --- | --- |
| Ingestion | Parsers for six formats, two chunkers, duplicate handling with shared ownership, add, update and delete with index changes |
| Search | BM25 index, reciprocal rank fusion and a feature re-ranker (all hand-written), vector search on pgvector, the hybrid retriever |
| Storage | In-memory and Postgres stores sharing one contract test suite, the connection layer with reconnection, the pgvector index |
| LLM layer | OpenAI-compatible client, retry with backoff, response cache, provider fallback, error types |
| Agent | Two-tool loop, evidence board, the checks that refuse unsupported answers, system prompt |
| Product | HTTP API, web page, seed command, container image |
| Evaluation | Metrics, HotpotQA download and sampling, 26 hand-written incident questions and 13 synthetic documents, retrieval and answer evaluation commands, failure analysis |
| Optimisation | Pruned BM25 search, 16-bit vectors, batched queries, each measured before and after |
| Documentation | README, this report, the evaluation report, the interview walkthrough |

## 4. Third-party material used

Ten agent skills from the open-source repository `wshobson/agents`, commit `0874a27`, MIT licence, Copyright 2024 Seth Hobson, were installed locally to guide the work: `python-testing-patterns`, `python-error-handling`, `python-type-safety`, `python-project-structure`, `python-configuration`, `python-anti-patterns`, `rag-implementation`, `hybrid-search-implementation`, `llm-evaluation`, `embedding-strategies`. They are not part of this repository (they are git-ignored). Their ideas were used, their habits were not: where they show docstrings, comments, `pip` or LangChain, this project follows its own rules instead.

Other tools loaded into the session: the Astral plugin for `uv`, `ruff` and `ty` guidance (`astral-sh/claude-code-plugins`), the "superpowers" plugin (test-driven development and planning disciplines), a memory plugin, and a personal prompt-coaching skill that corrects the candidate's English and prompts. None of them contributes code to the repository.

Data: HotpotQA (validation split, distractor configuration, via the Hugging Face dataset API). The incident documents and questions are original and fictional.

## 5. How the output was checked

Claude Code's output was not trusted on its own. Checks that were run:

- **Tests first.** Each feature began with a failing test. Usually the first failure was a missing module, and the implementation was then written in one step. The one clear lapse was the failure-analysis module, which was written together with its tests; a mutation check followed to prove the tests can fail.
- **Mutation checks.** Deliberately breaking the delete logic of the Postgres store failed 10 tests. Removing the keyword-index lock failed 2. Removing the engine's write lock failed 1. Making the search pruning unsound failed 3 tests.
- **Contract tests.** The in-memory and Postgres stores run the same test suite, so the fast fake cannot drift from the real thing.
- **Measurement before and after.** The BM25 optimisation was compared on 355 real queries; every ranking was identical.
- **A real browser.** The page was exercised in a browser, which found a missing favicon, a "1 chunks" wording error and a confirmation window too short to use.
- **A fresh clone.** The full gate was run on a clean clone of the repository earlier in the project. It found a stale lint cache that the working copy had hidden.
- **The full corpus.** The deployed stack was run on all 81,243 chunks, inside a container limited to 1 GiB of memory.

## 6. Mistakes the AI made and how they were found

These matter for the interview because the candidate may be asked about them.

| Mistake | How it was found | Fix |
| --- | --- | --- |
| The identifier and fields of CSV and JSON rows were stored only as metadata, so a search for `CHG-1041` could not find its own row | The evaluation's failure analysis on the full corpus | Index the fields in the text, with tests. Recall on the incident questions rose from 0.770 to 0.841 |
| The dataset download threw away its progress on an HTTP 429 and hid the error behind a pipe | The first real download | A resumable partial file, `Retry-After` handling and pacing, tested first |
| Pragmas (`noqa`, `ty: ignore`, `fmt: skip`) were written repeatedly against the project's rule | The comment checker and review | Removed each time and the cause fixed instead. This kept happening, so every change is checked for them |
| A test helper used `keywords or Bm25Index()`, and an empty index is falsy, so the test silently used a different index | A failing test | Explicit `is None` checks |
| Tests connected to `localhost`, which on this Windows machine waits 20 seconds for an IPv6 timeout | A suite that took two minutes for nine tests | `127.0.0.1` |
| The tests and the app shared one database and the tests wipe theirs | Review before the container was added | A separate test database |
| The default Groq model name was taken from documentation | The first real request returned HTTP 404 | The default now comes from the model list that the key itself returned |
| Hugging Face Docker Spaces were assumed to be free | Creating the Space: the Docker option was disabled without a paid plan | Render for the service and Neon for the database |
| An old server was "stopped" with a pattern that missed it, and kept answering | A health check that looked fine but proved nothing, because Neon and the local database held identical data | All server processes were stopped, and the target was then proved with a document that exists only in Neon |
| The three failed evaluation questions were guessed to be rate limits | Reading the results file: they were HTTP 400 broken tool calls from the model | The client retries that error type |
| The answer scorer compared a non-breaking hyphen and curly quotes with plain ASCII and failed | One correct answer (`inc-14`) marked wrong | The normaliser maps them to ASCII. Accuracy rose from 0.821 to 0.839 |

## 7. What the AI cannot vouch for

- Answer quality was measured once, for one configuration and one model (evaluation report, section 6). Prompt-injection resistance is still untested with a real model, and a second configuration could not be run because the free daily token allowance ran out.
- Provider details were learned only by using the providers: the first Groq default was retired (HTTP 404), Gemini refused the key because the project has no free tier, and the two Groq models used allow 200,000 tokens a day and 8,000 tokens a minute respectively. These limits were read from error messages and can change.
- The hosted deployment lives in the candidate's own Render and Neon accounts. The candidate signed in and typed the secrets, and the AI prepared the files, drove the browser form and checked the result. The memory test of the free plan was a simulation (512 MB, 0.1 CPU, in Docker), not a measurement on Render itself.

## 8. Candidate to complete

### 8.1 Decisions I made myself

Facts from the session, which the candidate can confirm and extend:

- Chose the goal of a design I can explain over a fancy one.
- Set the no-comments rule and the Conventional Commits rule, and required that commits carry no AI attribution.
- Chose free AI only, which led to Gemini and Groq with local embeddings.
- Approved the runtime dependencies before they were added.

Add the decisions below that are truly yours, in your own words:

- [ ] Why I accepted hybrid search without the re-ranker as the default:
- [ ] Why I chose Render and Neon for hosting (Hugging Face Docker Spaces turned out to need a paid plan):
- [ ] Any design change I asked for that is not in this list:

### 8.2 What I changed or wrote myself

Write only what you actually did, such as lines you edited by hand, tests you added, bugs you found, or prompts that you rewrote because the first result was wrong.

- [ ]
- [ ]
- [ ]

### 8.3 What I verified myself

Examples to keep only if true: read every module in the walkthrough, ran the tests locally, ran the system on my own documents, reproduced a number from the evaluation report.

- [ ]
- [ ]

### 8.4 What I learned and would do differently

- [ ]
- [ ]

# Interview walkthrough

A guide for explaining this project aloud, with no AI help. Read the code in the order below. Every function named here exists; open it and read it before the interview.

## The two-minute version

"It is an investigation engine over many documents. Files are parsed, split into chunks and stored in Postgres. A question goes to an agent, which is a loop that lets a language model call two tools: `search` and `submit_answer`. `search` runs hybrid retrieval, which combines vector search in pgvector with a BM25 keyword index I wrote, merged by reciprocal rank fusion. Results come back as numbered evidence, E1, E2, and so on. The model must finish by submitting an answer with evidence ids. My code, not the model, checks that the ids are real, that an answer cites something, and that a claimed conflict names two pieces of evidence. When the evidence is not enough the answer is a refusal, and when sources disagree it is a conflict report. I measured retrieval on 66,607 documents and 51 scored questions, and found and fixed a real indexing defect through the failure analysis."

## Reading order

| Step | File | What to be able to say |
| --- | --- | --- |
| 1 | `domain.py` | Three small records. A `Chunk` id is `document id:index`. A `SourceRef` names a document |
| 2 | `ingestion/parsers.py` | One function per format, chosen by file extension. Structured rows add their id and fields to the text so they can be searched |
| 3 | `ingestion/chunking.py` | Two strategies: pack whole sentences up to a word limit, or slide a fixed window with overlap |
| 4 | `ingestion/dedup.py` and `ingestion/service.py` | Fingerprints, shared ownership, the change log |
| 5 | `storage/memory.py` and `storage/postgres.py` | The same behaviour twice. `tests/test_ingestion_service.py` runs on both |
| 6 | `retrieval/bm25.py`, `fusion.py`, `rerank.py`, `hybrid.py` | The hand-written ranking code |
| 7 | `retrieval/indexing.py` and `postgres_vector.py` | How chunks reach the two indexes, and the repair path |
| 8 | `llm/` | Client, retries, cache, fallback, errors |
| 9 | `agent/` | Tools, evidence board, the guards, the loop |
| 10 | `app/engine.py`, `api.py`, `main.py` | Locks, HTTP errors, startup |
| 11 | `evaluation/` | Metrics, the commands, the failure analysis |

## The pipeline, step by step

### Adding a document

1. `parse_bytes` picks a parser from the extension and returns `Document` records. CSV, JSON and JSONL give one document per row.
2. `IngestionService.ingest` hashes each document (title, text, metadata). Same hash as stored: unchanged, skip. Different hash: delete the old version, then add the new one.
3. The chunker splits the text. Each chunk gets a fingerprint: the text with whitespace collapsed and lower-cased, then SHA-256.
4. A chunk whose fingerprint is already stored is not stored again. The document is recorded as another owner of the existing chunk. This is why the incident report that exists twice appears once in the index and is cited with both file names.
5. `save` writes the document, the new chunks and the ownership rows in one transaction.
6. The service returns an `IndexChanges` list: chunks to add and chunk ids to remove. `SearchIndexer.apply` embeds the new chunks in batches and writes them to the vector index and the keyword index.

### Deleting a document

The store removes the document's ownership rows. A chunk with no owner left is deleted. A chunk that still has owners but was "homed" in the deleted document is renamed to belong to the first remaining owner (`b:0` instead of `a:0`). That rename is reported as one removal plus one addition, so both indexes follow.

### Answering a question

1. `Investigator.investigate` sends the system prompt and the question to the model with two tool definitions.
2. If the model calls `search`, the retriever runs and the results are rendered as `[E1] "title" (sources: ...)` followed by the text. The same chunk keeps the same id for the whole run.
3. If the model calls `submit_answer`, `parse_submission` validates it. A rejected submission is returned to the model as an error message and counts toward a limit of two.
4. A plain text reply is not accepted: the loop sends a reminder. After six steps, or two invalid submissions, the result is an abstention that says why.
5. The final `Investigation` carries the status, the answer, the citations with every source document, the conflicts, the searches made and the token count.

### One search

`HybridRetriever.retrieve`: embed the query, ask pgvector for the 50 nearest chunks, ask BM25 for its best 50, fuse the two rankings, optionally re-rank, return the top few.

## The algorithms

### BM25 (`retrieval/bm25.py`)

For each query term `t` that a chunk `d` contains:

```
score += idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * length(d) / average_length))
idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
```

`f` is how often the term occurs in the chunk, `df` is how many chunks contain it, `N` is the number of chunks. `k1 = 1.5` limits how much repeating a word helps. `b = 0.75` makes long chunks pay for being long. Rare words matter more because `idf` is larger.

The index is an inverted index: a dictionary from term to `{chunk id: count}`. Adding and removing a chunk touches only its own terms, so updates are cheap.

**The speed-up.** A question such as "which service was modified by the change that..." contains very common words with huge posting lists. Terms are processed from highest `idf` to lowest. A term can add at most `idf * (k1 + 1)` to any score. If the k-th best score so far is already larger than the sum of that bound over the remaining terms, no chunk outside the current candidates can reach the top k. From then on only candidates are updated. The answer is exactly the same, which a test against a plain implementation and a comparison on 355 real queries confirm. Mean query time went from 167 ms to 52 ms.

### Reciprocal rank fusion (`retrieval/fusion.py`)

```
score(chunk) = sum over rankings of 1 / (60 + rank)
```

A chunk ranked 3rd by vectors and 2nd by keywords scores `1/63 + 1/62 = 0.03200`. Fusion uses ranks, not scores, because a cosine similarity and a BM25 score are on different scales and cannot be added. The constant 60 stops the first place from dominating.

### The re-ranker (`retrieval/rerank.py`)

`1.0 * fused + 0.5 * coverage + 0.5 * title_match`, where `fused` is the fusion score divided by the best one, `coverage` is the share of the question's non-trivial words found in the chunk or its title, and `title_match` is the share of the title's words found in the question. It improves MRR and hurts multi-hop recall, so it is off by default. Be ready to say why: a second hop document shares few words with the question.

### Duplicate handling (`ingestion/dedup.py`)

Whole-chunk fingerprints, not near-duplicate detection. "Same text twice" is handled exactly. A near copy, such as a report with one sentence changed, is stored as a separate chunk. That is a limit, not an accident.

### HNSW (`storage/database.py`)

Postgres builds a layered graph over the vectors. A search walks the graph instead of comparing with all 81,243 vectors, and takes about 5 ms. It is approximate, which is why the service scores 0.876 recall@5 against 0.886 for exact search. `hnsw.ef_search` is set to 100 so that asking for 50 neighbours really returns 50 (the default of 40 would cap it).

## Decisions and the reasons

| Decision | Why | What I gave up |
| --- | --- | --- |
| Hand-written agent loop, no framework | I can explain every line, and the limits and checks are in my code | Less ready-made functionality |
| Two tools only | Search and submit are enough for multi-step questions, and fewer tools mean fewer ways to fail | The model cannot fetch a whole document |
| Guards in code, not in the prompt | A prompt is a request, code is a rule | A little more code |
| Postgres with pgvector | Needs one database for everything and gives the vector search the task requires | Not as fast as a dedicated vector store |
| BM25 in memory, rebuilt at startup | Simple, fast, exact, and it is the hand-written ranking component | Memory use (199 MB for 81k chunks), and one worker only |
| 16-bit vectors | The database drops from 403 MB to 279 MB and fits the free tier | A tiny rounding error |
| Local ONNX embeddings | Free, private, no rate limit | An English-only model |
| Responses cached by request hash | Repeated evaluation runs cost nothing | Stale answers if the prompt changes without changing the request. The key includes the whole request, so a prompt change is a different key |
| One locked database connection | Easy to reason about and fine for this load | Database calls are serialised. A pool would be the next step |
| Raw file upload, not multipart | No extra dependency | One file per request |

## Questions to be ready for

**Why hybrid and not just vectors?** On HotpotQA vectors win (recall@5 0.933 against 0.733). On the incident documents keyword search wins (0.929 against 0.730) because the questions contain identifiers like `CHG-1041`. Hybrid is never worst and is best overall.

**Why did the re-ranker hurt?** It rewards word overlap with the question. In a bridge question the second document is the one that does not overlap.

**How do you know the optimisation did not change results?** A test compares the search with a plain BM25 computation on 300 synthetic documents for many queries and limits. A script compared 355 real queries before and after. A deliberately broken pruning rule fails three tests.

**What happens if the model returns rubbish?** The client retries transient errors (429, 5xx) with exponential backoff and honours `Retry-After`. A rejected request is not retried. A malformed response is an error. With two providers configured, a failure on the first falls through to the second. The API turns the remaining failures into 429, 502 or 503 without exposing the provider's message.

**What happens if the process dies between the database write and the index write?** Chunks exist without vectors. At the next startup `index_missing_vectors` finds them and embeds them. The keyword index is always rebuilt from the stored chunks.

**What prevents a document from hijacking the model?** The system prompt says evidence is data, not instructions. More importantly, the code accepts only citations of ids it returned, so a document cannot make the model cite something that was never retrieved. This is tested with a scripted model only, not with a real one.

**Where does it fail?** Bridge questions where the second document is not named. Duplicate documents crowding out the one that disagrees. Date formats (`5 March` against `2025-03-05`). Vocabulary differences ("delay" against "degraded").

**Why is the evaluation incomplete?** The answer-quality run needs a language model key. The retrieval half is complete and the command for the other half is written and tested.

## Changes you might be asked to make live

| Request | Where | Sketch |
| --- | --- | --- |
| Support PDF | `ingestion/parsers.py` | Add `.pdf` to `PARSERS` with a function that returns a `Document` per file or per page. Needs a PDF library, so ask first |
| Add a document type filter to search | `retrieval/hybrid.py`, `api.py` | Pass a filter into `retrieve`, drop chunks whose metadata does not match before fusing |
| Count duplicate owners as one rank slot | `retrieval/hybrid.py` | After fusion, skip chunks whose sources are all already represented |
| Limit questions per minute | `app/api.py` | A small sliding-window counter keyed by client address, returning 429 |
| Change the fusion constant | `retrieval/fusion.py` | `DEFAULT_RANK_CONSTANT` |
| Normalise dates | `ingestion/parsers.py` | Append `2025-03-05` for every `5 March 2025` found in a record |
| Show how many tokens an answer used | Already returned as `usage` in `/api/ask` | Add it to the page |

## Bugs you could be asked to find

Each of these was a real bug in this project. Be able to describe it.

- Rows from CSV and JSON files were searchable only by text, so their identifiers were invisible. Fixed in `ingestion/parsers.py` (`searchable_fields`).
- `Bm25Index` is falsy when empty because it defines `__len__`. `index or default` silently replaces an empty index. Use `is None`.
- On Windows `localhost` waits for an IPv6 connection to time out. Use `127.0.0.1`.
- `ef_search` below the number of neighbours asked for silently returns fewer results.

## Honest limits to state first

- Answer quality has not been measured with a real model yet.
- The deployment needs the candidate's own accounts.
- The evaluation set is small. Differences below about 0.04 are noise.
- Uploads and deletions are open to anyone who has the URL.

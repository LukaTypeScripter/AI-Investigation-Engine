# Evaluation Report

This report covers what was measured, what the numbers say, why the system fails where it fails, and what to change next. The detailed retrieval analysis with the per-question failure lists is in [retrieval-analysis.md](retrieval-analysis.md); raw result files are in this folder.

**Status.** Retrieval, chunking, storage and speed are measured on the full corpus. Answer quality is measured for one configuration (the agent over hybrid retrieval, 56 questions, Groq `openai/gpt-oss-120b`). A second answer configuration could not be run because the model's free daily token allowance was used up. Section 6 says exactly what was and was not measured.

## 1. Setup

| | |
| --- | --- |
| Corpus | 66,607 documents: 66,581 distinct HotpotQA validation paragraphs plus 26 synthetic incident documents from 13 files in six formats |
| Chunks | 81,243 with the default chunker (`sentence-120`), 58 duplicate chunks stored once |
| Questions | 56: 30 from HotpotQA (seed 42), 26 written for the incident corpus |
| Scored for retrieval | 51 (the 5 unanswerable questions have no gold document) |
| Metrics | Recall@K, full recall@K (every gold document in the top K), MRR, measured on documents, not chunks. A chunk shared by duplicate documents credits all of them |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384 dimensions, ONNX on CPU |
| Hardware | One Windows laptop, CPU only |

| Category | HotpotQA | Incident corpus | Total |
| --- | --- | --- | --- |
| simple | 0 | 6 | 6 |
| multi-step | 20 | 6 | 26 |
| multi-document | 10 | 4 | 14 |
| unanswerable | 0 | 5 | 5 |
| conflicting | 0 | 5 | 5 |

The incident corpus is fictional (an "Orion Pay" outage) and contains the situations the task asks for: a duplicated report, a stale runbook, a status page that contradicts the postmortem, and a rollback date that differs between two sources.

## 2. Retrieval

Four configurations on the full corpus, after the parser fix described in section 4.2:

| Configuration | recall@1 | recall@5 | recall@10 | full recall@5 | full recall@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| keyword only (BM25) | 0.418 | 0.814 | 0.882 | 0.647 | 0.765 | 0.853 |
| vectors only | 0.425 | 0.850 | 0.922 | 0.784 | 0.882 | 0.841 |
| hybrid (rank fusion) | 0.435 | **0.886** | **0.941** | **0.784** | **0.882** | 0.855 |
| hybrid + re-ranker | 0.438 | 0.866 | 0.931 | 0.725 | 0.863 | **0.875** |

With 51 questions one question moves a score by about 0.02. Differences of one or two points are noise. Larger gaps are real.

By category (recall@5):

| Category | keyword | vectors | hybrid | hybrid + re-ranker |
| --- | --- | --- | --- | --- |
| simple (6) | 1.000 | 0.833 | 1.000 | 1.000 |
| multi-step (26) | 0.769 | 0.923 | 0.904 | 0.865 |
| multi-document (14) | 0.821 | 0.738 | 0.833 | 0.833 |
| conflicting (5) | 0.800 | 0.800 | 0.800 | 0.800 |

### What the retrieval numbers say

1. **No single retriever wins everywhere.** On HotpotQA, vectors are best (recall@5 0.933) and keyword search is worst (0.733). On the incident documents, keyword search is best (0.929) and vectors are worst (0.730), because those questions contain identifiers such as `CHG-1041` that embeddings blur. Hybrid is never worst on either and is best overall. Its argument is robustness, not winning each subset.
2. **The re-ranker trades completeness for precision.** It raises MRR (0.855 to 0.875) and lowers full recall@5 (0.784 to 0.725). It scores a chunk by fused rank, by how many query terms it contains and by how much of the title appears in the query. A bridge question's second document shares few words with the question, so the re-ranker pushes it down. It helps questions with one answer document and hurts multi-hop questions, which is why it is off by default (`RERANK_RESULTS=false`).
3. **One query cannot find a document the question does not name.** All five HotpotQA misses of the hybrid retriever are bridge questions. "A group of coral islets 60 miles west of which city" needs the document for Key West, and the question never says so. This is what the agent's follow-up searches are for. It can only be measured with a model (section 6).

## 3. Chunking

Three chunkers on a 5,026 document subset, with hybrid retrieval:

| Chunker | Chunks | recall@5 | full recall@10 | MRR |
| --- | --- | --- | --- | --- |
| sentence-aware, 120 words | 6,102 | 0.895 | 0.922 | 0.858 |
| sentence-aware, 60 words | 10,461 | 0.895 | 0.902 | 0.863 |
| fixed window, 60 words, 15 overlap | 10,477 | 0.895 | 0.922 | 0.866 |

The three are within one question on every metric (hybrid without the re-ranker; the re-ranker gives the same picture). The 120-word chunker needs 71% fewer chunks, so 71% fewer embeddings and less storage, for no measurable loss. It is the default. Full table: [chunking-results.md](chunking-results.md).

This test is limited. HotpotQA paragraphs average 87 words, so most documents become one or two chunks whatever the chunker does. The comparison says little about long documents.

## 4. Failure analysis

### 4.1 What still fails

The hybrid retriever misses 11 of 51 questions at depth 5: five bridge questions from HotpotQA and six incident questions.

| Cause | Example | Fix to try |
| --- | --- | --- |
| The second document is not named by the question | "...60 miles west of which city?" | Already addressed by the agent's follow-up searches. Measure it |
| Duplicate crowding | The duplicated incident report takes two of five slots and pushes out the status page that contradicts it | Count owners of one chunk as one rank slot |
| Vocabulary mismatch | The question says "payment delay", the status page says "payments degraded" | Query expansion by the model before searching |
| Date formats | "5 March 2025" against `2025-03-05` | Add a normalised date alias to the indexed text |
| Two-step attribute questions | "Which service affected by INC-2207 was changed by a low-risk change?" | Needs two searches, so it is for the agent |

### 4.2 A defect found by this analysis

The first full run showed that rows from CSV and JSON files kept their identifier, date, service and risk in metadata only, not in the indexed text. A search for `CHG-1041` could not find the row for `CHG-1041`, and a model would never have seen the risk rating even if the row had been retrieved. The fix appends the fields to the text as one line (`id: CHG-1041; date: 2025-03-04; service: payment-gateway; risk: medium`).

| Incident questions, recall@5 | Before | After |
| --- | --- | --- |
| keyword | 0.817 | 0.929 |
| vectors | 0.706 | 0.730 |
| hybrid | 0.770 | 0.841 |
| hybrid + re-ranker | 0.794 | 0.865 |

HotpotQA numbers did not change, which is what a correct fix should do, because no HotpotQA document is affected. Simple lookups went from recall 0.833 and MRR 0.750 to 1.000 and 1.000 for hybrid.

## 5. Production path, speed and size

### 5.1 The deployed stack gives the same retrieval quality

The evaluation above runs in memory. The service runs on Postgres with an HNSW index on 16-bit vectors. The same 51 questions through the real service stack (Postgres, pgvector, BM25 rebuilt at startup, the real embedding model):

| | In memory (exact search) | Service (HNSW, 16-bit) |
| --- | --- | --- |
| recall@5 | 0.886 | 0.876 |
| recall@10 | 0.941 | 0.941 |
| full recall@5 | 0.784 | 0.765 |
| MRR | 0.855 | 0.845 |

The differences are within one question and come from approximate nearest-neighbour search and 16-bit rounding. A search takes 61 ms on average (median 48 ms, 95th percentile 145 ms) including embedding the query. Startup, which rebuilds the keyword index from 81,243 stored chunks, takes 4.6 s on a laptop and 11 s in a container.

### 5.2 Optimisations and what they bought

Each change was measured on the full corpus, and rankings were compared with the previous version.

| Change | Before | After | Check |
| --- | --- | --- | --- |
| BM25 skips postings that cannot reach the top results, and shares term strings | 445 MB, 167 ms per query | 199 MB, 52 ms per query | Identical rankings on 355 real queries, plus a test against a plain BM25 computation |
| Vectors stored as 16-bit floats | database 403 MB | database 279 MB | Retrieval quality in section 5.1 |
| Sources of all new evidence fetched in one query | one query per hit | one query per search | Tests. The saving is network round trips, so it matters on a hosted database |

How the BM25 change works. Terms are scanned from rarest to most common. Each term can add at most `idf × (k1 + 1)` to a score. Once the k-th best score so far exceeds the sum of what the remaining terms could add, no unseen document can enter the top results, so the remaining terms only update the documents already scored. The result is exact, not approximate. A test that removes the safety condition fails, which shows the pruning path is exercised.

Resource use of the container: image 1.14 GB, 255 MiB after startup and 459 MiB after the embedding model loads, against a 1 GiB limit.

## 6. Answer quality

**Measured for one configuration.** The agent (hybrid retrieval, no re-ranker) answered all 56 questions with Groq `openai/gpt-oss-120b`, over the 5,026-document set that keeps every gold document and fills the rest with seeded random ones. The raw results are in [answers-groq-hybrid_rrf.json](answers-groq-hybrid_rrf.json). A second configuration (`vector_only`) was started and then stopped: the first run needed about 190,000 tokens, and the model's free allowance is 200,000 tokens per rolling day, so the second run could not finish. Nothing below is an estimate of it.

```bash
uv run python -m investigation_engine.evaluation.cli answers --provider groq \
  --samples samples/incidents --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl \
  --max-documents 5000 --configs hybrid_rrf --pause 2 --out reports/answers-groq-hybrid_rrf.json
```

| Category | Questions | Accuracy | Citation precision | Citation recall | Searches | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| all | 56 | **0.839** | 0.899 | 0.837 | 1.68 | 3,442 |
| simple | 6 | 1.000 | 0.806 | 1.000 | 1.00 | 1,632 |
| multi-step | 26 | 0.885 | 0.981 | 0.897 | 1.50 | 2,807 |
| multi-document | 14 | 0.786 | 0.821 | 0.702 | 1.36 | 2,323 |
| unanswerable | 5 | 1.000 | not scored | not scored | 5.00 | 13,668 |
| conflicting | 5 | **0.400** | 0.800 | 0.700 | 1.00 | 1,826 |

| Metric | Meaning |
| --- | --- |
| accuracy | Right status, and for answered questions the answer contains a gold answer as whole words |
| citation precision | Share of citations whose sources include a gold document |
| citation recall | Share of gold documents covered by the citations |
| hallucination rate | Share of unanswerable questions that were answered instead of refused (0.000 here) |
| searches, tokens | Effort per question |

Exact match (0.043) and F1 (0.287) are low by construction. The agent answers in full sentences and the gold answers are short phrases, so accuracy, which asks whether the gold phrase appears in the answer, is the headline number.

### What the numbers say

1. **It refuses when it should.** All five unanswerable questions were refused and the hallucination rate is 0. But all five ended the same way: five searches and six model calls, which is the step limit. The agent refuses by running out of steps, not by deciding that the evidence is missing. That costs 13,668 tokens per question, eight times a simple question.
2. **Conflicts are the weak spot.** Two of five were reported. In the three misses (`inc-22`, `inc-23`, `inc-26`) the agent made one search and answered from the first source it found. For `inc-22` it said the delay lasted 3 hours 10 minutes, the postmortem's figure, and did not report that the status page says 47 minutes. Nothing in the loop makes the agent look for a second source once the first one answers the question. This is also what the retrieval analysis predicted: duplicates crowd out the one document that disagrees.
3. **Multi-document questions are the second weakest (0.786).** Of the three misses, two are quota errors (below) and one is an over-cautious refusal: for "who has the highest scope in profession, Rob Thomas or James Dean Bradfield" the agent cited both documents and then said the evidence does not rank them.
4. **Multi-step questions are mostly answered (0.885).** Of the three misses, two are correct answers that the strict matcher rejects: "for their assistance after the 1917 Halifax Explosion" against an answer that says "the help Nova Scotia provided after the 1917 Halifax Explosion", and "drawing the name out of a hat" against "drawing the new name out of a hat". One is wrong: the answer says 1951 and the gold answer is 1952, with both documents found and cited. If the two wording mismatches counted as correct, accuracy would be 49 of 56 (0.875). The reported number keeps the strict rule.
5. **Citations are mostly sound.** Precision 0.899 and recall 0.837. Recall is lowest on multi-document questions (0.702), where the answer needs two documents and the citations do not always cover both.
6. **Cost.** An average question took 2.6 model calls and 3,442 tokens. The whole 56-question run used about 190,000 tokens, which is the entire free daily allowance of the model.

### Defects that running a real model exposed

| Defect | Evidence | Fix |
| --- | --- | --- |
| The default Groq model no longer exists | HTTP 404 on the first real request | New default `openai/gpt-oss-120b`, from the model list the key returned |
| The model sometimes writes a broken tool call and Groq answers HTTP 400 (a tool that does not exist, arguments that are not JSON) | 3 of 56 questions on the first run | The client retries these. One of the three then passed. The other two were stopped by the quota before they could show whether the retry helps |
| The matcher compared a non-breaking hyphen (`ALRT‑5531`) and curly quotes with their ASCII forms and failed | `inc-14` was marked wrong although its answer was right | The normaliser maps typographic dashes and quotes to ASCII. Accuracy rose from 0.821 to 0.839 with no new model call |
| A model with no quota left made every question wait 90 to 180 s before an error | The deployed service, when the daily allowance ran out | An optional second Groq model takes over at once (`GROQ_FALLBACK_MODEL`) |

The two questions that still end as errors (`5ac41f1d…` and `inc-16`) are quota errors, not model failures. They are counted as wrong, so the 0.839 includes two questions the model never got to answer. They failed earlier with the broken tool calls above.

### Questions this run could not answer

- **Do follow-up searches recover the bridge questions that one search misses?** The evaluation has no single-search answer baseline, so this report cannot say.
- **Is the re-ranker better inside the agent?** Not run (`hybrid_rerank`).
- **How do the models compare?** Gemini could not be used: Google rejected the key with "free tier is not available for this project". Groq's `openai/gpt-oss-20b` was tried on three questions (one simple, one unanswerable, one conflicting) with the same pattern: the first two right, the conflict missed. Three questions is a smoke test, not a comparison.
- **Prompt injection.** The guard (evidence is quoted data, and the code refuses citations of ids that were never returned) is covered by unit tests with a scripted model. It has not been tried against a real model with a hostile document.

The deployed service (Render with Neon) was also checked by hand: a question about change `CHG-1041` was answered correctly with three citations in 1.8 s, and a question that the documents cannot answer was refused. This is a check that the deployment works, not part of the measured results.

## 7. Where to improve next

In order of expected value:

1. **Make the agent look for a second source before it answers.** Conflicting questions got one search each, and three of five were missed. A rule such as "for a number, a date or a cause, search again with different wording or another kind of document before submitting" is the cheapest fix. Measure it on the five conflict questions and check that simple questions do not get slower.
2. **Let the agent refuse earlier.** The five unanswerable questions cost 13,668 tokens each because refusal only comes at the step limit. A rule such as "after two searches that return nothing relevant, submit `insufficient_evidence`" would cut most of that cost.
3. **Rank duplicates as one slot.** Duplicate documents currently crowd out the one document that disagrees with them, which is exactly what the conflict questions need.
4. **Run the second answer configuration** (`vector_only`) once the daily allowance resets, and then `hybrid_rerank`, to answer the open questions above.
5. **Index dates in a normalised form** and keep the original wording as well.
6. **Limit the re-ranker to the first few results,** or lower the weight of the term-coverage feature, so it keeps its precision gain without hurting multi-hop questions.
7. **Let the model rewrite the query** when the first search returns weak evidence.
8. **Grow the evaluation set.** Fifty-six questions detect differences of about 0.04 at best. The incident questions were written by the same author as the documents and are probably easier than real data.

## 8. Limits of this evaluation

- Fifty-one scored questions is small for retrieval, and 56 for answers. Treat differences under about 0.04 as noise; one question is 0.018.
- The incident documents and questions share an author, so that half of the evaluation is likely optimistic.
- Retrieval is scored on a single query. The agent issues several, so single-query recall understates what the agent can reach on multi-hop questions.
- Scoring documents instead of chunks penalises duplicates. Counting a duplicated chunk once would raise the conflict-question scores.
- The chunking comparison used short documents.
- English text only.
- Answer quality was measured for one configuration, one model and one run. The model runs at temperature 0, but a hosted model is not guaranteed to repeat itself.
- The matcher is strict: it accepts a gold phrase only as whole words in order, so correct answers in other words are counted wrong (two cases above).
- Two of the 56 answers are missing because of the quota, and are counted as wrong.
- Answers were generated over a 5,026-document subset, not the full 66,607 documents. Retrieval over the full corpus is measured in section 2.

## 9. Reproduce

```bash
uv run python -m investigation_engine.evaluation.cli download
uv run python -m investigation_engine.evaluation.cli build-questions --count 30 --seed 42 --out eval/hotpot_questions.jsonl
uv run python -m investigation_engine.evaluation.cli retrieval --samples samples/incidents --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl --out reports/retrieval-after-fix.json
uv run python -m investigation_engine.evaluation.cli chunking --samples samples/incidents --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl --max-documents 5000 --out reports/chunking.json
uv run python -m investigation_engine.evaluation.cli failures --report reports/retrieval-after-fix.json --config hybrid_rrf --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl
```

Embeddings are cached in `data/embeddings.sqlite`. The first full run takes 45 to 60 minutes on a laptop CPU at about 30 passages per second; later runs only embed what changed.

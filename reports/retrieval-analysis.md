# Retrieval evaluation: results and analysis

## What was measured

- Corpus: 66,581 HotpotQA validation paragraphs (distinct by title) plus 26 synthetic incident documents, 66,607 documents in total. With the `sentence-120` chunker they become 81,243 chunks, and 58 duplicate chunks were detected and stored once.
- Questions: 56 in total. Five are unanswerable and have no gold document, so retrieval is scored on the other 51: 30 from HotpotQA (sampled with seed 42) and 21 about the incident corpus.
- Retrieval is single-shot: the question text is the only query. Documents are scored, not chunks, and a chunk shared by duplicate documents credits every one of them.
- Metrics: Recall@K, full recall@K (every gold document is in the top K) and MRR.
- Four configurations: keyword only (BM25), vectors only (`bge-small`), hybrid (reciprocal rank fusion) and hybrid with the feature re-ranker.

Raw tables: `retrieval-baseline.md` (before the fix below), `retrieval-after-fix.md`, `chunking-results.md`. Failing questions per configuration: `failures-*-after-fix.md`.

## Results after the fix

| config | recall@1 | recall@5 | recall@10 | full_recall@5 | full_recall@10 | mrr |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 0.418 | 0.814 | 0.882 | 0.647 | 0.765 | 0.853 |
| vector_only | 0.425 | 0.850 | 0.922 | 0.784 | 0.882 | 0.841 |
| hybrid_rrf | 0.435 | 0.886 | 0.941 | 0.784 | 0.882 | 0.855 |
| hybrid_rerank | 0.438 | 0.866 | 0.931 | 0.725 | 0.863 | 0.875 |

With 51 questions, one question moves a score by about 0.02. Differences of one or two points are not evidence. Larger gaps, such as keyword search against the others on HotpotQA, are.

## Findings

### 1. No single retriever wins on every kind of data

| source | questions | best recall@5 | worst recall@5 |
| --- | --- | --- | --- |
| HotpotQA | 30 | vector_only 0.933 | bm25_only 0.733 |
| incident corpus | 21 | bm25_only 0.929 | vector_only 0.730 |

Natural-language questions with paraphrases favour embeddings. Questions that contain identifiers such as `CHG-1041` favour keyword search. Hybrid was second on HotpotQA (0.917) and third on the incident corpus (0.841), but it was never the worst, and it has the best recall@5 and recall@10 overall. That robustness is the argument for it, not a win on either subset.

### 2. A defect in my own indexing, found by the failure analysis, and its fix

In the baseline, rows from CSV and JSON files kept their identifier, date, service and risk only in metadata. Their text did not contain them. A query for `CHG-1041` could not find the row for `CHG-1041`, and the model would never have seen the risk rating even if the row had been retrieved. The fix appends the fields to the text as one line (`id: CHG-1041; date: 2025-03-04; service: payment-gateway; risk: medium`).

Effect on the 21 incident questions (recall@5, before to after): BM25 0.817 to 0.929, hybrid 0.770 to 0.841, vectors 0.706 to 0.730, hybrid with re-ranker 0.794 to 0.865. HotpotQA scores did not change, as expected, because no HotpotQA document is affected. The simple lookups went from recall 0.833 and MRR 0.750 to 1.000 and 1.000 for hybrid.

### 3. The re-ranker improves the top of the list and hurts completeness

| | recall@1 | MRR | full_recall@5 |
| --- | --- | --- | --- |
| hybrid_rrf | 0.435 | 0.855 | 0.784 |
| hybrid_rerank | 0.438 | 0.875 | 0.725 |

The re-ranker scores a chunk by the fused rank, by how many query terms it contains and by how much of its title appears in the query. That favours documents that share words with the question. In a bridge question the second document usually shares few words with it, so it drops down the list. On HotpotQA, MRR rises to 0.958 while full recall@5 falls from 0.833 to 0.733. For the incident corpus the two are equal on full recall. Conclusion: it helps questions with one answer document and hurts multi-hop questions. Lowering the coverage weight, or applying it only to the top few results, are the next things to try.

### 4. What still fails, and why

Hybrid misses 11 of 51 questions at depth 5: five from HotpotQA and six incident questions.

- **A supporting document that the question does not name (all five HotpotQA misses are bridge questions).** "A group of coral islets 60 miles west of which city" needs `Key West`, and the question never names it. "Peter Gifford's rock band" needs `Midnight Oil`. One query cannot find a document that only becomes known after reading the first one. This is what the agent's follow-up searches are meant to solve, and it can only be measured with a live model. In one of the five, the right document (`2015 Orange Bowl`) also competes with near-identical neighbours (the 2012 and 2014 Orange Bowls).
- **Duplicate crowding.** The duplicated incident report fills two of the five document slots, so the status page that contradicts the postmortem is pushed out for the two conflict questions about the delay.
- **Vocabulary mismatch.** The question says "payment delay", the status page says "payments degraded".
- **Date formats.** "Which alert fired first on 5 March 2025" retrieves alerts, but from other dates, and misses one of the three from 5 March. The alerts are written `2025-03-05`, so the question's wording does not match. A normalised date alias in the indexed text is the likely fix.
- **Attribute questions that need two steps.** "Which service affected by INC-2207 was changed by a low-risk change" requires first listing the affected services and then looking up the changes.

### 5. Performance

Mean time per question over the 81,243 chunks: vector search 8.9 ms, BM25 117 ms, hybrid 112 ms, hybrid with re-ranker 150 ms. The pure-Python BM25 index is the bottleneck, because common words have long posting lists. Pruning low-information query words, or running keyword search inside the database, would address it. All of it is fast enough for interactive use.

### 6. Chunking

On a 5,026-document subset the three chunkers are tied within one question on every metric. `sentence-120` needs 6,102 chunks, against 10,461 for `sentence-60` and 10,477 for 60-word windows with 15 words of overlap. The smaller chunks need 71% more embeddings and storage for no measurable gain, so `sentence-120` is used.

## Limits of this evaluation

- 51 scorable questions is small. Treat differences under about 0.04 as noise.
- The incident documents and the incident questions were written by the same author, so that subset is likely easier than real data.
- Retrieval is scored on a single query. Answer quality, citations, abstention and conflict detection depend on the language model and are evaluated separately.
- Scoring documents instead of chunks penalises duplicates. Counting a duplicated chunk once would raise the conflict-question scores.

## Reproduce

```bash
uv run python -m investigation_engine.evaluation.cli download
uv run python -m investigation_engine.evaluation.cli build-questions --count 30 --seed 42 --out eval/hotpot_questions.jsonl
uv run python -m investigation_engine.evaluation.cli retrieval --samples samples/incidents --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl --out reports/retrieval-after-fix.json
uv run python -m investigation_engine.evaluation.cli failures --report reports/retrieval-after-fix.json --config hybrid_rrf --questions eval/hotpot_questions.jsonl eval/incident_questions.jsonl
```

Embeddings are cached in `data/embeddings.sqlite`, so a repeat run only embeds what changed. The first full run takes roughly 45 to 60 minutes on this laptop's CPU, at about 30 passages per second.

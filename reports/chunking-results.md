Chunking comparison on a 5,026 document subset, 51 scorable questions.

| config | recall@1 | recall@5 | recall@10 | full_recall@5 | full_recall@10 | mrr |
| --- | --- | --- | --- | --- | --- | --- |
| sentence-120/hybrid_rrf | 0.425 | 0.895 | 0.941 | 0.824 | 0.922 | 0.858 |
| sentence-120/hybrid_rerank | 0.428 | 0.899 | 0.931 | 0.843 | 0.902 | 0.863 |
| sentence-60/hybrid_rrf | 0.435 | 0.895 | 0.931 | 0.824 | 0.902 | 0.863 |
| sentence-60/hybrid_rerank | 0.438 | 0.895 | 0.922 | 0.824 | 0.882 | 0.874 |
| window-60-15/hybrid_rrf | 0.435 | 0.895 | 0.941 | 0.824 | 0.922 | 0.866 |
| window-60-15/hybrid_rerank | 0.428 | 0.889 | 0.931 | 0.824 | 0.902 | 0.863 |

conflicting

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| sentence-120/hybrid_rrf | 0.800 | 0.600 | 0.650 |
| sentence-120/hybrid_rerank | 0.800 | 0.600 | 0.667 |
| sentence-60/hybrid_rrf | 0.800 | 0.600 | 0.740 |
| sentence-60/hybrid_rerank | 0.800 | 0.600 | 0.667 |
| window-60-15/hybrid_rrf | 0.800 | 0.600 | 0.740 |
| window-60-15/hybrid_rerank | 0.800 | 0.600 | 0.667 |

multi_document

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| sentence-120/hybrid_rrf | 0.833 | 0.714 | 0.821 |
| sentence-120/hybrid_rerank | 0.810 | 0.714 | 0.836 |
| sentence-60/hybrid_rrf | 0.833 | 0.714 | 0.827 |
| sentence-60/hybrid_rerank | 0.833 | 0.714 | 0.875 |
| window-60-15/hybrid_rrf | 0.833 | 0.714 | 0.818 |
| window-60-15/hybrid_rerank | 0.810 | 0.714 | 0.836 |

multi_step

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| sentence-120/hybrid_rrf | 0.962 | 0.923 | 0.942 |
| sentence-120/hybrid_rerank | 0.981 | 0.962 | 0.942 |
| sentence-60/hybrid_rrf | 0.962 | 0.923 | 0.942 |
| sentence-60/hybrid_rerank | 0.962 | 0.923 | 0.942 |
| window-60-15/hybrid_rrf | 0.962 | 0.923 | 0.942 |
| window-60-15/hybrid_rerank | 0.962 | 0.923 | 0.942 |

simple

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| sentence-120/hybrid_rrf | 0.833 | 0.833 | 0.750 |
| sentence-120/hybrid_rerank | 0.833 | 0.833 | 0.750 |
| sentence-60/hybrid_rrf | 0.833 | 0.833 | 0.708 |
| sentence-60/hybrid_rerank | 0.833 | 0.833 | 0.750 |
| window-60-15/hybrid_rrf | 0.833 | 0.833 | 0.750 |
| window-60-15/hybrid_rerank | 0.833 | 0.833 | 0.750 |

[exited with code 0]

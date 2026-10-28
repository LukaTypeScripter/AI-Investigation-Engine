| config | recall@1 | recall@5 | recall@10 | full_recall@5 | full_recall@10 | mrr |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 0.418 | 0.814 | 0.882 | 0.647 | 0.765 | 0.853 |
| vector_only | 0.425 | 0.850 | 0.922 | 0.784 | 0.882 | 0.841 |
| hybrid_rrf | 0.435 | 0.886 | 0.941 | 0.784 | 0.882 | 0.855 |
| hybrid_rerank | 0.438 | 0.866 | 0.931 | 0.725 | 0.863 | 0.875 |

conflicting

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 0.800 | 0.600 | 0.667 |
| vector_only | 0.800 | 0.600 | 0.640 |
| hybrid_rrf | 0.800 | 0.600 | 0.650 |
| hybrid_rerank | 0.800 | 0.600 | 0.667 |

multi_document

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 0.821 | 0.643 | 0.893 |
| vector_only | 0.738 | 0.714 | 0.805 |
| hybrid_rrf | 0.833 | 0.714 | 0.804 |
| hybrid_rerank | 0.833 | 0.643 | 0.907 |

multi_step

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 0.769 | 0.577 | 0.854 |
| vector_only | 0.923 | 0.846 | 0.894 |
| hybrid_rrf | 0.904 | 0.808 | 0.888 |
| hybrid_rerank | 0.865 | 0.731 | 0.894 |

simple

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 1.000 | 1.000 | 0.917 |
| vector_only | 0.833 | 0.833 | 0.861 |
| hybrid_rrf | 1.000 | 1.000 | 1.000 |
| hybrid_rerank | 1.000 | 1.000 | 0.889 |

[exited with code 0]

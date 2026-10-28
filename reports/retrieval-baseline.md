| config | recall@1 | recall@5 | recall@10 | full_recall@5 | full_recall@10 | mrr |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 0.379 | 0.768 | 0.833 | 0.588 | 0.706 | 0.801 |
| vector_only | 0.395 | 0.840 | 0.902 | 0.784 | 0.863 | 0.816 |
| hybrid_rrf | 0.395 | 0.856 | 0.922 | 0.765 | 0.882 | 0.817 |
| hybrid_rerank | 0.408 | 0.837 | 0.902 | 0.706 | 0.843 | 0.839 |

conflicting

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 0.800 | 0.600 | 0.667 |
| vector_only | 0.700 | 0.600 | 0.633 |
| hybrid_rrf | 0.800 | 0.600 | 0.640 |
| hybrid_rerank | 0.800 | 0.600 | 0.667 |

multi_document

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 0.750 | 0.571 | 0.786 |
| vector_only | 0.738 | 0.714 | 0.800 |
| hybrid_rrf | 0.798 | 0.714 | 0.778 |
| hybrid_rerank | 0.798 | 0.643 | 0.836 |

multi_step

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 0.756 | 0.538 | 0.854 |
| vector_only | 0.923 | 0.846 | 0.875 |
| hybrid_rrf | 0.904 | 0.808 | 0.888 |
| hybrid_rerank | 0.865 | 0.731 | 0.894 |

simple

| config | recall@5 | full_recall@5 | mrr |
| --- | --- | --- | --- |
| bm25_only | 0.833 | 0.833 | 0.722 |
| vector_only | 0.833 | 0.833 | 0.750 |
| hybrid_rrf | 0.833 | 0.833 | 0.750 |
| hybrid_rerank | 0.833 | 0.833 | 0.750 |

[exited with code 0]

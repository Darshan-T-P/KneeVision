# RAG retrieval evaluation (top-4, 26 queries)

| Configuration | grade_hit | hit | recall | MRR |
|---|---|---|---|---|
| A: base corpus, original retriever | 0.77 | 0.81 | 0.55 | 0.65 |
| B: +AAOS, original retriever | 0.31 | 0.77 | 0.49 | 0.75 |
| C: +AAOS, current retriever (+context match) | 1.00 | 1.00 | 0.98 | 0.98 |

## Topic-specific labels only (20 queries with context-specific targets)

| Configuration | topic_hit | topic_recall | topic_MRR |
|---|---|---|---|
| A: base corpus, original retriever | 0.40 | 0.27 | 0.27 |
| B: +AAOS, original retriever | 0.80 | 0.75 | 0.75 |
| C: +AAOS, current retriever (+context match) | 1.00 | 0.96 | 0.84 |

_A cannot retrieve AAOS chunks, so its topic scores are capped for AAOS-labelled queries._

## Config C: queries whose topic-specific target was NOT retrieved

None.

# RAG generation faithfulness (real llama3.2:1b, retrieval + check + retry/fallback)

| Set | Queries | First draft flagged | Fixed by retry | Fell back to excerpts | Final summary from LLM |
|---|---|---|---|---|---|
| tuning | 26 | 5 (19%) | 3 | 2 (8%) | 24 (92%) |
| held-out | 14 | 2 (14%) | 2 | 0 (0%) | 14 (100%) |

## What gets flagged in first drafts

- mentions-unsupported-intervention: 7

Mean time per query: 23.9s (includes the retry when it happens).

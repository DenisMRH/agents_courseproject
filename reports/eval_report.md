# E2E evaluation report

Evaluation uses `data/qa/qa.jsonl` only as labels and never as RAG context.

| Category | N | Outcome | Escalation | Source hit | Tool calls | Rejection/offtopic |
|---|---:|---:|---:|---:|---:|---:|
| edge_conflict | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| edge_manipulation | 18 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| edge_no_data | 18 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| escalation_negative | 18 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| escalation_sales | 18 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| info | 45 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| offtopic | 9 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| transactional | 45 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **overall** | 180 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Sample failures

No sampled failures.

# Evaluation metrics

The evaluation script uses `data/qa/qa.jsonl` as labels only. The agent never indexes QA examples into RAG.

| Metric | Meaning |
|---|---|
| Outcome accuracy | Predicted `outcome_type` equals `expected_outcome_type` with a small allowance for clarification/rejection boundary cases. |
| Escalation accuracy | Sales and negative escalation cases create a ticket; non-escalation cases do not. |
| Source hit rate | At least one retrieved source document matches a referenced document from the QA label. |
| Tool-call correctness | Transactional and calculation cases with `client_id` call `get_client_context`; purely informational cases are not penalized for no tools. |
| Rejection/offtopic correctness | Manipulation, no-data and offtopic cases return rejection rather than hallucinated policy or client data. |

Metrics are reported per category and overall in `reports/eval_report.md`.

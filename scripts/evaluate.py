#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import AgentConfig
from src.workflow import SupportAgent


@dataclass
class Bucket:
    total: int = 0
    outcome_ok: int = 0
    escalation_ok: int = 0
    source_ok: int = 0
    tool_ok: int = 0
    rejection_ok: int = 0
    failures: List[str] = field(default_factory=list)


def load_qa(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def source_hit(expected_refs: List[str], actual_sources: List[str]) -> bool:
    if not expected_refs:
        return True
    expected_docs = {ref.split("#", 1)[0] for ref in expected_refs}
    actual_docs = {source.split("#", 1)[0] for source in actual_sources}
    return bool(expected_docs & actual_docs)


def expected_tool_use(case: Dict[str, Any]) -> bool:
    text = (case["question"] + " " + " ".join((item.get("content") or item.get("text") or "") for item in case.get("history", []))).lower()
    if case.get("expected_outcome_type") == "calculation":
        return True
    if case.get("category") == "transactional":
        return True
    if any(word in text for word in ["моя заяв", "мой кредит", "остат", "платеж", "платёж", "просроч", "реструкт"]):
        return bool(case.get("client_id"))
    return False


def tool_ok(case: Dict[str, Any], tool_names: List[str]) -> bool:
    need_tool = expected_tool_use(case)
    if need_tool:
        return "get_client_context" in tool_names
    return True


def outcome_ok(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    if expected == "clarification" and actual in {"clarification", "rejection"}:
        return True
    return False


def rejection_expected(case: Dict[str, Any]) -> bool:
    return case.get("expected_outcome_type") == "rejection"


def render_report(buckets: Dict[str, Bucket], total: Bucket) -> str:
    lines = [
        "# E2E evaluation report",
        "",
        "Evaluation uses `data/qa/qa.jsonl` only as labels and never as RAG context.",
        "",
        "| Category | N | Outcome | Escalation | Source hit | Tool calls | Rejection/offtopic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def pct(ok: int, n: int) -> str:
        return f"{(ok / n * 100):.1f}%" if n else "n/a"

    for category in sorted(buckets):
        bucket = buckets[category]
        lines.append(
            f"| {category} | {bucket.total} | {pct(bucket.outcome_ok, bucket.total)} | "
            f"{pct(bucket.escalation_ok, bucket.total)} | {pct(bucket.source_ok, bucket.total)} | "
            f"{pct(bucket.tool_ok, bucket.total)} | {pct(bucket.rejection_ok, bucket.total)} |"
        )
    lines.append(
        f"| **overall** | {total.total} | {pct(total.outcome_ok, total.total)} | "
        f"{pct(total.escalation_ok, total.total)} | {pct(total.source_ok, total.total)} | "
        f"{pct(total.tool_ok, total.total)} | {pct(total.rejection_ok, total.total)} |"
    )
    lines.extend(["", "## Sample failures", ""])
    failures: List[str] = []
    for category, bucket in sorted(buckets.items()):
        failures.extend(f"- {category}: {item}" for item in bucket.failures[:3])
    lines.extend(failures[:25] if failures else ["No sampled failures."])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    config = AgentConfig.from_env(force_local=True)
    agent = SupportAgent(config)
    buckets: Dict[str, Bucket] = defaultdict(Bucket)
    total = Bucket()

    for case in load_qa(config.qa_path):
        response = agent.answer(
            case["question"],
            client_id=case.get("client_id"),
            channel=case.get("channel", "chat_site"),
            history=case.get("history") or [],
        )
        category = case["category"]
        bucket = buckets[category]
        for target in (bucket, total):
            target.total += 1

        expected_escalation = case["category"].startswith("escalation_") or case.get("expected_outcome_type") == "escalation"
        metrics = {
            "outcome": outcome_ok(case["expected_outcome_type"], response.outcome_type),
            "escalation": expected_escalation == response.escalation.required,
            "source": source_hit(case.get("referenced_documents") or [], [chunk.source for chunk in response.sources]),
            "tool": tool_ok(case, [call.name for call in response.tool_calls]),
            "rejection": (response.outcome_type == "rejection") == rejection_expected(case) if rejection_expected(case) else True,
        }
        for target in (bucket, total):
            target.outcome_ok += int(metrics["outcome"])
            target.escalation_ok += int(metrics["escalation"])
            target.source_ok += int(metrics["source"])
            target.tool_ok += int(metrics["tool"])
            target.rejection_ok += int(metrics["rejection"])
        if not all(metrics.values()) and len(bucket.failures) < 5:
            failed = ", ".join(name for name, ok in metrics.items() if not ok)
            bucket.failures.append(
                f"{case['id']} failed {failed}; expected={case['expected_outcome_type']}, got={response.outcome_type}, intent={response.intent}"
            )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(buckets, total)
    report_path = config.reports_dir / "eval_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()

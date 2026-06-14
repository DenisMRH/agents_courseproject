#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluate import (  # noqa: E402
    CATEGORY_LABELS,
    expected_tool_use,
    load_qa,
    outcome_ok,
    rejection_expected,
)
from src.config import AgentConfig  # noqa: E402
from src.display import label_intent, label_outcome  # noqa: E402
from src.workflow import SupportAgent  # noqa: E402


RAW_COPY_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\.?|приложение\s+\d+|таблица\s+\d+)",
    re.IGNORECASE,
)


@dataclass
class QualityBucket:
    total: int = 0
    passed: int = 0
    formal_ok: int = 0
    exact_source_ok: int = 0
    readable_ok: int = 0
    failures: List[str] = field(default_factory=list)


def exact_source_hit(expected_refs: List[str], actual_sources: List[str]) -> bool:
    if not expected_refs:
        return True
    return bool(set(expected_refs) & set(actual_sources))


def readable_answer(answer: str, outcome_type: str) -> bool:
    text = answer.strip()
    if RAW_COPY_RE.match(text):
        return False
    min_len = 35 if outcome_type in {"clarification", "rejection", "escalation"} else 70
    if len(text) < min_len:
        return False
    return bool(re.search(r"[.!?]$", text))


def formal_checks(case: Dict[str, Any], response: Any) -> Dict[str, bool]:
    tool_names = [call.name for call in response.tool_calls]
    expected_escalation = case["category"].startswith("escalation_") or case.get("expected_outcome_type") == "escalation"
    needs_tool = expected_tool_use(case)
    return {
        "outcome": outcome_ok(case["expected_outcome_type"], response.outcome_type),
        "escalation": expected_escalation == response.escalation.required,
        "tool": ("get_client_context" in tool_names) if needs_tool else True,
        "rejection": (response.outcome_type == "rejection") == rejection_expected(case) if rejection_expected(case) else True,
    }


def pct(ok: int, total: int) -> str:
    return f"{ok / total * 100:.1f}%" if total else "n/a"


def load_cases(path: Path) -> Iterable[Dict[str, Any]]:
    yield from load_qa(path)


def render_report(buckets: Dict[str, QualityBucket], total: QualityBucket) -> str:
    lines = [
        "# Отчёт честной quality-проверки",
        "",
        "Проверка использует `data/qa/qa.jsonl` только как разметку. Эти примеры не индексируются в RAG.",
        "",
        "Это более строгий автоматический прокси качества, чем `scripts/evaluate.py`: кейс засчитывается только если одновременно пройдены формальные проверки, найден ожидаемый источник с точной секцией, ответ не выглядит как сырой копипаст заголовка регламента и текст достаточно читабелен. Это не заменяет ручную смысловую ревизию всех ответов.",
        "",
        f"Итоговый quality score: **{total.passed}/{total.total}** ({pct(total.passed, total.total)}).",
        "",
        "| Категория | Кол-во | Quality pass | Формальные проверки | Точная секция источника | Читаемость |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for category in sorted(buckets):
        bucket = buckets[category]
        label = CATEGORY_LABELS.get(category, category)
        lines.append(
            f"| {label} | {bucket.total} | {pct(bucket.passed, bucket.total)} | "
            f"{pct(bucket.formal_ok, bucket.total)} | {pct(bucket.exact_source_ok, bucket.total)} | "
            f"{pct(bucket.readable_ok, bucket.total)} |"
        )
    lines.append(
        f"| **итого** | {total.total} | {pct(total.passed, total.total)} | "
        f"{pct(total.formal_ok, total.total)} | {pct(total.exact_source_ok, total.total)} | "
        f"{pct(total.readable_ok, total.total)} |"
    )

    lines.extend(["", "## Проваленные кейсы", ""])
    failures: List[str] = []
    for category, bucket in sorted(buckets.items()):
        label = CATEGORY_LABELS.get(category, category)
        failures.extend(f"- {label}: {failure}" for failure in bucket.failures[:5])
    lines.extend(failures[:40] if failures else ["Проваленных кейсов по этому quality-gate нет."])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    config = AgentConfig.from_env(force_local=True)
    agent = SupportAgent(config)
    buckets: Dict[str, QualityBucket] = defaultdict(QualityBucket)
    total = QualityBucket()
    dumps: List[Dict[str, Any]] = []

    for case in load_cases(config.qa_path):
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

        formal = formal_checks(case, response)
        formal_ok = all(formal.values())
        exact_ok = exact_source_hit(case.get("referenced_documents") or [], [chunk.source for chunk in response.sources])
        readable_ok = readable_answer(response.answer, response.outcome_type)
        passed = formal_ok and exact_ok and readable_ok

        for target in (bucket, total):
            target.formal_ok += int(formal_ok)
            target.exact_source_ok += int(exact_ok)
            target.readable_ok += int(readable_ok)
            target.passed += int(passed)

        failed_checks: List[str] = []
        if not formal_ok:
            failed_checks.extend(name for name, ok in formal.items() if not ok)
        if not exact_ok:
            failed_checks.append("exact_source")
        if not readable_ok:
            failed_checks.append("readability")
        if failed_checks and len(bucket.failures) < 10:
            bucket.failures.append(
                f"{case['id']}: {', '.join(failed_checks)}; "
                f"ожидалось={label_outcome(case['expected_outcome_type'])}, "
                f"получено={label_outcome(response.outcome_type)}, интент={label_intent(response.intent)}, "
                f"источники={[chunk.source for chunk in response.sources[:3]]}"
            )

        dumps.append(
            {
                "id": case["id"],
                "category": category,
                "question": case["question"],
                "client_id": case.get("client_id"),
                "expected_outcome_type": case["expected_outcome_type"],
                "expected_sources": case.get("referenced_documents") or [],
                "actual_outcome_type": response.outcome_type,
                "intent": response.intent,
                "answer": response.answer,
                "sources": [chunk.to_dict() for chunk in response.sources],
                "tool_calls": [call.to_dict() for call in response.tool_calls],
                "escalation": response.escalation.to_dict(),
                "quality": {
                    "passed": passed,
                    "formal_ok": formal_ok,
                    "exact_source_ok": exact_ok,
                    "readable_ok": readable_ok,
                    "failed_checks": failed_checks,
                },
            }
        )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(buckets, total)
    report_path = config.reports_dir / "quality_eval_report.md"
    dump_path = config.reports_dir / "qa_answer_dump.jsonl"
    report_path.write_text(report, encoding="utf-8")
    dump_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in dumps) + "\n", encoding="utf-8")
    print(report)
    print(f"Отчёт записан в {report_path}")
    print(f"Дамп ответов записан в {dump_path}")


if __name__ == "__main__":
    main()

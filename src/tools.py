from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .schemas import ToolCall


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def rub(value: Any) -> str:
    if value is None:
        return "нет данных"
    try:
        return f"{int(value):,}".replace(",", " ") + " руб."
    except Exception:
        return str(value)


class ClientTools:
    def __init__(self, config: AgentConfig):
        self.config = config

    def get_client_profile(self, client_id: str) -> Optional[Dict[str, Any]]:
        with _connect_readonly(self.config.clients_db_path) as conn:
            row = conn.execute("select * from clients where client_id = ?", (client_id,)).fetchone()
        return dict(row) if row else None

    def get_applications(self, client_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with _connect_readonly(self.config.clients_db_path) as conn:
            rows = conn.execute(
                """
                select *
                from applications
                where client_id = ?
                order by application_date desc, application_id desc
                limit ?
                """,
                (client_id, limit),
            ).fetchall()
        return _rows_to_dicts(rows)

    def get_loans(self, client_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with _connect_readonly(self.config.clients_db_path) as conn:
            rows = conn.execute(
                """
                select *
                from credit_products
                where client_id = ?
                order by contract_date desc, contract_id desc
                limit ?
                """,
                (client_id, limit),
            ).fetchall()
        return _rows_to_dicts(rows)

    def get_client_context(self, client_id: str) -> Dict[str, Any]:
        return {
            "profile": self.get_client_profile(client_id),
            "applications": self.get_applications(client_id),
            "loans": self.get_loans(client_id),
        }

    @staticmethod
    def summarize_context(context: Dict[str, Any]) -> str:
        profile = context.get("profile")
        applications = context.get("applications") or []
        loans = context.get("loans") or []
        if not profile:
            return "клиент не найден"
        return (
            f"{profile['client_id']}: {profile['legal_form']} {profile['name']}, "
            f"скоринг {profile['credit_score']}, выручка {rub(profile['annual_revenue'])}; "
            f"заявок: {len(applications)}, кредитов: {len(loans)}"
        )

    def logged_context(self, client_id: str, calls: List[ToolCall]) -> Dict[str, Any]:
        context = self.get_client_context(client_id)
        calls.append(
            ToolCall(
                name="get_client_context",
                args={"client_id": client_id},
                result_summary=self.summarize_context(context),
            )
        )
        return context

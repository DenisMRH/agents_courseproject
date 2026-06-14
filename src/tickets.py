from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from .config import AgentConfig


class TicketStore:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.config.ensure_runtime()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.config.tickets_db_path) as conn:
            conn.execute(
                """
                create table if not exists support_tickets (
                    ticket_id text primary key,
                    created_at text not null,
                    client_id text,
                    channel text not null,
                    trigger text not null,
                    priority text not null,
                    question text not null,
                    summary text not null,
                    status text not null,
                    payload_json text not null
                )
                """
            )
            conn.commit()

    def create_ticket(
        self,
        *,
        client_id: Optional[str],
        channel: str,
        trigger: str,
        priority: str,
        question: str,
        summary: str,
        payload: Dict[str, Any],
    ) -> str:
        ticket_id = f"T-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}"
        with sqlite3.connect(self.config.tickets_db_path) as conn:
            conn.execute(
                """
                insert into support_tickets (
                    ticket_id, created_at, client_id, channel, trigger, priority,
                    question, summary, status, payload_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    datetime.now(timezone.utc).isoformat(),
                    client_id,
                    channel,
                    trigger,
                    priority,
                    question,
                    summary,
                    "open",
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            conn.commit()
        return ticket_id

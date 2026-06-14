from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from .config import AgentConfig


class TraceStore:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.config.ensure_runtime()

    def write(self, payload: Dict[str, Any]) -> str:
        trace_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        path: Path = self.config.trace_dir / f"{trace_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace_id

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "project_support_lending_msb_clean" / "Проект поддержки кредитования МСБ"
DOCUMENTS_DIR = DATA_ROOT / "data" / "documents"
CLIENTS_DB_PATH = DATA_ROOT / "data" / "clients" / "clients.sqlite"
QA_PATH = DATA_ROOT / "data" / "qa" / "qa.jsonl"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
CHROMA_DIR = RUNTIME_DIR / "chroma"
TRACE_DIR = RUNTIME_DIR / "traces"
TICKETS_DB_PATH = RUNTIME_DIR / "support_tickets.sqlite"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _parse_env_line(line: str) -> Optional[tuple[str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" in line:
        key, value = line.split("=", 1)
        return key.strip(), value.strip().strip('"').strip("'")
    for quote in ("'", '"'):
        if quote in line and line.endswith(quote):
            key, value = line.split(quote, 1)
            return key.strip(), value[:-1].strip()
    return None


def load_local_env(path: Path | None = None) -> Dict[str, str]:
    path = path or PROJECT_ROOT / ".env"
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed and parsed[0]:
            values[parsed[0]] = parsed[1]
    return values


@dataclass(frozen=True)
class AgentConfig:
    project_root: Path = PROJECT_ROOT
    documents_dir: Path = DOCUMENTS_DIR
    clients_db_path: Path = CLIENTS_DB_PATH
    qa_path: Path = QA_PATH
    runtime_dir: Path = RUNTIME_DIR
    chroma_dir: Path = CHROMA_DIR
    trace_dir: Path = TRACE_DIR
    tickets_db_path: Path = TICKETS_DB_PATH
    reports_dir: Path = REPORTS_DIR
    top_k: int = 5
    use_gigachat: bool = True
    use_chroma: bool = True
    gigachat_credentials: Optional[str] = None
    gigachat_client_id: Optional[str] = None
    gigachat_scope: str = "GIGACHAT_API_PERS"
    verify_ssl_certs: bool = False

    @classmethod
    def from_env(cls, force_local: bool = False) -> "AgentConfig":
        file_env = load_local_env()
        env = {**file_env, **os.environ}
        lowered = {key.lower(): value for key, value in env.items()}

        def get_any(*names: str) -> Optional[str]:
            for name in names:
                value = env.get(name) or lowered.get(name.lower())
                if value:
                    return value
            return None

        credentials = get_any(
            "GIGACHAT_CREDENTIALS",
            "GIGACHAT_AUTHORIZATION_KEY",
            "GIGACHAT_API_KEY",
            "Authorization_key",
            "authorization_key",
        )
        mode = (get_any("MSB_AGENT_MODE", "LLM_MODE") or "").lower()
        use_gigachat = bool(credentials) and not force_local and mode not in {"local", "offline", "mock"}
        use_chroma = (get_any("MSB_AGENT_USE_CHROMA") or "1").lower() not in {"0", "false", "no"}

        return cls(
            use_gigachat=use_gigachat,
            use_chroma=use_chroma,
            gigachat_credentials=credentials,
            gigachat_client_id=get_any("GIGACHAT_CLIENT_ID", "client_id"),
            gigachat_scope=get_any("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS",
            verify_ssl_certs=(get_any("GIGACHAT_VERIFY_SSL") or "false").lower() in {"1", "true", "yes"},
        )

    def ensure_runtime(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

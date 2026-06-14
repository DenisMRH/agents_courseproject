from __future__ import annotations

from typing import Optional

from .config import AgentConfig


class GigaChatClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._model = None
        self._available = False
        if not config.use_gigachat or not config.gigachat_credentials:
            return
        try:
            from langchain_gigachat.chat_models import GigaChat  # type: ignore

            self._model = GigaChat(
                credentials=config.gigachat_credentials,
                scope=config.gigachat_scope,
                verify_ssl_certs=config.verify_ssl_certs,
            )
            self._available = True
        except Exception:
            self._model = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._model is not None

    def invoke(self, prompt: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            result = self._model.invoke(prompt)
            return getattr(result, "content", str(result))
        except Exception:
            return None

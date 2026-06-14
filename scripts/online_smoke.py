#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import AgentConfig
from src.llm import GigaChatClient
from src.rag import RagIndex


def fail(message: str) -> None:
    raise SystemExit(f"online_smoke_failed={message}")


def retry(label: str, func, attempts: int = 3):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - diagnostic script
            last_exc = exc
            if attempt < attempts:
                print(f"{label}_retry={attempt}; error={type(exc).__name__}: {exc}")
                time.sleep(2 * attempt)
    assert last_exc is not None
    fail(f"{type(last_exc).__name__}: {last_exc}")


def main() -> None:
    config = AgentConfig.from_env()
    print(f"use_gigachat={config.use_gigachat}")
    print(f"has_credentials={bool(config.gigachat_credentials)}")

    llm = GigaChatClient(config)
    print(f"chat_available={llm.available}")
    if not llm.available:
        fail("chat_unavailable")

    response = llm.invoke("Ответь одним словом: проверка")
    print(f"chat_response_present={bool(response)}")
    if not response:
        fail("empty_chat_response")
    if response:
        print("chat_response_preview=" + response[:120].replace("\n", " "))

    rag = RagIndex(config)
    embedding_function = rag._gigachat_embedding_function()
    print(f"embedding_function={bool(embedding_function)}")
    if not embedding_function:
        fail("embedding_function_unavailable")

    def check_embeddings_and_chroma() -> None:
        vector = embedding_function(["Синтетическая проверка без клиентских и проектных данных."])[0]
        print(f"embedding_len={len(vector)}")

        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=str(config.runtime_dir / "chroma_online_smoke"))
        collection = client.get_or_create_collection("online_smoke", embedding_function=embedding_function)
        doc_id = "smoke-" + uuid4().hex[:8]
        collection.add(ids=[doc_id], documents=["Синтетический тест без данных клиента."])
        result = collection.query(query_texts=["синтетический тест"], n_results=1)
        print("chroma_query_ids=" + ",".join(result["ids"][0]))

    retry("embedding_chroma", check_embeddings_and_chroma)


if __name__ == "__main__":
    main()

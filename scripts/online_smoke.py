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


def yes_no(value: bool) -> str:
    return "да" if value else "нет"


def fail(message: str) -> None:
    raise SystemExit(f"ошибка_онлайн_проверки={message}")


def retry(label: str, func, attempts: int = 3):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - diagnostic script
            last_exc = exc
            if attempt < attempts:
                print(f"{label}_повтор={attempt}; ошибка={type(exc).__name__}: {exc}")
                time.sleep(2 * attempt)
    assert last_exc is not None
    fail(f"{type(last_exc).__name__}: {last_exc}")


def main() -> None:
    config = AgentConfig.from_env()
    print(f"используется_gigachat={yes_no(config.use_gigachat)}")
    print(f"учётные_данные_найдены={yes_no(bool(config.gigachat_credentials))}")

    llm = GigaChatClient(config)
    print(f"чат_доступен={yes_no(llm.available)}")
    if not llm.available:
        fail("чат_недоступен")

    response = llm.invoke("Ответь одним словом: проверка")
    print(f"ответ_чата_получен={yes_no(bool(response))}")
    if not response:
        fail("пустой_ответ_чата")
    if response:
        print("предпросмотр_ответа_чата=" + response[:120].replace("\n", " "))

    rag = RagIndex(config)
    embedding_function = rag._gigachat_embedding_function()
    print(f"функция_эмбеддингов={yes_no(bool(embedding_function))}")
    if not embedding_function:
        fail("функция_эмбеддингов_недоступна")

    def check_embeddings_and_chroma() -> None:
        vector = embedding_function(["Синтетическая проверка без клиентских и проектных данных."])[0]
        print(f"размерность_эмбеддинга={len(vector)}")

        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=str(config.runtime_dir / "chroma_online_smoke"))
        collection = client.get_or_create_collection("online_smoke", embedding_function=embedding_function)
        doc_id = "smoke-" + uuid4().hex[:8]
        collection.add(ids=[doc_id], documents=["Синтетический тест без данных клиента."])
        result = collection.query(query_texts=["синтетический тест"], n_results=1)
        print("идентификаторы_запроса_chroma=" + ",".join(result["ids"][0]))

    retry("эмбеддинги_chroma", check_embeddings_and_chroma)


if __name__ == "__main__":
    main()

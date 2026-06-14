#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import AgentConfig
from src.rag import RagIndex


def main() -> None:
    config = AgentConfig.from_env(force_local=True)
    index = RagIndex(config)
    chunks = index.ingest(force=True)
    documents = sorted({chunk.document_id for chunk in chunks})
    missing_meta = [chunk.chunk_id for chunk in chunks if not (chunk.document_id and chunk.chunk_id and chunk.source)]
    print(f"Indexed documents: {len(documents)}")
    print(f"Indexed chunks: {len(chunks)}")
    print("Documents: " + ", ".join(documents))
    if missing_meta:
        raise SystemExit(f"Chunks without required metadata: {missing_meta[:5]}")
    if len(documents) != 5:
        raise SystemExit(f"Expected 5 documents, got {len(documents)}")


if __name__ == "__main__":
    main()

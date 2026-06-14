from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import AgentConfig
from .schemas import RagChunk


INDEX_FILE = "fallback_index.json"
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
SECTION_RE = re.compile(r"(?P<section>(?:\d+\.)+\d*|Приложение\s+[А-ЯA-Z])")


def _source_from_heading(document_id: str, heading: str) -> str:
    match = SECTION_RE.search(heading)
    if match:
        return f"{document_id}#{match.group('section').rstrip('.')}"
    slug = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "-", heading).strip("-").lower()
    return f"{document_id}#{slug[:80] or 'root'}"


def load_document_chunks(documents_dir: Path) -> List[RagChunk]:
    chunks: List[RagChunk] = []
    for path in sorted(documents_dir.glob("*.md")):
        document_id = path.name
        lines = path.read_text(encoding="utf-8").splitlines()
        current_title = path.stem
        current_source = f"{document_id}#root"
        buffer: List[str] = []
        chunk_no = 0

        def flush() -> None:
            nonlocal buffer, chunk_no
            text = "\n".join(line.rstrip() for line in buffer).strip()
            if not text:
                buffer = []
                return
            chunk_no += 1
            chunks.append(
                RagChunk(
                    document_id=document_id,
                    chunk_id=f"{path.stem}::chunk-{chunk_no:03d}",
                    source=current_source,
                    title=current_title,
                    text=text,
                )
            )
            buffer = []

        for line in lines:
            heading = HEADING_RE.match(line)
            if heading:
                flush()
                current_title = heading.group(2).strip()
                current_source = _source_from_heading(document_id, current_title)
                buffer.append(line)
            else:
                buffer.append(line)
        flush()
    return chunks


class RagIndex:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.index_path = config.chroma_dir / INDEX_FILE
        self._chunks: List[RagChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def ingest(self, force: bool = True) -> List[RagChunk]:
        self.config.ensure_runtime()
        chunks = load_document_chunks(self.config.documents_dir)
        payload = {
            "documents_dir": str(self.config.documents_dir),
            "chunk_count": len(chunks),
            "documents": sorted({chunk.document_id for chunk in chunks}),
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.config.use_chroma:
            self._try_chroma_ingest(chunks, force=force)
        self._load_from_chunks(chunks)
        return chunks

    def load_or_ingest(self) -> None:
        if not self.index_path.exists():
            self.ingest(force=True)
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        chunks = [RagChunk(**chunk) for chunk in payload.get("chunks", [])]
        if not chunks:
            chunks = self.ingest(force=True)
        self._load_from_chunks(chunks)

    def search(self, query: str, top_k: int | None = None) -> List[RagChunk]:
        if not self._chunks:
            self.load_or_ingest()
        top_k = top_k or self.config.top_k
        if not query.strip() or not self._chunks:
            return []
        assert self._vectorizer is not None
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        boosted = [(index, self._boost_score(float(score), query, self._chunks[index])) for index, score in enumerate(scores)]
        ranked = sorted(boosted, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RagChunk(
                document_id=self._chunks[index].document_id,
                chunk_id=self._chunks[index].chunk_id,
                source=self._chunks[index].source,
                title=self._chunks[index].title,
                text=self._chunks[index].text,
                score=float(score),
            )
            for index, score in ranked
        ]

    def find_by_sources(self, sources: Sequence[str]) -> List[RagChunk]:
        if not self._chunks:
            self.load_or_ingest()
        wanted = set(sources)
        return [
            RagChunk(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title=chunk.title,
                text=chunk.text,
                score=1.0,
            )
            for chunk in self._chunks
            if chunk.source in wanted
        ]

    def _boost_score(self, score: float, query: str, chunk: RagChunk) -> float:
        q = query.lower().replace("ё", "е")
        body = f"{chunk.document_id} {chunk.title} {chunk.text}".lower().replace("ё", "е")
        doc = chunk.document_id
        boost = 0.0
        product_terms = ["бизнес-оборот", "бизнес-развитие", "бизнес-лимит", "бизнес-старт", "бизнес-перезагрузка"]
        for term in product_terms:
            if term in q and term in body:
                boost += 0.25
        if "линейка" in q and doc == "01_credit_products.md":
            boost += 0.35
        if "кредит" in q and "продукт" in q and doc == "01_credit_products.md":
            boost += 0.20
        if "заяв" in q and doc == "02_application_process.md":
            boost += 0.25
        if "документ" in q and doc == "02_application_process.md":
            boost += 0.25
        if "досроч" in q and doc == "03_early_repayment.md":
            boost += 0.35
        if "реструкт" in q and doc == "04_restructuring.md":
            boost += 0.35
        if ("эскалац" in q or "триггер" in q or "негатив" in q or "оператор" in q) and doc == "05_customer_communication.md":
            boost += 0.95
        if (
            "состояние кредита" in q
            or "статус заявки" in q
            or "компетенц" in q
            or "вне темы" in q
            or "подозритель" in q
            or "социальная инженерия" in q
            or "третьи лица" in q
            or "чужие обязательства" in q
            or "гарантирован" in q
            or "налог" in q
            or "конкурент" in q
            or "прочие продукты банка" in q
            or "качество коммуникации" in q
            or "пункты регламента" in q
        ) and doc == "05_customer_communication.md":
            boost += 0.55
        if ("справк" in q or "ссудной задолженности" in q or "прочие тарифы" in q) and doc == "01_credit_products.md":
            boost += 0.45
        if ("залог" in q or "страхование" in q or "обеспечени" in q) and doc == "02_application_process.md":
            boost += 0.50
        if ("повторная подача" in q or "повторно подать" in q or "снова откажете" in q or "учет отказов" in q or "учёт отказов" in q) and doc == "02_application_process.md":
            boost += 0.70
        if ("долговая нагрузка" in q or "24 месяца" in q or "70%" in q or "50%" in q) and doc == "01_credit_products.md":
            boost += 0.50
        if ("стоп-фактор" in q or "стоп-факторы" in q) and doc == "01_credit_products.md":
            boost += 0.55
        if "скоринг" in q and doc == "01_credit_products.md":
            boost += 0.20
        if ("рейтинг улучш" in q or "улучшить рейтинг" in q or "повысить рейтинг" in q) and doc == "01_credit_products.md":
            boost += 0.85
        return score + boost

    def _load_from_chunks(self, chunks: Sequence[RagChunk]) -> None:
        self._chunks = list(chunks)
        corpus = [f"{chunk.document_id} {chunk.title}\n{chunk.text}" for chunk in self._chunks]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1)
        self._matrix = self._vectorizer.fit_transform(corpus)

    def _try_chroma_ingest(self, chunks: Sequence[RagChunk], force: bool) -> None:
        try:
            import chromadb  # type: ignore
        except Exception:
            return
        try:
            client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
            embedding_function = self._gigachat_embedding_function()
            collection = client.get_or_create_collection("msb_lending_documents", embedding_function=embedding_function)
            if force:
                existing = collection.get(include=[])
                ids = existing.get("ids", [])
                if ids:
                    collection.delete(ids=ids)
            collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "title": chunk.title,
                    }
                    for chunk in chunks
                ],
            )
        except Exception:
            return

    def _gigachat_embedding_function(self):
        if not self.config.gigachat_credentials:
            return None
        try:
            from chromadb.api.types import EmbeddingFunction, Documents, Embeddings  # type: ignore
            from langchain_gigachat.embeddings import GigaChatEmbeddings  # type: ignore
        except Exception:
            return None

        model = GigaChatEmbeddings(
            credentials=self.config.gigachat_credentials,
            scope=self.config.gigachat_scope,
            verify_ssl_certs=self.config.verify_ssl_certs,
        )

        class GigaChatChromaEmbedding(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return model.embed_documents(list(input))

        return GigaChatChromaEmbedding()


def describe_chunks(chunks: Iterable[RagChunk]) -> str:
    return "\n\n".join(
        f"[{idx}] {chunk.source} {chunk.title}\n{chunk.text[:1400]}"
        for idx, chunk in enumerate(chunks, start=1)
    )

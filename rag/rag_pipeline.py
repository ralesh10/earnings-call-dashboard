"""Source-aware retrieval augmented generation for the dashboard assistant.

The module deliberately keeps document loading and OpenAI access on the server
side.  The browser only receives the generated answer and the filenames that
were used as supporting context.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from dotenv import load_dotenv

try:  # FAISS is used in production, but a small local fallback keeps tests portable.
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised only in minimal environments
    faiss = None


load_dotenv()

FALLBACK_MESSAGE = "I do not have enough information in the project docs to answer that."
MAX_QUESTION_LENGTH = 2_000
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-5-nano"

CURATED_SOURCE_FILES = (
    "README.md",
    "RESEARCH_README.md",
    "RESEARCH_SUMMARY.md",
    "data/validation_summary.csv",
    "data/validation_scorecard.csv",
    "data/artifacts/controlled_comparison/metrics.csv",
    "artifacts/model_comparison/metrics.csv",
    "artifacts/model_comparison/baseline_metrics.csv",
    "artifacts/experimental_rich/metrics.csv",
)


class RagConfigurationError(RuntimeError):
    """Raised when the server cannot use the configured RAG provider."""


@dataclass(frozen=True)
class SourceRecord:
    """A searchable text fragment with enough metadata for a citation."""

    text: str
    source: str
    locator: str | None = None

    @property
    def citation(self) -> str:
        return self.source if not self.locator else f"{self.source} ({self.locator})"


@dataclass(frozen=True)
class RetrievedRecord:
    record: SourceRecord
    score: float


@dataclass(frozen=True)
class KnowledgeBase:
    index: Any
    records: tuple[SourceRecord, ...]


def _get_openai_client() -> Any:
    """Create the provider client lazily so importing this module needs no key."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RagConfigurationError("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def is_openai_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def validate_question(question: Any) -> str:
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be empty")
    if len(normalized) > MAX_QUESTION_LENGTH:
        raise ValueError(f"question must be {MAX_QUESTION_LENGTH:,} characters or fewer")
    return normalized


def _fallback_chunks(text: str, chunk_size: int = 2_400, overlap: int = 300) -> list[str]:
    """Simple paragraph/character fallback for local environments without Chonkie."""

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            current = f"{current[-overlap:]}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def chunk_document(text: str) -> list[str]:
    """Chunk Markdown text using Chonkie, with a deterministic local fallback."""

    try:
        from chonkie import RecursiveChunker, RecursiveRules

        try:
            chunker = RecursiveChunker(
                chunk_size=512,
                rules=RecursiveRules(),
                chunk_overlap=64,
            )
        except TypeError:
            # Chonkie 1.x removed the overlap constructor argument. Its
            # recursive splitter remains compatible with the pipeline, while
            # the local fallback below retains overlap for older/minimal setups.
            chunker = RecursiveChunker(chunk_size=512, rules=RecursiveRules())
        chunks = chunker(text)
        return [chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in chunks]
    except ImportError:  # pragma: no cover - depends on the installed environment
        return _fallback_chunks(text)


def _csv_record_text(source: str, row_number: int, row: dict[str, str | None]) -> str:
    fields = [
        f"{key}={value.strip()}"
        for key, value in row.items()
        if value is not None and value.strip()
    ]
    return f"Source file: {source}\nRow {row_number}:\n" + "\n".join(fields)


def load_source_records(source_root: Path | None = None) -> tuple[SourceRecord, ...]:
    """Load the curated Markdown and metric files into citation-aware records."""

    root = source_root or Path(__file__).resolve().parents[1]
    records: list[SourceRecord] = []

    for relative_path in CURATED_SOURCE_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"RAG source file is missing: {relative_path}")

        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, start=2):
                    records.append(
                        SourceRecord(
                            text=_csv_record_text(relative_path, row_number, row),
                            source=relative_path,
                            locator=f"row {row_number}",
                        )
                    )
            continue

        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_document(text), start=1):
            records.append(
                SourceRecord(
                    text=chunk,
                    source=relative_path,
                    locator=f"section {index}",
                )
            )

    return tuple(records)


class _NumpyInnerProductIndex:
    """Small FAISS-compatible fallback used only when faiss-cpu is unavailable."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.vectors):
            return (
                np.empty((len(query), 0), dtype=np.float32),
                np.empty((len(query), 0), dtype=np.int64),
            )
        scores = query @ self.vectors.T
        count = min(k, self.vectors.shape[0])
        order = np.argsort(-scores, axis=1)[:, :count]
        return np.take_along_axis(scores, order, axis=1), order


def _normalize(vectors: np.ndarray) -> None:
    if faiss is not None:
        faiss.normalize_L2(vectors)
        return
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)


def _embed(texts: Sequence[str], api_client: Any | None = None) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    client = api_client or _get_openai_client()
    response = client.embeddings.create(
        model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        input=list(texts),
    )
    embeddings = np.asarray([item.embedding for item in response.data], dtype=np.float32)
    _normalize(embeddings)
    return embeddings


def create_faiss_index(
    chunks: Iterable[str | SourceRecord],
    api_client: Any | None = None,
) -> tuple[Any, list[str]]:
    """Embed chunks into a normalized inner-product index.

    The return shape remains compatible with the original pipeline.
    """

    chunk_texts = [chunk.text if isinstance(chunk, SourceRecord) else str(chunk) for chunk in chunks]
    embeddings = _embed(chunk_texts, api_client=api_client)
    if embeddings.size == 0:
        raise ValueError("cannot build a vector index without chunks")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) if faiss is not None else _NumpyInnerProductIndex(dimension)
    index.add(embeddings)
    return index, chunk_texts


def _create_index_with_records(
    records: Sequence[SourceRecord],
    api_client: Any | None = None,
) -> tuple[Any, tuple[SourceRecord, ...]]:
    index, _ = create_faiss_index(records, api_client=api_client)
    return index, tuple(records)


def retrieve_top_k_with_sources(
    query: str,
    index: Any,
    records: Sequence[SourceRecord],
    top_k: int = 5,
    api_client: Any | None = None,
) -> list[RetrievedRecord]:
    """Retrieve the most similar records with scores and citation metadata."""

    if not records:
        return []
    query_vector = _embed([query], api_client=api_client)
    distances, indices = index.search(query_vector, min(top_k, len(records)))
    return [
        RetrievedRecord(records[index], float(distances[0][position]))
        for position, index in enumerate(indices[0])
        if index != -1
    ]


def retrieve_top_k(
    query: str,
    index: Any,
    chunk_texts: Sequence[str],
    top_k: int = 3,
    api_client: Any | None = None,
) -> list[str]:
    """Backward-compatible text-only retrieval helper."""

    records = tuple(SourceRecord(text=text, source="unknown") for text in chunk_texts)
    return [item.record.text for item in retrieve_top_k_with_sources(query, index, records, top_k, api_client)]


def _context_text(retrieved: Sequence[RetrievedRecord | str]) -> str:
    context: list[str] = []
    for item in retrieved:
        if isinstance(item, RetrievedRecord):
            context.append(f"Source: {item.record.citation}\n{item.record.text}")
        else:
            context.append(str(item))
    return "\n\n---\n\n".join(context)


def generate(
    user_query: str,
    retrieved_chunks: Sequence[RetrievedRecord | str],
    api_client: Any | None = None,
) -> str:
    """Generate a strictly context-grounded answer."""

    if not retrieved_chunks:
        return FALLBACK_MESSAGE

    client = api_client or _get_openai_client()
    prompt = f"""
You are the research assistant for an ML project dashboard that studies whether
earnings-call language is associated with five-session abnormal stock returns.

Answer the user's question using ONLY the context below.

Rules:
1. Do not use outside knowledge or invent metrics, dates, model names, or claims.
2. If the context is insufficient, reply exactly:
   {FALLBACK_MESSAGE}
3. Keep different experiments, splits, and artifact versions distinct. Do not
   combine metrics from different runs into a new result.
4. Explain metrics and methodology in plain language when asked.
5. This is research documentation, not investment advice.
6. Be concise: answer in no more than three short sentences or four compact
   bullets, with a maximum of 100 words.
7. Lead with the direct answer. Do not restate the question or repeat broad
   project background unless it is necessary to answer accurately.

Context:
{_context_text(retrieved_chunks)}

Question:
{user_query}
""".strip()

    response = client.chat.completions.create(
        model=os.getenv("RAG_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        messages=[
            {"role": "system", "content": "You answer only from supplied project context."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content
    return (answer or FALLBACK_MESSAGE).strip()


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """Build the embedding index once per warm serverless function instance."""

    records = load_source_records()
    index, indexed_records = _create_index_with_records(records)
    return KnowledgeBase(index=index, records=indexed_records)


def answer_question(question: str) -> tuple[str, list[str]]:
    """Retrieve, generate, and return an answer plus unique source filenames."""

    normalized_question = validate_question(question)
    knowledge_base = get_knowledge_base()
    retrieved = retrieve_top_k_with_sources(
        normalized_question,
        knowledge_base.index,
        knowledge_base.records,
        top_k=5,
    )
    answer = generate(normalized_question, retrieved)
    sources = list(dict.fromkeys(item.record.source for item in retrieved))
    return answer, sources

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rag import rag_pipeline


class FakeEmbeddings:
    def create(self, *, model, input):
        rows = []
        for text in input:
            lowered = text.lower()
            rows.append(
                [
                    float("alpha" in lowered),
                    float("beta" in lowered),
                    float("metric" in lowered),
                ]
            )
        return SimpleNamespace(data=[SimpleNamespace(embedding=row) for row in rows])


class FakeChatCompletions:
    def create(self, *, model, messages):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer."))]
        )


class FakeClient:
    embeddings = FakeEmbeddings()
    chat = SimpleNamespace(completions=FakeChatCompletions())


def test_source_loader_includes_markdown_and_metric_rows():
    records = rag_pipeline.load_source_records()
    sources = {record.source for record in records}

    assert "README.md" in sources
    assert "RESEARCH_SUMMARY.md" in sources
    assert "data/validation_scorecard.csv" in sources
    assert any(record.locator and record.locator.startswith("row ") for record in records)


def test_index_and_retrieval_preserve_source_metadata():
    records = (
        rag_pipeline.SourceRecord("alpha research note", "README.md", "section 1"),
        rag_pipeline.SourceRecord("beta metric result", "metrics.csv", "row 2"),
    )
    index, _ = rag_pipeline.create_faiss_index(records, api_client=FakeClient())
    retrieved = rag_pipeline.retrieve_top_k_with_sources(
        "alpha",
        index,
        records,
        top_k=1,
        api_client=FakeClient(),
    )

    assert len(retrieved) == 1
    assert retrieved[0].record.source == "README.md"
    assert retrieved[0].record.citation == "README.md (section 1)"
    assert np.isfinite(retrieved[0].score)


def test_generate_returns_required_fallback_without_context():
    assert rag_pipeline.generate("What happened?", []) == rag_pipeline.FALLBACK_MESSAGE


def test_generate_uses_only_the_supplied_context():
    retrieved = [
        rag_pipeline.RetrievedRecord(
            rag_pipeline.SourceRecord("AUC was 0.631.", "metrics.csv", "row 4"),
            0.9,
        )
    ]

    assert rag_pipeline.generate("What was the AUC?", retrieved, api_client=FakeClient()) == "Grounded answer."


@pytest.mark.parametrize("question", ["", "   ", None, 123])
def test_question_validation_rejects_invalid_values(question):
    with pytest.raises(ValueError):
        rag_pipeline.validate_question(question)


def test_question_validation_enforces_maximum_length():
    with pytest.raises(ValueError, match="2,000"):
        rag_pipeline.validate_question("x" * 2_001)

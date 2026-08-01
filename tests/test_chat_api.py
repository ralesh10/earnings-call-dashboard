from __future__ import annotations

from api.chat import handle_chat_payload


def test_chat_api_returns_grounded_answer_and_deduplicated_sources():
    status, payload = handle_chat_payload(
        {"question": "What is the holdout AUC?"},
        configured=True,
        answerer=lambda question: ("The answer.", ["README.md", "README.md", "metrics.csv"]),
    )

    assert status == 200
    assert payload == {
        "answer": "The answer.",
        "sources": ["README.md", "metrics.csv"],
    }


def test_chat_api_rejects_empty_question():
    status, payload = handle_chat_payload({"question": " "}, configured=True)

    assert status == 400
    assert "must not be empty" in payload["error"]


def test_chat_api_rejects_oversized_question():
    status, payload = handle_chat_payload({"question": "x" * 2_001}, configured=True)

    assert status == 400
    assert "2,000" in payload["error"]


def test_chat_api_reports_missing_configuration():
    status, payload = handle_chat_payload({"question": "hello"}, configured=False)

    assert status == 503
    assert "not configured" in payload["error"]


def test_chat_api_maps_provider_failure_to_bad_gateway():
    def failing_answerer(question):
        raise RuntimeError("provider unavailable")

    status, payload = handle_chat_payload(
        {"question": "hello"},
        configured=True,
        answerer=failing_answerer,
    )

    assert status == 502
    assert "could not answer" in payload["error"]

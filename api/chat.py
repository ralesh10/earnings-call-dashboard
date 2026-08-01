"""Vercel serverless endpoint for the dashboard research assistant."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable

from rag.rag_pipeline import (
    RagConfigurationError,
    answer_question,
    is_openai_configured,
    validate_question,
)


LOGGER = logging.getLogger(__name__)


def handle_chat_payload(
    payload: Any,
    *,
    answerer: Callable[[str], tuple[str, list[str]]] | None = None,
    configured: bool | None = None,
) -> tuple[int, dict[str, Any]]:
    """Validate a decoded request and return an HTTP-like result for easy testing."""

    if not isinstance(payload, dict):
        return 400, {"error": "Request body must be a JSON object."}

    try:
        question = validate_question(payload.get("question"))
    except ValueError as exc:
        return 400, {"error": str(exc)}

    if configured is None:
        configured = is_openai_configured()
    if not configured:
        return 503, {"error": "The research assistant is not configured on this deployment."}

    try:
        answer, sources = (answerer or answer_question)(question)
    except RagConfigurationError:
        return 503, {"error": "The research assistant is not configured on this deployment."}
    except Exception:
        LOGGER.exception("RAG request failed")
        return 502, {"error": "The research assistant could not answer that request."}

    return 200, {"answer": answer, "sources": list(dict.fromkeys(sources))}


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):  # noqa: N801 - required Vercel entrypoint name
    """Minimal same-origin JSON handler used by Vercel Python Functions."""

    def do_POST(self) -> None:  # noqa: N802
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            _write_json(self, 400, {"error": "Content-Length must be valid."})
            return

        if content_length <= 0 or content_length > 16_384:
            _write_json(self, 400, {"error": "Request body is missing or too large."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _write_json(self, 400, {"error": "Request body must contain valid JSON."})
            return

        status, response = handle_chat_payload(payload)
        _write_json(self, status, response)

    def do_GET(self) -> None:  # noqa: N802
        _write_json(self, 405, {"error": "Use POST /api/chat to ask a question."})

    def do_OPTIONS(self) -> None:  # noqa: N802
        _write_json(self, 204, {})

"""Vercel Python serverless function: POST /api/ask {"question": "..."}.

Returns the answer plus the full search trace (queries + truncated tool
results) so the UI can show the under-the-hood view. The Anthropic API key
lives in the ANTHROPIC_API_KEY environment variable on Vercel — it is read
server-side by the SDK and never reaches the client.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import answer  # noqa: E402

MAX_QUESTION_CHARS = 500
TOOL_PREVIEW_CHARS = 600


INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — Vercel's Python app mode routes ALL paths here
        if self.path.split("?")[0] in ("/", "/index.html"):
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802 (Vercel expects this name/casing)
        try:
            length = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "invalid JSON body"})

        question = (body.get("question") or "").strip()
        if not question:
            return self._send(400, {"error": "question is required"})
        if len(question) > MAX_QUESTION_CHARS:
            return self._send(
                400, {"error": f"question too long (max {MAX_QUESTION_CHARS} chars)"}
            )

        try:
            result = answer(question)
        except Exception as e:  # surface as JSON, not a 500 HTML page
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        return self._send(
            200,
            {
                "answer": result["answer"],
                "searched": result["searched"],
                "queries": result["queries"],
                "tool_previews": [
                    t[:TOOL_PREVIEW_CHARS] + ("…" if len(t) > TOOL_PREVIEW_CHARS else "")
                    for t in result["tool_results"]
                ],
            },
        )

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

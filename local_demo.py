"""Run the Vercel web demo locally — no Vercel account needed.

Usage:
    python local_demo.py          # serves http://localhost:8001

Serves index.html and routes POST /api/ask through the same `handler`
class Vercel runs (api/ask.py), so the local demo exercises the exact
serverless code path. Requires ANTHROPIC_API_KEY in the environment.

(Opening index.html directly as a file:// page does NOT work — the page
calls fetch('/api/ask'), which needs this server or a Vercel deployment
behind it. That's the "Failed to fetch" failure mode.)
"""

import importlib.util
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8001  # web.py uses 8000; keep both runnable side by side

_spec = importlib.util.spec_from_file_location("vercel_ask", ROOT / "api" / "ask.py")
_ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ask)


class Handler(_ask.handler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path == "/api/ask":
            super().do_POST()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):  # quieter console
        pass


if __name__ == "__main__":
    print(f"Serving the web demo on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

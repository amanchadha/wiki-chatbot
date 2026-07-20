"""Minimal web UI for the wiki chatbot (stdlib only — no new dependencies).

Usage:
    python web.py            # serves http://localhost:8000

Shows a question box plus sample queries drawn from eval/cases.jsonl,
grouped by category and ordered by query complexity, so the system's
breadth of behavior (direct answers, retrieval, disambiguation, multi-hop
chains, honest abstention) is visible at a glance. Every answer displays
its search trace.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

from agent import answer

PORT = 8000
CASES_PATH = Path(__file__).resolve().parent / "eval" / "cases.jsonl"

# Display order = query complexity, simplest first. (id, title, blurb)
CATEGORIES = [
    ("no_search", "Direct answers", "No lookup needed — math, language, common knowledge"),
    ("optional_search", "Optional lookup", "Stable facts the model may verify or answer directly"),
    ("retrieval_required", "Single-hop retrieval", "Volatile or specific facts that must be verified"),
    ("disambiguation", "Disambiguation", "Ambiguous entities — retrieval must pick the right article"),
    ("multi_hop", "Multi-hop chains", "Chained searches: find an entity, then a fact about it"),
    ("unanswerable", "Unanswerable", "No real answer exists — honesty is the correct behavior"),
]
SAMPLES_PER_CATEGORY = 3

_client = anthropic.Anthropic()


def load_samples() -> list[dict]:
    by_cat: dict[str, list[str]] = {}
    for line in CASES_PATH.read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            by_cat.setdefault(case["category"], []).append(case["question"])
    return [
        {
            "category": cat,
            "title": title,
            "blurb": blurb,
            "questions": by_cat.get(cat, [])[:SAMPLES_PER_CATEGORY],
        }
        for cat, title, blurb in CATEGORIES
    ]


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wiki Chatbot</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 780px;
         margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1rem; margin: 1.2rem 0 .3rem; }
  .blurb { opacity: .7; font-size: .85rem; margin: 0 0 .4rem; }
  form { display: flex; gap: .5rem; margin: 1rem 0; }
  input[type=text] { flex: 1; padding: .6rem; font-size: 1rem; }
  button { padding: .6rem 1.2rem; font-size: 1rem; cursor: pointer; }
  .chip { display: inline-block; margin: .15rem; padding: .3rem .7rem;
          border: 1px solid #8884; border-radius: 999px; font-size: .85rem;
          cursor: pointer; background: transparent; }
  .chip:hover { background: #8882; }
  #out { margin-top: 1.2rem; white-space: pre-wrap; }
  .trace { font-size: .85rem; opacity: .75; border-left: 3px solid #8886;
           padding-left: .7rem; margin: .8rem 0; }
  .spin { opacity: .6; }
</style>
</head>
<body>
<h1>Wiki Chatbot</h1>
<p class="blurb">Claude + a <code>search_wikipedia</code> tool. It decides per
question whether to search; every answer shows its search trace. Sample
queries below are grouped by type, simplest to most complex.</p>
<form id="f">
  <input type="text" id="q" placeholder="Ask a question…" autofocus>
  <button type="submit">Ask</button>
</form>
<div id="out"></div>
<div id="cats"></div>
<script>
async function loadCats() {
  const groups = await (await fetch('/cases')).json();
  const el = document.getElementById('cats');
  for (const g of groups) {
    if (!g.questions.length) continue;
    const h = document.createElement('h2'); h.textContent = g.title;
    const b = document.createElement('p'); b.className = 'blurb'; b.textContent = g.blurb;
    el.append(h, b);
    for (const q of g.questions) {
      const c = document.createElement('button');
      c.className = 'chip'; c.textContent = q;
      c.onclick = () => { document.getElementById('q').value = q; ask(q); };
      el.append(c);
    }
  }
}
async function ask(q) {
  const out = document.getElementById('out');
  out.innerHTML = '<p class="spin">Thinking…</p>';
  try {
    const r = await fetch('/ask', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q}) });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    const trace = d.searched
      ? d.queries.map((x, i) => `[search ${i + 1}] ${x}`).join('\\n')
      : 'no search — answered from model knowledge';
    out.innerHTML = '';
    const t = document.createElement('div'); t.className = 'trace'; t.textContent = trace;
    const a = document.createElement('div'); a.textContent = d.answer;
    out.append(t, a);
  } catch (e) {
    out.innerHTML = '';
    const err = document.createElement('p');
    err.textContent = 'Error: ' + e.message; out.append(err);
  }
}
document.getElementById('f').onsubmit = (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  if (q) ask(q);
};
loadCats();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/cases":
            self._send(200, json.dumps(load_samples()).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            question = json.loads(self.rfile.read(length))["question"].strip()
            if not question:
                raise ValueError("empty question")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._send(400, f"bad request: {e}".encode(), "text/plain")
            return
        try:
            result = answer(question, client=_client)
        except Exception as e:  # surface API errors to the page
            self._send(502, f"{type(e).__name__}: {e}".encode(), "text/plain")
            return
        body = json.dumps(
            {
                "answer": result["answer"],
                "searched": result["searched"],
                "queries": result["queries"],
            }
        ).encode()
        self._send(200, body, "application/json")

    def log_message(self, fmt, *args):  # quieter console
        pass


if __name__ == "__main__":
    print(f"Serving on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

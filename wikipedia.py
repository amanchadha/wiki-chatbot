"""Wikipedia retrieval via the live MediaWiki API.

search_wikipedia(query) does two calls:
  1. list=search  -> top N matching article titles + matched-text snippets
  2. prop=extracts -> plain-text extract of the top hit (truncated)

Snippets are included per result because plain-text extracts strip
wikitables — facts that live in tables (award winners, officeholder lists)
never appear in the extract, but the search index covers table text, so
the snippet often carries exactly the matched fact.

Returns a formatted string for the model. Errors and zero-hit searches
return descriptive strings rather than raising, so the model can retry
with a reformulated query.
"""

import html
import random
import re
import threading
import time

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "wiki-chatbot-takehome/0.1 (prompt-eng assignment)"}

MAX_RESULTS = 3
EXTRACT_CHAR_LIMIT = 6000
INFOBOX_CHAR_LIMIT = 1500
TIMEOUT = 10

# Rate-limit resilience: MediaWiki 429'd us under parallel eval load.
# All requests go through _get(), which (a) throttles to one request per
# MIN_INTERVAL seconds process-wide, and (b) retries 429/503 and transient
# network errors with exponential backoff + jitter, honoring Retry-After.
MAX_RETRIES = 3
MIN_INTERVAL = 0.6  # seconds between requests, across all threads

_throttle_lock = threading.Lock()
_last_request = 0.0


def _get(params: dict) -> dict:
    global _last_request
    for attempt in range(MAX_RETRIES + 1):
        with _throttle_lock:
            wait = MIN_INTERVAL - (time.monotonic() - _last_request)
            if wait > 0:
                time.sleep(wait)
            _last_request = time.monotonic()
        try:
            resp = requests.get(
                API_URL, params=params, headers=HEADERS, timeout=TIMEOUT
            )
            if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
                retry_after = float(resp.headers.get("retry-after") or 0)
                time.sleep(max(retry_after, 2 ** (attempt + 1)) + random.uniform(0, 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** (attempt + 1) + random.uniform(0, 1))
    raise requests.RequestException("unreachable")


def _search_titles(query: str) -> list[dict]:
    data = _get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": MAX_RESULTS,
            "format": "json",
        }
    )
    return data["query"]["search"]


def _fetch_extract(title: str) -> str:
    data = _get(
        {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
        }
    )
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""


def _fetch_infobox(title: str) -> str:
    """Infobox key=value lines from the article's wikitext.

    Plain-text extracts strip infoboxes (like tables), but settlement
    populations, office dates, spouses etc. often live ONLY there. Best
    effort: any failure returns '' rather than breaking the search result.
    """
    try:
        data = _get(
            {
                "action": "query",
                "prop": "revisions",
                "rvslots": "main",
                "rvprop": "content",
                "titles": title,
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            }
        )
        wikitext = data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
    except Exception:
        return ""

    m = re.search(r"\{\{[Ii]nfobox", wikitext)
    if not m:
        return ""
    depth, end = 0, None
    for i in range(m.start(), min(len(wikitext), m.start() + 20000)):
        if wikitext[i] == "{":
            depth += 1
        elif wikitext[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return ""

    lines = []
    for line in wikitext[m.start():end].splitlines():
        line = line.strip()
        if not line.startswith("|") or "=" not in line:
            continue
        line = re.sub(r"<ref[^>]*>.*?</ref>", "", line, flags=re.S)
        line = re.sub(r"<ref[^/>]*/>", "", line)
        line = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", line)  # [[a|b]] -> b
        line = re.sub(r"<[^>]+>", " ", line)
        line = line.replace("{{", "").replace("}}", "").strip()
        # Skip empty-valued boilerplate fields (| name = ) — settlement
        # infoboxes carry dozens of them, which starves the char budget
        # before the populated fields (e.g. population_total) appear.
        value = line.partition("=")[2].strip()
        if value and len(line) <= 200:
            lines.append(line)
    return "\n".join(lines)[:INFOBOX_CHAR_LIMIT]


def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a formatted result string for the model."""
    try:
        results = _search_titles(query)
    except requests.RequestException as e:
        # "SEARCH ERROR:" prefix is load-bearing: the eval harness counts
        # tool failures by matching it (tool_error_rate).
        return f"SEARCH ERROR: Wikipedia search failed ({e}). You may retry."

    if not results:
        return (
            f"No Wikipedia articles found for '{query}'. "
            "Try different or broader search terms."
        )

    # Take the first result whose extract is non-trivial (some heavy-template
    # pages return near-empty extracts).
    extract, chosen = "", None
    for result in results:
        try:
            extract = _fetch_extract(result["title"])
        except requests.RequestException:
            continue
        if len(extract) > 100:
            chosen = result
            break

    if chosen is None:
        titles = ", ".join(r["title"] for r in results)
        return (
            f"Found articles ({titles}) but could not retrieve usable content. "
            "You may retry with a more specific query."
        )

    truncated = extract[:EXTRACT_CHAR_LIMIT]
    if len(extract) > EXTRACT_CHAR_LIMIT:
        truncated += "\n[...extract truncated]"

    # Matched-text snippets for every result (tables included — extracts
    # strip those, so a table-borne fact may appear only here).
    snippet_lines = []
    for r in results:
        snippet = html.unescape(re.sub(r"<[^>]+>", "", r.get("snippet", ""))).strip()
        if snippet:
            snippet_lines.append(f"- {r['title']}: ...{snippet}...")

    parts = []
    if snippet_lines:
        parts.append("Search result snippets (matched text):\n" + "\n".join(snippet_lines))
    parts.append(f"# {chosen['title']}\n\n{truncated}")

    infobox = _fetch_infobox(chosen["title"])
    if infobox:
        parts.append(
            f"Infobox of '{chosen['title']}' (structured facts — populations, "
            f"dates, roles, spouses often appear only here):\n{infobox}"
        )

    other_titles = [r["title"] for r in results if r["title"] != chosen["title"]]
    if other_titles:
        parts.append(
            "Other matching articles (search again with one of these titles "
            "if this is the wrong page): " + ", ".join(other_titles)
        )
    return "\n\n".join(parts)

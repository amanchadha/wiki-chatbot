# Wiki Chatbot

A question-answering system built on Claude with a `search_wikipedia` tool.
Claude decides per-question whether retrieval is needed, chains searches for
multi-hop questions, and grounds answers in retrieved evidence. An eval suite
grades three dimensions — search decision, retrieval quality, answer quality —
and drove every iteration of the system prompt (see `RATIONALE.md` for the
full iteration story).

**Final numbers (40-case suite, two consecutive runs of the final config):**
pass rate 1.0 / 0.975*, search recall 0.955, search precision 1.00 (across all
seven runs of the project), right-article rate 1.0, zero faithfulness
violations, zero tool errors. *See `RATIONALE.md` — the 40/40 run followed two
eval-side grading calibrations; the system config was identical in both runs.

## Setup

Requires Python 3.9+ and an Anthropic API key.

```bash
pip install -r requirements.txt          # anthropic, requests
export ANTHROPIC_API_KEY=sk-ant-...
```

Wikipedia access uses the live MediaWiki API — no dump, index, or extra keys.
Optional (only for the lexical-metrics experiment, not needed to run the bot
or the eval): `pip install sacrebleu rouge-score bert-score`.

## Usage

```bash
python cli.py "Who is the current president of Botswana?"   # one-shot
python cli.py                                               # interactive
python cli.py --demo                                        # sample queries
```

Every answer is printed with its search trace — each query issued, or an
explicit "no search — answered from model knowledge" line.

## Web demo (Vercel)

A minimal single-page demo (`index.html` + `api/ask.py`) that shows the
answer **and the full search trace** — every query issued with an expandable
tool-result preview, or an explicit "no search" line. Sample-query chips
mirror the CLI's `--demo` set. No chat history, no streaming.

**The API key stays server-side**: `api/ask.py` runs as a Vercel Python
serverless function and the SDK reads `ANTHROPIC_API_KEY` from the function's
environment; the browser only ever calls `/api/ask`.

Deploy:

```bash
npm i -g vercel
vercel login
vercel env add ANTHROPIC_API_KEY   # paste key; applies to the serverless env only
vercel --prod
```

Notes: `vercel.json` sets `maxDuration: 60` for the function — multi-hop
questions with several searches can take ~30s. To test locally without a
Vercel account, any shim that serves `index.html` and routes POST
`/api/ask` through the `handler` class in `api/ask.py` works.

## Running the eval

```bash
python eval/run.py            # full 40-case run (~5-8 minutes, ~$2 API cost)
python eval/run.py r03 mh-06  # subset by case id
```

Each run writes `eval/runs/<timestamp>_<prompt_version>/`:

- `summary.json` — all metrics for the run (also appended to
  `eval/runs/history.jsonl`, one row per run, tagged with `prompt_version`
  and `judge_version`)
- `cases/<id>.json` — per-case transcript: queries issued, full tool
  results, final answer, all judge fields, lexical metrics, token usage

Failing cases are printed with their transcripts at the end of each run.

### Case schema (`eval/cases.jsonl`)

| Field | Meaning |
|---|---|
| `id`, `category`, `question` | `category` ∈ retrieval_required, no_search, optional_search, disambiguation, unanswerable, multi_hop |
| `expect_search` | `true` / `false` / `"optional"` — optional cases are excluded from search precision/recall |
| `expected_answer` | Clean ground truth (also the reference for lexical metrics) |
| `notes` | Optional grading guidance for the judge — authoritative over the judge's own knowledge (tolerances, anticipated wrong answers, "grade against retrieved evidence") |
| `min_searches` | multi_hop only: minimum searches the chain requires (drives `multi_hop_underchained`) |
| `abstention_ok` | Dead-end chains: honest abstention counts as a pass |

### Metrics (three dimensions)

1. **Search decision** (pure code): recall/precision of the search-vs-answer
   choice, spurious and missed search lists, `multi_hop_underchained`.
2. **Retrieval quality** (judged): `evidence_sufficiency` (did tool results
   contain the needed fact?), `right_article` (did retrieval land on the
   right page?), plus code-level `tool_error_rate` so infra failures never
   pollute the judged metrics.
3. **Answer quality** (judged): verdict (correct / partially_correct /
   incorrect / honest_abstention), `faithfulness` (does the answer follow the
   evidence?), abstention appropriateness, and a one-line judge reason per
   case.

Every failure decomposes along the chain: wrong + no search = decision
failure; wrong + search + insufficient evidence = retrieval failure; wrong +
sufficient evidence = generation failure. That triage drove each iteration.

### Models

- **Agent: `claude-opus-4-8`** — adaptive thinking, `effort: low`. The
  search/no-search decision is the core prompt-engineered behavior, and Opus
  4.8 has the most faithful prompt-steered tool triggering; it also accepts
  no sampling parameters, which helps eval reproducibility.
- **Judge: `claude-sonnet-5`** — deliberately a *different* model from the
  agent to avoid self-grading bias. Judge output is schema-enforced JSON.
  A 10-case blind human spot-check matched the judge 10/10 on the baseline.

### Judge versions

Judge prompt changes are version-tagged (`judge_version` in every summary
row) so scores stay comparable across runs:

- **judge-v1** — rubric judge: 4 fields, per-notch examples, authoritative
  case notes, outdated-expectation disagreements flagged in `reason`.
- **judge-v2** — faithfulness carve-out: under a *total search outage*
  (every tool result an error), an answer from memory that is clearly
  labeled unverified is no longer graded `unfaithful`; the same answer
  without the label still is.

### Lexical-metrics experiment

BLEU / ROUGE / BERTScore / MoverScore were computed per case as a sidecar and
compared against judge verdicts (`eval/metric_analysis.py`). **BLEU and ROUGE
were dropped**: on short factual answers a fluent wrong answer overlaps the
reference as well as a right one (the baseline's one incorrect answer scored
*above* the correct-group ROUGE-1 mean), and correct answers span 0.0–0.76 so
no threshold separates the groups. **BERTScore deferred** (model download
infeasible on the dev network — recomputable offline from saved run
artifacts); **MoverScore dropped** (unbuildable). The LLM judge is the
answer-quality signal of record.

## Reproduction notes

- `wikipedia.py` throttles to ~1.6 requests/s process-wide and retries
  429/503 with exponential backoff; the eval runs 2 workers. Raising either
  reintroduces MediaWiki rate-limiting (a 52% tool-failure run taught us).
- Tool results include search snippets and infobox fields because plain-text
  extracts strip tables and infoboxes — award winners and settlement
  populations live only there.
- Expected run-to-run variance: ±1 case on search recall (borderline stable
  historical years, e.g. r08), occasionally graded ±1 on pass rate. Search
  precision has been 1.0 on every run.
- Live-Wikipedia caveat: current-events cases (r01–r12, mh-01…) encode facts
  as of July 2026; if an officeholder changes, update `expected_answer` /
  `notes` — the judge grades against evidence where noted.
- `eval/runs/history.jsonl` contains one invalid row (timestamp
  `20260719T213519Z`, all 40 cases errored) from an API-credit outage
  mid-run; ignore it when reading the history. A few additional rows are
  small subset runs made during development — full-suite rows are the ones
  with `n_cases` ≥ 32.

## Repository layout

| File | Role |
|---|---|
| `prompts.py` | System prompt + tool definition — the artifact under eval-driven iteration; `PROMPT_VERSION` tags every run |
| `wikipedia.py` | `search_wikipedia(query)`: MediaWiki search → extract + snippets + infobox, throttle + retries |
| `agent.py` | Manual tool-use loop (max 5 round trips); returns answer + `searched`/`queries`/`tool_results`/usage |
| `cli.py` | One-shot / interactive / demo entry point |
| `index.html` + `api/ask.py` + `vercel.json` | Vercel web demo: static page + Python serverless function (key server-side) |
| `eval/run.py` | Harness: agent + judge + metrics per run |
| `eval/cases.jsonl` | 40 graded cases across 6 categories |
| `eval/metric_analysis.py` | Lexical-metric vs judge agreement analysis |
| `eval/runs/` | Per-run artifacts + `history.jsonl` |
| `RATIONALE.md` | Design rationale, iteration history, failure analysis |

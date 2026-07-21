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

**Citation-mode baseline (`--cite`, same 40 cases):** 40/40 pass, citation
coverage 1.0 on every searched answer, **zero invented or misattributed
citations**, zero unsupported side claims, default-mode behavioral metrics
unchanged; cost ≈ +6% input / +12% output tokens.

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
python cli.py --cite "How many moons does Saturn have?"     # citation mode
python web.py                                               # web UI at http://localhost:8000
```

Citation mode (`--cite`) appends citation instructions to the system prompt:
answers carry inline superscript markers on retrieved-text claims plus a
source list naming the Wikipedia articles used. Off by default.

Every answer is shown with its search trace — each query issued, or an
explicit "no search — answered from model knowledge" line.

The web UI (stdlib only, no extra dependencies) also displays sample
queries drawn from `eval/cases.jsonl`, grouped by query type and ordered by
complexity — direct answers → optional lookup → single-hop retrieval →
disambiguation → multi-hop chains → unanswerable — so the system's breadth
of behavior is demonstrable in one glance. Click any chip to run it live.

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
Vercel account:

```bash
python local_demo.py               # http://localhost:8001
```

This serves `index.html` and routes POST `/api/ask` through the same
`handler` class Vercel runs. **Opening `index.html` directly as a file
does not work** — the page calls `fetch('/api/ask')`, which needs a server
behind it; without one every chip click fails with "Failed to fetch".

## Running the eval

```bash
python eval/run.py                       # full 40-case run (~5-8 min, ~$2 API cost)
python eval/run.py r03 mh-06             # subset by case id
python eval/run.py --cite                # cite-mode run (citation rubrics added)
python eval/run.py --rejudge <run_dir>   # re-grade a finished run's frozen
                                         # transcripts under the current judge
                                         # (no agent re-run; ~$0.50)
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
| `expect_search` | `true` / `false` / `"optional"` — optional cases are excluded from search precision/recall (see note below) |
| `expected_answer` | Clean ground truth (also the reference for lexical metrics) |
| `notes` | Optional grading guidance for the judge — authoritative over the judge's own knowledge (tolerances, anticipated wrong answers, "grade against retrieved evidence") |
| `min_searches` | multi_hop only: minimum searches the chain requires (drives `multi_hop_underchained`) |
| `abstention_ok` | Dead-end chains: honest abstention counts as a pass |

**On `"optional"` (e.g. "Who painted the Mona Lisa?"):** optional does not
mean a lookup is needed — these are stable, pre-training-cutoff facts the
model knows cold, and in every final run it answers them from memory without
searching (correctly). They're classified `optional` rather than `false`
because they sit on the boundary of the prompt's two decision lists: the
fact is common knowledge (answer directly), but it's also literally a "named
specific about a particular work" (search first), so *either* behavior is
defensible and neither is punished — the case is excluded from search
precision and recall, while answer correctness (and, if it does search,
retrieval quality) is still graded. In the demo, this category exists to
show calibrated restraint: the model declines to burn a search even when a
question pattern-matches the search-trigger list, which is the live
demonstration of the suite-wide 1.0 search precision.

**Why an "optional" search value?** Some questions sit exactly on the
boundary of the system prompt's two decision lists. "Who painted the Mona
Lisa?" is stable, pre-cutoff common knowledge (the answer-directly list)
*and* literally a "named specific about a particular work" (the
search-first list) — so searching and answering from memory are both
defensible, and punishing either would make the metric arbitrary. Optional
cases are therefore excluded from search precision/recall in both
directions; answer correctness is still graded, and if the model does
search, retrieval quality (right article, faithfulness) is graded too.
Empirically, the model answers all four optional cases from memory,
correctly, in every final run — no lookup is *required*, and that is the
point: the demo's "Optional lookup" chip exists to display this calibrated
restraint (a fact that pattern-matches the search triggers but doesn't
waste a search), which is the live face of the suite's 1.0 search
precision.

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
row) so scores stay comparable across runs.

**A note on the numbering:** versions are allocated in the order changes
landed, across *both* judge modes — they are not a single default-mode
lineage. v3 went to the citation rubrics, which apply only to cite-mode
runs; when the side-claims rubric was added to the shared base judge next,
it became v4. So for **default-mode** runs the effective lineage is
v1 → v2 → v4 (v3 never applied to them), and for **cite-mode** runs it is
v3 → v5 (v5 = v4's base + v3's citation rubrics). Each summary row's
`judge_version` states exactly which grader produced it.

- **judge-v1** — rubric judge: 4 fields, per-notch examples, authoritative
  case notes, outdated-expectation disagreements flagged in `reason`.
- **judge-v2** — faithfulness carve-out: under a *total search outage*
  (every tool result an error), an answer from memory that is clearly
  labeled unverified is no longer graded `unfaithful`; the same answer
  without the label still is.
- **judge-v3** — citation rubrics, applied to cite-mode (`--cite`) runs
  only; default-mode runs continue to grade under judge-v2. Adds
  `citation_coverage` (are retrieved-text claims marked?) and
  `citation_support` (does the cited article's retrieved text actually
  back each claim?). `invented` — any citation on a claim with no
  retrieved support — is the failure citation mode is designed to prevent.
- **judge-v4** — hallucination coverage for side details: new
  `unsupported_side_claims` rubric lists every specific factual claim
  beyond the key claim (names, dates, numbers, roles) supported neither by
  retrieved evidence nor the expected answer — closing the gap where
  faithfulness guards only the key claim while embellishments (common in
  multi-hop chain narration) ride along ungraded. Searched cases only;
  labeled-unverified and common-knowledge framing excluded. Summary
  metrics: `answers_with_unsupported_side_claims`,
  `unsupported_side_claims_total`.
- **judge-v5** — judge-v4 + the v3 citation rubrics (cite-mode runs).

The final v1.3 runs were **re-judged** under judge-v4 via
`python eval/run.py --rejudge <run_dir>` — judge-only replay over frozen
agent transcripts (no agent re-run, no behavior drift; `rejudged_from`
links the new summary row to its source run).

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
| `web.py` | Local web UI (stdlib only): category-grouped sample chips from `eval/cases.jsonl`, port 8000 |
| `index.html` + `api/ask.py` + `vercel.json` | Vercel web demo: static page + Python serverless function (key server-side) |
| `local_demo.py` | Runs the Vercel demo locally (serves `index.html`, routes `/api/ask`), port 8001 |
| `eval/run.py` | Harness: agent + judge + metrics per run |
| `eval/cases.jsonl` | 40 graded cases across 6 categories |
| `eval/metric_analysis.py` | Lexical-metric vs judge agreement analysis |
| `eval/runs/` | Per-run artifacts + `history.jsonl` |
| `RATIONALE.md` | Design rationale, iteration history, failure analysis |

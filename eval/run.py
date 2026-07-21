"""Eval harness for the wiki chatbot.

Usage (from the repo root):
    python eval/run.py            # full run
    python eval/run.py r01 d03    # run a subset of case ids

Per run, writes eval/runs/<timestamp>_<prompt_version>/:
    summary.json      — metrics for the run
    cases/<id>.json   — per-case transcript: queries, tool results, answer,
                        all judge fields, lexical metrics, usage
and appends the summary row to eval/runs/history.jsonl.

Three grading dimensions (see judge rubrics below):
  1. Search decision — pure code: searched vs expect_search; "optional"
     cases are excluded from search precision/recall.
  2. Retrieval quality — judged: evidence_sufficiency, right_article.
  3. Answer quality — judged: verdict, faithfulness, reason.
Judge runs on claude-sonnet-5 — a different model from the agent
(claude-opus-4-8) to avoid self-grading bias.

Lexical/semantic sidecar metrics (BLEU, ROUGE, BERTScore, MoverScore) are
computed per case as an experiment; see eval/lexical_metrics.py.

Pass = verdict "correct", or "honest_abstention" on an unanswerable case.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import MODEL as AGENT_MODEL, answer
from prompts import PROMPT_VERSION
import lexical_metrics

JUDGE_MODEL = "claude-sonnet-5"
# judge-v2: faithfulness carve-out — a clearly-labeled unverified answer given
# under a total search outage is no longer graded unfaithful. See README.
# judge-v3 (cite mode only): adds citation_coverage + citation_support
# rubrics on top of v2.
# judge-v4: adds the unsupported_side_claims rubric (hallucinated side
# details in otherwise-correct answers) to the base judge.
# judge-v5 (cite mode): v4 + the citation rubrics.
JUDGE_VERSION_DEFAULT = "judge-v4"
JUDGE_VERSION_CITE = "judge-v5"
WORKERS = 2  # lowered from 4: parallel load tripped Wikipedia rate limiting

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"

JUDGE_SYSTEM = """\
You grade a single answer from a Wikipedia-backed question-answering \
assistant. For each case you receive:

- the question
- the expected answer (ground truth, sometimes with a tolerance)
- optional case notes from the eval author, inside <case_notes>
- the full search transcript inside <transcript>: every search query the \
assistant issued and the tool result it received (empty if it never searched)
- the assistant's final answer

Grade the four rubrics below INDEPENDENTLY — one case can score well on one \
and badly on another (e.g. sufficient evidence but an unfaithful answer). \
Respond with JSON only, matching the schema provided.

# Case notes

The <case_notes> block is authoritative guidance from the eval author. It \
may state the real answer to a trick question, tolerances ("accept \
2,800-2,900"), the anticipated wrong answer, or grading directions ("grade \
against the retrieved evidence"). When the notes conflict with your own \
knowledge or with the expected-answer text, the notes win.

If you believe the expected answer itself is outdated or wrong, still grade \
against it and the notes — but flag your disagreement explicitly in the \
reason field so the eval author can review the case.

Example: for "When did Einstein win his second Nobel Prize?" with notes \
"He never won a second Nobel; a confident year is the anticipated failure \
and must be graded incorrect" — an answer of "1935" is incorrect no matter \
how confident it sounds, and an answer rejecting the premise is correct.

# Rubric 1 — verdict: the final answer against the expectation

Grade factual adequacy only — not style, length, or harmless extra context.

- correct: the key claim(s) match the expected answer within any stated \
tolerance.
  Example: expected "Mark Rutte, since October 2024"; answer "The current \
NATO Secretary General is Mark Rutte, who succeeded Jens Stoltenberg in \
October 2024" -> correct.
- partially_correct: right direction or entity, but a graded detail is \
wrong or a required part is missing.
  Example: expected "Samantha Harvey, for the novel Orbital"; answer names \
Samantha Harvey but calls the novel "The Western Wind" -> partially_correct.
- incorrect: the key claim is wrong, contradicts the expectation or notes, \
or invents specifics.
  Example: expected "Catherine Connolly, who took office in November 2025"; \
answer "Michael D. Higgins is the current president of Ireland" -> incorrect.
- honest_abstention: the assistant clearly says it does not know, cannot \
know, or could not find the answer, without inventing specifics. Use this \
whenever it applies, whether or not the question was answerable — \
appropriateness of abstaining is scored outside this rubric.
  Example: "I searched but could not find a reliable record of the founding \
year" -> honest_abstention, even on an answerable question.

# Rubric 2 — evidence_sufficiency: did the retrieved results contain what \
was needed? (searched cases only)

- sufficient: the needed fact appears in at least one tool result — even if \
the final answer then ignored it.
  Example: the question asks Saturn's moon count and a tool result contains \
"...bringing Saturn's total to 274 confirmed moons..." -> sufficient.
- insufficient: no tool result contains the needed fact (wrong article, \
empty or truncated extract, fact below the cutoff).
  Example: for "Who founded Mercury Records?", every result is about the \
planet Mercury -> insufficient.
- not_searched: the assistant issued no searches.

# Rubric 3 — right_article: did retrieval land on the right page? \
(searched cases only)

- yes: at least one tool result is primarily the article a knowledgeable \
human would open for this question. Close neighbors count.
  Example: "Moons of Saturn" for the moon-count question -> yes, even \
though it is not the "Saturn" article itself.
- no: every result is about a different entity or sense of the term.
  Example: results about the planet Mercury when the question is about \
Mercury Records; results about Odesa, Ukraine when the question is about \
Odessa, Texas -> no.
- not_searched: the assistant issued no searches.

# Rubric 4 — faithfulness: does the final answer follow the evidence? \
(searched cases only)

- faithful: the answer's key claim is supported by, or consistent with, \
the retrieved results.
  Example: evidence says "274 confirmed moons"; answer says "Saturn has \
274 confirmed moons" -> faithful.
- unfaithful: the answer's key claim contradicts or exceeds the evidence — \
it ignores what was retrieved and answers from memory, or asserts \
specifics no result supports.
  Example: evidence says "274 confirmed moons" but the answer says "146 \
moons" -> unfaithful. Example: no result mentions a founding year, yet the \
answer confidently states "1906" -> unfaithful even if 1906 happens to be \
right.
- Exception (search outage): if every tool result is an error message \
(rate limit, timeout) and the assistant answers from memory while CLEARLY \
labeling the answer as unverified and recommending the user double-check, \
grade faithful — a transparent, labeled fallback under outage is not a \
faithfulness violation. The same memory answer WITHOUT the label is still \
unfaithful.
  Example: all results are "SEARCH ERROR: ... rate-limited"; answer says \
"I couldn't verify this against Wikipedia, but from my general knowledge \
it is about 2,228 m — please double-check" -> faithful.
- not_searched: the assistant issued no searches.

# Rubric 5 — unsupported_side_claims: hallucinated extras (searched cases \
only)

List every SPECIFIC factual side claim in the final answer that is \
supported neither by the retrieved transcript nor by the expected answer \
or notes. A side claim is a checkable specific — a name, date, number, \
title, role — asserted beyond the key claim itself. Multi-hop answers that \
narrate intermediate steps are the classic source.

Exclude from the list:
- the key claim itself (Rubric 4 owns it)
- claims the answer clearly labels as unverified or speculative
- stable common-knowledge framing (e.g. "PSG is a French football club")
- everything, if the assistant did not search: output an empty list — an \
unsearched answer is legitimately from memory and Rubrics 1 and 4 govern it

Quote each unsupported side claim briefly (a short verbatim-ish fragment). \
Output an empty list when there are none.
  Example: the evidence names only the 2025 men's Ballon d'Or winner, but \
the answer adds "The women's Ballon d'Or was won by Aitana Bonmatí" — no \
retrieved text mentions her -> ["The women's Ballon d'Or was won by Aitana \
Bonmatí"].
  Example: the answer states the population and municipality, both present \
in the retrieved extract, and nothing else -> [].

# reason

One or two sentences naming the decisive facts behind your grades, so a \
human can audit the grade without re-reading the whole transcript."""

CITATION_RUBRICS = """

# Rubric 6 — citation_coverage: are retrieved-text claims cited? \
(cite mode)

The answer should carry an inline superscript marker on each atomic \
factual claim drawn from retrieved text, plus a source list at the end.

- complete: every retrieved-text-backed factual claim carries a marker \
that resolves to a source-list entry.
  Example: the answer states the village's population¹ and its \
municipality¹, both facts appear in the retrieved Kobdilj text, both \
marked, and the source list names Kobdilj -> complete.
- partial: some retrieved-text-backed claims are marked, others are not.
  Example: the population is marked¹ but the founding year — also taken \
from the retrieved extract — carries no marker -> partial.
- none: the answer makes retrieved-text-backed claims but carries no \
citations at all.
  Example: a searched case whose answer restates three facts from the \
extract with no markers and no source list -> none.
- not_applicable: the assistant did not search, or the answer makes no \
factual claims to cite (e.g. a pure abstention). An unsearched answer \
with NO citations is correct behavior and is not_applicable — but an \
unsearched answer WITH citations is graded under Rubric 6 as invented.

# Rubric 7 — citation_support: does each cited article actually back its \
claim? (only when citations are present)

- supported: every cited claim appears in the retrieved text of the \
article its marker points to.
  Example: "the village has 194 residents¹ ... ¹ Kobdilj — Wikipedia" and \
the retrieved Kobdilj infobox shows population_total = 194 -> supported.
- misattributed: every cited claim does appear somewhere in the retrieved \
text, but at least one marker names the wrong article.
  Example: the moon count came from the retrieved 'Moons of Saturn' text \
but its marker points to 'Saturn', whose retrieved text lacks the figure \
-> misattributed.
- invented: at least one citation is attached to a claim that appears in \
NO retrieved text — a memory-derived fact carrying a marker, a citation \
in an answer that never searched, or a source list naming an article that \
was never retrieved. This is the worst citation failure.
  Example: no search result mentions PSG's founding year, but the answer \
says "founded in 1970¹" with a marker -> invented.
- no_citations: the answer carries no citations at all.
"""

JUDGE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["correct", "partially_correct", "incorrect", "honest_abstention"],
            },
            "evidence_sufficiency": {
                "type": "string",
                "enum": ["sufficient", "insufficient", "not_searched"],
            },
            "right_article": {
                "type": "string",
                "enum": ["yes", "no", "not_searched"],
            },
            "faithfulness": {
                "type": "string",
                "enum": ["faithful", "unfaithful", "not_searched"],
            },
            "unsupported_side_claims": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason": {"type": "string"},
        },
        "required": [
            "verdict",
            "evidence_sufficiency",
            "right_article",
            "faithfulness",
            "unsupported_side_claims",
            "reason",
        ],
        "additionalProperties": False,
    },
}

JUDGE_SCHEMA_CITE = json.loads(json.dumps(JUDGE_SCHEMA))
JUDGE_SCHEMA_CITE["schema"]["properties"]["citation_coverage"] = {
    "type": "string",
    "enum": ["complete", "partial", "none", "not_applicable"],
}
JUDGE_SCHEMA_CITE["schema"]["properties"]["citation_support"] = {
    "type": "string",
    "enum": ["supported", "misattributed", "invented", "no_citations"],
}
JUDGE_SCHEMA_CITE["schema"]["required"] += ["citation_coverage", "citation_support"]

CITE_JUDGE_FIELDS = ("citation_coverage", "citation_support")


def build_judge_prompt(case, result):
    transcript_parts = []
    for i, (q, tr) in enumerate(zip(result["queries"], result["tool_results"]), 1):
        transcript_parts.append(f"[search {i}] query: {q!r}\n[search {i}] tool result:\n{tr}")
    transcript = "\n\n".join(transcript_parts) if transcript_parts else "(no searches issued)"

    notes_block = (
        f"<case_notes>\n{case['notes']}\n</case_notes>\n\n" if case.get("notes") else ""
    )
    return (
        f"Question: {case['question']}\n\n"
        f"Expected answer: {case['expected_answer']}\n\n"
        f"{notes_block}"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"Assistant's final answer:\n{result['answer']}"
    )


def judge(client, case, result, cite=False):
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=JUDGE_SYSTEM + (CITATION_RUBRICS if cite else ""),
        output_config={"format": JUDGE_SCHEMA_CITE if cite else JUDGE_SCHEMA},
        messages=[{"role": "user", "content": build_judge_prompt(case, result)}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def run_case(case, cite=False):
    client = anthropic.Anthropic()
    try:
        result = answer(case["question"], client=client, cite=cite)
    except Exception as e:  # network/API failure: record, don't kill the run
        return {**case, "error": f"{type(e).__name__}: {e}"}
    try:
        j = judge(client, case, result, cite=cite)
    except Exception as e:
        return {**case, "error": f"judge: {type(e).__name__}: {e}"}
    passed = bool(
        j["verdict"] == "correct"
        or (
            j["verdict"] == "honest_abstention"
            and (case["category"] == "unanswerable" or case.get("abstention_ok", False))
        )
    )
    return {
        **case,
        "searched": result["searched"],
        "queries": result["queries"],
        "tool_results": result["tool_results"],
        "answer": result["answer"],
        "num_turns": result["num_turns"],
        "usage": result["usage"],
        **{k: j[k] for k in
           ("verdict", "evidence_sufficiency", "right_article", "faithfulness",
            "unsupported_side_claims", "reason")},
        **({k: j[k] for k in CITE_JUDGE_FIELDS} if cite else {}),
        "passed": passed,
    }


def search_metrics(results):
    binary = [r for r in results if "error" not in r and r["expect_search"] != "optional"]
    should = [r for r in binary if r["expect_search"] is True]
    searched = [r for r in binary if r["searched"]]
    tp = sum(1 for r in should if r["searched"])
    return {
        "search_recall": round(tp / len(should), 3) if should else None,
        "search_precision": round(tp / len(searched), 3) if searched else None,
        "spurious_searches": [
            r["id"] for r in binary if r["searched"] and r["expect_search"] is False
        ],
        "missed_searches": [r["id"] for r in should if not r["searched"]],
    }


def retrieval_metrics(ok):
    searched = [r for r in ok if r["searched"]]
    if not searched:
        return {"n_searched_cases": 0}
    suff = sum(1 for r in searched if r["evidence_sufficiency"] == "sufficient")
    right = sum(1 for r in searched if r["right_article"] == "yes")
    unfaithful = [r["id"] for r in searched if r["faithfulness"] == "unfaithful"]
    return {
        "n_searched_cases": len(searched),
        "evidence_sufficiency_rate": round(suff / len(searched), 3),
        "right_article_rate": round(right / len(searched), 3),
        "unfaithful_ids": unfaithful,
    }


def tool_error_metrics(ok):
    """Code-level count of failed tool calls (SEARCH ERROR results), so
    infra failures are separable from judged retrieval quality."""
    calls = sum(len(r["tool_results"]) for r in ok)
    errors = sum(
        sum(1 for tr in r["tool_results"] if tr.startswith("SEARCH ERROR:"))
        for r in ok
    )
    return {
        "tool_calls": calls,
        "tool_errors": errors,
        "tool_error_rate": round(errors / calls, 3) if calls else None,
    }


def multihop_metrics(ok):
    """Chain execution: did multi_hop cases issue at least the number of
    searches their chain minimally requires (min_searches per case)?"""
    mh = [r for r in ok if r["category"] == "multi_hop"]
    if not mh:
        return {}
    under = [
        r["id"]
        for r in mh
        if r["expect_search"] is True and len(r["queries"]) < r.get("min_searches", 1)
    ]
    return {
        "multi_hop_underchained": under,
        "mean_searches_multi_hop": round(
            sum(len(r["queries"]) for r in mh) / len(mh), 2
        ),
    }


def citation_metrics(ok):
    """Cite-mode only: coverage of retrieved-text claims and validity of
    every citation. invented_citation_ids is the metric that must stay
    empty — a citation on a memory-derived claim is the failure this mode
    is designed to prevent."""
    graded = [r for r in ok if r.get("citation_coverage")]
    if not graded:
        return {}
    applicable = [r for r in graded if r["citation_coverage"] != "not_applicable"]
    complete = sum(1 for r in applicable if r["citation_coverage"] == "complete")
    return {
        "citation_coverage_complete_rate": (
            round(complete / len(applicable), 3) if applicable else None
        ),
        "partial_citation_ids": [
            r["id"] for r in graded if r["citation_coverage"] == "partial"
        ],
        "uncited_answer_ids": [
            r["id"] for r in graded if r["citation_coverage"] == "none"
        ],
        "misattributed_citation_ids": [
            r["id"] for r in graded if r["citation_support"] == "misattributed"
        ],
        "invented_citation_ids": [
            r["id"] for r in graded if r["citation_support"] == "invented"
        ],
    }


def side_claim_metrics(ok):
    """Hallucination coverage beyond the key claim: side details asserted
    with no support in the retrieved evidence (judge-v4 rubric 5). Graded
    over searched cases only — unsearched answers are legitimately from
    memory and are governed by verdict/faithfulness."""
    graded = [
        r for r in ok
        if r["searched"] and isinstance(r.get("unsupported_side_claims"), list)
    ]
    if not graded:
        return {}
    with_claims = [r["id"] for r in graded if r["unsupported_side_claims"]]
    return {
        "answers_with_unsupported_side_claims": f"{len(with_claims)}/{len(graded)}",
        "unsupported_side_claim_ids": with_claims,
        "unsupported_side_claims_total": sum(
            len(r["unsupported_side_claims"]) for r in graded
        ),
    }


def abstention_metrics(ok):
    unans = [r for r in ok if r["category"] == "unanswerable"]
    answerable = [r for r in ok if r["category"] != "unanswerable"]
    return {
        "abstained_on_unanswerable": (
            f"{sum(1 for r in unans if r['verdict'] == 'honest_abstention')}/{len(unans)}"
            if unans else None
        ),
        "wrong_abstention_ids": [
            r["id"]
            for r in answerable
            if r["verdict"] == "honest_abstention" and not r.get("abstention_ok")
        ],
    }


def rejudge(src_dir: str):
    """Replay ONLY the judge over a completed run's saved agent transcripts.

    Agent outputs (queries, tool results, answers, usage, lexical) are
    frozen; every judge field is regraded under the current judge version.
    Writes a new run dir '<timestamp>_<prompt_version>_rejudge' and appends
    a history row with 'rejudged_from' set. Cases are judged against the
    notes stored in the source run's case files (not the current
    cases.jsonl), so historical grades stay tied to their era's case
    definitions.
    """
    src = Path(src_dir)
    orig_summary = json.loads((src / "summary.json").read_text())
    cite = orig_summary.get("mode") == "cite"
    olds = [
        json.loads(p.read_text()) for p in sorted((src / "cases").glob("*.json"))
    ]
    client = anthropic.Anthropic()

    def one(old):
        if "error" in old:
            return old
        try:
            j = judge(client, old, old, cite=cite)  # old carries both case + result fields
        except Exception as e:
            return {**old, "error": f"judge: {type(e).__name__}: {e}"}
        passed = bool(
            j["verdict"] == "correct"
            or (
                j["verdict"] == "honest_abstention"
                and (old["category"] == "unanswerable" or old.get("abstention_ok", False))
            )
        )
        judge_fields = (
            "verdict", "evidence_sufficiency", "right_article", "faithfulness",
            "unsupported_side_claims", "reason",
        ) + (CITE_JUDGE_FIELDS if cite else ())
        return {**old, **{k: j[k] for k in judge_fields}, "passed": passed}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, olds))

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / f"{timestamp}_{orig_summary['prompt_version']}_rejudge"
    (run_dir / "cases").mkdir(parents=True)
    for r in results:
        (run_dir / "cases" / f"{r['id']}.json").write_text(json.dumps(r, indent=2))

    by_cat = {}
    for r in ok:
        by_cat.setdefault(r["category"], []).append(r)

    summary = {
        "timestamp": timestamp,
        "prompt_version": orig_summary["prompt_version"],
        "mode": orig_summary.get("mode", "default"),
        "rejudged_from": src.name,
        "agent_model": orig_summary["agent_model"],
        "judge_model": JUDGE_MODEL,
        "judge_version": JUDGE_VERSION_CITE if cite else JUDGE_VERSION_DEFAULT,
        "n_cases": len(results),
        "n_errors": len(errors),
        "pass_rate": round(sum(r["passed"] for r in ok) / len(ok), 3) if ok else None,
        "pass_rate_by_category": {
            cat: f"{sum(r['passed'] for r in rs)}/{len(rs)}"
            for cat, rs in sorted(by_cat.items())
        },
        **search_metrics(results),
        **retrieval_metrics(ok),
        **tool_error_metrics(ok),
        **multihop_metrics(ok),
        **side_claim_metrics(ok),
        **citation_metrics(ok),
        **abstention_metrics(ok),
        "failed_ids": [r["id"] for r in ok if not r["passed"]],
        "error_ids": [r["id"] for r in errors],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (RUNS_DIR / "history.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nRejudge artifacts: {run_dir}")


def main():
    argv = sys.argv[1:]
    if "--rejudge" in argv:
        rejudge(argv[argv.index("--rejudge") + 1])
        return
    cite = "--cite" in argv
    only_ids = {a for a in argv if not a.startswith("--")}
    cases = [
        json.loads(line)
        for line in (EVAL_DIR / "cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if only_ids:
        cases = [c for c in cases if c["id"] in only_ids]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / f"{timestamp}_{PROMPT_VERSION}{'_cite' if cite else ''}"
    (run_dir / "cases").mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda c: run_case(c, cite=cite), cases))

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    # Lexical/semantic sidecar metrics (batch: model load dominates)
    lex = lexical_metrics.compute_batch(
        [(r["id"], r["answer"], r["expected_answer"]) for r in ok]
    )
    for r in ok:
        r["lexical"] = lex.get(r["id"], {})

    for r in results:
        (run_dir / "cases" / f"{r['id']}.json").write_text(json.dumps(r, indent=2))

    by_cat = {}
    for r in ok:
        by_cat.setdefault(r["category"], []).append(r)

    summary = {
        "timestamp": timestamp,
        "prompt_version": PROMPT_VERSION,
        "mode": "cite" if cite else "default",
        "agent_model": AGENT_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_version": JUDGE_VERSION_CITE if cite else JUDGE_VERSION_DEFAULT,
        "n_cases": len(cases),
        "n_errors": len(errors),
        "pass_rate": round(sum(r["passed"] for r in ok) / len(ok), 3) if ok else None,
        "pass_rate_by_category": {
            cat: f"{sum(r['passed'] for r in rs)}/{len(rs)}"
            for cat, rs in sorted(by_cat.items())
        },
        **search_metrics(results),
        **retrieval_metrics(ok),
        **tool_error_metrics(ok),
        **multihop_metrics(ok),
        **side_claim_metrics(ok),
        **citation_metrics(ok),
        **abstention_metrics(ok),
        "mean_searches_per_case": round(
            sum(len(r["queries"]) for r in ok) / len(ok), 2
        ) if ok else None,
        "total_tokens": {
            "input": sum(r["usage"]["input_tokens"] for r in ok),
            "output": sum(r["usage"]["output_tokens"] for r in ok),
        },
        "lexical_metric_availability": lexical_metrics.AVAILABILITY,
        "failed_ids": [r["id"] for r in ok if not r["passed"]],
        "error_ids": [r["id"] for r in errors],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    RUNS_DIR.mkdir(exist_ok=True)
    with (RUNS_DIR / "history.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nRun artifacts: {run_dir}")

    failures = [r for r in ok if not r["passed"]] + errors
    if failures:
        print(f"\n===== {len(failures)} FAILING CASES =====")
        for r in failures:
            print(f"\n--- {r['id']} ({r['category']}) ---")
            print(f"Q: {r['question']}")
            if "error" in r:
                print(f"ERROR: {r['error']}")
                continue
            print(f"expect_search={r['expect_search']}  searched={r['searched']}")
            for i, q in enumerate(r["queries"]):
                print(f"  [search {i + 1}] {q!r}")
            print(f"A: {r['answer']}")
            print(
                f"verdict={r['verdict']}  evidence={r['evidence_sufficiency']}  "
                f"article={r['right_article']}  faithful={r['faithfulness']}"
            )
            if "citation_coverage" in r:
                print(
                    f"citations: coverage={r['citation_coverage']}  "
                    f"support={r['citation_support']}"
                )
            print(f"judge: {r['reason']}")


if __name__ == "__main__":
    main()

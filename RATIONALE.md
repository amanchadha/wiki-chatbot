# Design Rationale

**Approach in one line:** treat this as loop engineering, not prompt
authoring — build a deliberately simple v0, build an eval that decomposes
failures into decision → retrieval → generation stages, and let clustered
eval failures pick every iteration target.

**Time spent:** ~[X] hours. *(fill in before submission)*

**Models:** agent `claude-opus-4-8` (adaptive thinking, effort low); judge
`claude-sonnet-5` — a different model to avoid self-grading bias.

---

## 1. Prompt-engineering approach

The system prompt was never hand-tuned in the abstract; every change is a
diff to `prompts.py` motivated by a clustered eval failure, tagged with a
version, and verified by rerunning the suite. Key design decisions:

- **v0 was intentionally minimal** — "search when you're not certain" — to
  measure the model's native calibration rather than assume it's broken.
  The baseline proved it *is* broken in a specific way: Opus 4.8's
  *confidence* is miscalibrated on volatile/specific facts (it felt certain
  about a stale Saturn moon count, and about a Mercury Records founder it
  had wrong).
- **v1 keyed the search decision on the *kind* of fact, not felt
  confidence** — a closed list of verify-required shapes (current
  officeholders, counts/statistics, named specifics) against a closed list
  of answer-directly categories (arithmetic, language, stable common
  knowledge, creative, not-in-Wikipedia). Deliberately symmetric — no
  "CRITICAL/MUST" asymmetry — because Opus 4.8 follows instructions
  literally and an aggressive push toward the tool would have destroyed
  precision. Result: recall 0.33 → 0.93 with precision staying 1.0 and zero
  spurious searches.
- **Evidence discipline came from the same failure family**: "your answer
  must come from the retrieved text; re-search or say the search did not
  confirm it" (v1), an explicit outage policy — don't guess on volatile
  facts, labeled-unverified memory OK for stable facts (v1.1), and per-hop
  verification for chained questions after multi-hop cases showed the model
  verifying hop 1 and back-filling hop 2 from memory (v1.3).
- **Two of the five fixes were tool fixes, not prompt fixes.** Plain-text
  extracts strip tables and infoboxes, so award winners and settlement
  populations were structurally invisible; search snippets (v1.2) and
  infobox extraction (v1.3) fixed retrieval classes no prompt wording could
  reach. Knowing *which* layer to fix is exactly what the staged metrics
  bought us.

## 2. Eval design

Three graded dimensions per case, so each failure is attributable to a
stage: **search decision** (pure code: recall/precision vs a three-valued
`expect_search`, with "optional" cases excluded), **retrieval quality**
(judge: evidence sufficiency, right-article; plus code-level tool-error rate
so infra failures never masquerade as retrieval failures), and **answer
quality** (judge: verdict, faithfulness, abstention appropriateness).

Case design choices that mattered:

- **Stale-memory traps** (current officeholders/CEOs changed in 2024–2025)
  where the *anticipated failure* is a confident wrong answer — these
  separate "confident" from "correct", which a generic obscure-facts set
  cannot.
- **"Optional" is a deliberate third value, not indecision.** Cases like
  "Who painted the Mona Lisa?" sit exactly on the boundary of the prompt's
  two decision lists: they are stable, pre-cutoff common knowledge (the
  answer-directly list) *and* literally "named specifics about a particular
  work" (the search-first list). Since either behavior is defensible, these
  cases are excluded from search precision and recall in both directions —
  punishing either choice would make the eval arbitrary — while answer
  correctness (and, if the model does search, right-article and
  faithfulness) is still graded. In practice the model answers all four
  optional cases from memory, correctly, in every final run: the useful
  thing these cases demonstrate is calibrated restraint — not burning a
  search on a fact that pattern-matches the trigger list but is bedrock
  knowledge — which is the live face of the suite's 1.0 search precision.
- **Forced disambiguation** (Mercury Records vs the planet, Odessa TX vs
  Odesa UA, Java SD vs the language) to exercise wrong-article retrieval.
- **Unanswerables** where honest abstention is the pass condition, graded
  separately in both directions (abstained-when-unanswerable, and wrong
  abstentions on answerable cases).
- **Multi-hop** (added after single-hop converged): chained searches with
  per-case `min_searches`, two chains with intermediates too obscure to
  skip from memory, and one dead-end chain where the right answer is "not
  recorded" (`abstention_ok`).
- **`notes` are authoritative over the judge's own knowledge** — this is
  what makes trap grading reliable — and the judge must flag (not act on)
  disagreement with an expected answer. Judge prompt changes are
  version-tagged (judge-v1 → judge-v5; see the README for the numbering,
  which spans both the default and citation grading modes) for cross-run
  comparability, and a `--rejudge` mode replays a newer judge over a
  finished run's frozen transcripts — grading improvements never require
  re-running the agent. A 10-case blind human spot-check, selected before
  any verdicts existed, matched the judge 10/10.
- **Lexical metrics were run as a falsifiable experiment, not assumed
  useful.** BLEU/ROUGE were dropped on evidence: the baseline's one
  incorrect answer scored *above* the correct-group mean on ROUGE-1
  (fluent-wrong overlaps a short factual reference as well as right), and
  correct answers spanned 0.00–0.76, so no threshold can separate the
  groups. BERTScore is deferred (dev-network download failure; recomputable
  offline from saved artifacts), MoverScore was unbuildable.

## 3. Run history and key iterations

| Run | Config | Pass | Search recall / precision | What it taught us |
|---|---|---|---|---|
| 1 | v0, pre-rubric judge (superseded) | 0.94 | 0.33 / 1.0 | First recall signal; judge too shallow to attribute failures |
| 2 | v0 baseline, rubric judge | 0.969 | 0.33 / 1.0 | Systematic under-searching masked by lucky memory (9 latent passes); pass rate alone is misleading |
| 3 | v1 verify-volatile | 0.969 | 0.93 / 1.0 | Fact-kind trigger works; run polluted by 52% Wikipedia 429s → infra is a failure mode too |
| 4 | v1.1 outage-fallback + tool retries | 0.969 | 0.87 / 1.0 | Clean read; two genuine failures left: table-stripped extracts (r12), alternate-title shortcut (r02) |
| 5 | v1.2 snippets + follow-up clause | **1.0** (32/32) | 0.87 / 1.0 | Single-hop converged; verification now costs no more tokens than v0 |
| 6 | v1.2 + 8 multi-hop cases (baseline) | 0.975 | 0.955 / 1.0 | Chain collapse into memory (mh-04) and infobox-blind retrieval (mh-06); single-hop unregressed |
| 7 | v1.3 chain-verify + infobox | 0.975 | 0.955 / 1.0 | Multi-hop 8/8, zero unfaithful; one miss (r09) traced to Wikipedia's own article contradicting itself (prose 1906 vs infobox 1907) |
| 8 | v1.3 confirmation (identical config) | **1.0** (40/40) | 0.955 / 1.0 | Run-to-run spread ≈ 0 on all metrics |
| 9–10 | judge-v4 re-judge of runs 7–8 (frozen transcripts, no agent re-run) | unchanged | unchanged | New side-claims hallucination rubric: 1 unsupported side detail per run (~5% of searched answers); every judge-v2-era grade reproduced exactly |
| 11 | v1.4 `--cite` baseline (+ judge-v5 re-judge) | **1.0** (40/40) | 0.955 / 1.0 | Citation coverage 1.0, **zero invented/misattributed citations**; default-mode metrics identical (cite instructions append only under the flag); side claims dropped to 0/21 — requiring a marker per claim appears to suppress memory embellishment |

(A separate partial run, aborted by an API-credit outage, is flagged invalid
in `history.jsonl`.)

**Transparency note on the final 40/40:** run 8's perfect score came after
two *eval-side* calibrations made between runs 7 and 8 — (a) r09's grading
notes now accept either 1906 or 1907 because the Wikipedia article itself
states both (our infobox feature exposed the discrepancy; the model's answer
was faithful to retrieved evidence), and (b) a metric labeling fix so the
by-design dead-end abstention isn't counted as a wrong abstention. The agent
configuration was byte-identical across runs 7 and 8; without the r09
calibration, run 8 would read 39/40 with the one miss being a
faithful-to-evidence answer to an internally inconsistent source. We changed
the ruler, not the system — and are saying so.

## 4. Where it succeeds, where it fails, what we learned

**Succeeds:** search-decision calibration (recall 0.96, precision 1.0 on
every run of the project); disambiguation (right-article 1.0 once actually
exercised); evidence grounding (zero faithfulness violations in the final
config); honest abstention on unanswerables and dead-end chains; multi-hop
chaining including memory-proof intermediates; citation mode (opt-in
`--cite`): complete coverage with zero invented citations on its baseline,
and — a design finding worth keeping — the citation requirement eliminated
the residual unsupported side claims (0/21 vs ~1 per default-mode run),
because a claim that needs a marker to justify itself doesn't get written
from memory.

**Known residual failures, consciously accepted:**

- r08-class: stable decades-old historical years are sometimes answered
  from memory (correctly) despite the "specific years" trigger — a ±1-case
  noise band we kept, because tightening the wording risks the precision
  that never dropped.
- Unsupported side details on otherwise-grounded answers, measured at ~5%
  of searched answers by the judge-v4 hallucination rubric — small memory
  embellishments like a "(France/PSG)" affiliation or a "(12 August 1970)"
  day-and-month where evidence gave only the year. True facts in both
  observed instances, but unverified specifics of exactly the class that
  eventually goes wrong. Tracked in every run; a one-line prompt remedy is
  ready if the rate climbs on a larger case set.
- Facts deep in very long articles can fall below the 6k-char extract
  cut; snippets and infoboxes cover the common cases (tables, infoboxes)
  but not all of them.
- Live-Wikipedia ground truth drifts (and is occasionally
  self-contradictory, per r09) — the eval mitigates with
  grade-against-evidence notes, not immunity.

**Biggest lessons:** (1) pass rate without staged metrics is actively
misleading — the v0 run "passed" 97% while silently skipping verification on
two-thirds of the facts that needed it; (2) the model's confidence is not a
usable search trigger, but fact-shape is; (3) a meaningful share of "prompt
problems" were retrieval-layer problems (rate limits, stripped tables and
infoboxes) that no prompt could fix; (4) grade the grader — version-tagged
judge changes plus judge-only replay over frozen transcripts let
hallucination coverage be added retroactively, and measuring that residual
(~5%) *before* prompting against it kept us from trading answer quality for
a two-claim problem.

## 5. With more time

- Expand the case set (the honest next step once a 40-case suite saturates);
  multiple eval repetitions per config for confidence intervals.
- Recompute BERTScore offline and finish the semantic-metric verdict.
- Query rewriting and >2-hop planning (deferred from v0 by design; the eval
  now exists to justify them the moment cases demand it).
- Decide whether citation mode becomes the default: its eval gate has now
  been passed (coverage 1.0, zero invented citations, side claims
  suppressed to zero), so the remaining question is product taste — the
  ~10% token overhead and the visual weight of markers — plus validation on
  a larger case set before flipping the default.
- A second human grading pass on the full 40 cases to bound judge error, and
  an agent-model ablation (Sonnet/Haiku as agent) on the same suite.
- Automated prompt optimization, using
  APO ([Pryzant et al., 2023](https://arxiv.org/abs/2305.03495)) and
  AlphaEvolve-style prompt evolution
  ([Novikov et al., 2025](https://arxiv.org/abs/2506.13131)). I've used
  AlphaEvolve to evolve an optimized prompt with its evolutionary approach,
  and APO's textual-feedback loop is similarly good at refining prompts. We
  should run a comparative analysis against our manual loop-engineering
  method with the same eval suite.
- Multilingual coverage — all our cases are English right now.
- A failure-analysis dashboard: the per-case transcripts are already
  structured, so slicing failures by category and prompt version is easy.
- Domain-specific evals for specialized areas like healthcare, legal, and
  finance, where the bar for faithfulness and for knowing when to say "I
  don't know" should be higher.

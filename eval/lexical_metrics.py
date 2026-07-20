"""Lexical and semantic similarity metrics: BLEU, ROUGE, BERTScore, MoverScore.

Experimental sidecar to the LLM judge. Each metric compares the agent's final
answer against the case's expected_answer. Computed per case; after the
baseline run we check per-metric agreement with judge verdicts and make a
keep/drop call for each.

Every metric degrades gracefully: if its library is missing or broken, the
score is None and the failure is recorded in AVAILABILITY (rather than
killing the eval run).
"""

from typing import Optional

AVAILABILITY: dict[str, str] = {}

try:
    import sacrebleu

    AVAILABILITY["bleu"] = "ok"
except Exception as e:  # pragma: no cover
    sacrebleu = None
    AVAILABILITY["bleu"] = f"unavailable: {type(e).__name__}: {e}"

try:
    from rouge_score import rouge_scorer

    _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    AVAILABILITY["rouge"] = "ok"
except Exception as e:  # pragma: no cover
    _ROUGE = None
    AVAILABILITY["rouge"] = f"unavailable: {type(e).__name__}: {e}"

try:
    import bert_score as _bert_score

    AVAILABILITY["bertscore"] = "ok"
except Exception as e:  # pragma: no cover
    _bert_score = None
    AVAILABILITY["bertscore"] = f"unavailable: {type(e).__name__}: {e}"

# MoverScore disabled: its import-time model download exceeded a 180s timeout
# on this machine (2026-07-19). BERTScore covers the semantic-similarity angle.
get_idf_dict = word_mover_score = None
AVAILABILITY["moverscore"] = "unavailable: import-time model download timed out (>180s); disabled"


def _bleu(candidate: str, reference: str) -> Optional[float]:
    if sacrebleu is None:
        return None
    # sentence_bleu returns 0-100; normalize to 0-1 for comparability
    return round(sacrebleu.sentence_bleu(candidate, [reference]).score / 100, 4)


def _rouge(candidate: str, reference: str) -> dict:
    if _ROUGE is None:
        return {"rouge1_f": None, "rougeL_f": None}
    scores = _ROUGE.score(reference, candidate)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def compute_batch(pairs: list[tuple[str, str, str]]) -> dict[str, dict]:
    """pairs: list of (case_id, candidate_answer, reference_answer).

    Returns {case_id: {bleu, rouge1_f, rougeL_f, bertscore_f1, moverscore}}.
    BERTScore and MoverScore are batched (model load dominates per-call cost).
    """
    out: dict[str, dict] = {}
    ids = [p[0] for p in pairs]
    cands = [p[1] for p in pairs]
    refs = [p[2] for p in pairs]

    for cid, cand, ref in pairs:
        out[cid] = {"bleu": _bleu(cand, ref), **_rouge(cand, ref)}

    if _bert_score is not None and pairs:
        try:
            _, _, f1 = _bert_score.score(
                cands, refs, lang="en", rescale_with_baseline=False, verbose=False
            )
            for cid, f in zip(ids, f1.tolist()):
                out[cid]["bertscore_f1"] = round(f, 4)
        except Exception as e:
            AVAILABILITY["bertscore"] = f"failed at runtime: {type(e).__name__}: {e}"
            for cid in ids:
                out[cid]["bertscore_f1"] = None
    else:
        for cid in ids:
            out[cid]["bertscore_f1"] = None

    if word_mover_score is not None and pairs:
        try:
            idf_refs = get_idf_dict(refs)
            idf_cands = get_idf_dict(cands)
            scores = word_mover_score(
                refs, cands, idf_refs, idf_cands,
                stop_words=[], n_gram=1, remove_subwords=True,
            )
            for cid, s in zip(ids, scores):
                out[cid]["moverscore"] = round(float(s), 4)
        except Exception as e:
            AVAILABILITY["moverscore"] = f"failed at runtime: {type(e).__name__}: {e}"
            for cid in ids:
                out[cid]["moverscore"] = None
    else:
        for cid in ids:
            out[cid]["moverscore"] = None

    return out

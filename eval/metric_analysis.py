"""Agreement analysis: lexical/semantic metrics vs LLM-judge verdicts.

Usage:
    python eval/metric_analysis.py [run_dir]   # defaults to the latest run

For each metric (bleu, rouge1_f, rougeL_f, bertscore_f1, moverscore):
  * mean score per judge verdict group
  * AUC separating correct vs not-correct (incorrect + partially_correct),
    computed rank-based (Mann-Whitney); 0.5 = no signal, 1.0 = perfect
  * overlap check: the range of scores in each group, to show whether any
    threshold could separate them

honest_abstention cases are reported as their own group but excluded from
the AUC (an abstention has no meaningful lexical match to the reference and
would conflate "honest" with "wrong").
"""

import json
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"
METRICS = ["bleu", "rouge1_f", "rougeL_f", "bertscore_f1", "moverscore"]


def auc(pos: list, neg: list) -> float:
    """Rank-based AUC: P(random pos > random neg), ties count half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir())[-1]
    print(f"Analyzing: {run_dir.name}\n")

    cases = [
        json.loads(p.read_text())
        for p in sorted((run_dir / "cases").glob("*.json"))
    ]
    cases = [c for c in cases if "error" not in c and c.get("lexical")]

    groups: dict[str, list] = {}
    for c in cases:
        groups.setdefault(c["verdict"], []).append(c)

    print(f"{len(cases)} judged cases: " +
          ", ".join(f"{v}={len(cs)}" for v, cs in sorted(groups.items())))

    correct = groups.get("correct", [])
    wrong = groups.get("incorrect", []) + groups.get("partially_correct", [])

    for metric in METRICS:
        vals = {
            v: [c["lexical"][metric] for c in cs if c["lexical"].get(metric) is not None]
            for v, cs in groups.items()
        }
        if not any(vals.values()):
            print(f"\n{metric}: unavailable (all None)")
            continue
        print(f"\n{metric}:")
        for v in ("correct", "partially_correct", "incorrect", "honest_abstention"):
            xs = vals.get(v, [])
            if xs:
                print(
                    f"  {v:20s} n={len(xs):2d}  mean={sum(xs)/len(xs):.3f}  "
                    f"min={min(xs):.3f}  max={max(xs):.3f}"
                )
        pos = [c["lexical"][metric] for c in correct if c["lexical"].get(metric) is not None]
        neg = [c["lexical"][metric] for c in wrong if c["lexical"].get(metric) is not None]
        if pos and neg:
            print(f"  AUC (correct vs incorrect+partial): {auc(pos, neg):.3f}")
        else:
            print("  AUC: not computable (a group is empty)")


if __name__ == "__main__":
    main()

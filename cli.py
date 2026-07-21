"""CLI for the wiki chatbot.

Usage:
    python cli.py "Who discovered the structure of DNA?"   # one-shot
    python cli.py                                          # interactive
    python cli.py --demo                                   # sample queries
    python cli.py --cite "..."                             # citation mode:
        inline superscript markers on retrieved-text claims + source list
        (off by default; combines with any mode above)
"""

import sys

from agent import answer

# One sample per behavior the system is engineered for, ordered by query
# complexity (simplest first). The Vercel demo (index.html) shows the same
# set in the same order — keep the two lists in sync. The local web UI
# (web.py) draws a larger set from eval/cases.jsonl with the same ordering.
DEMO_SECTIONS = [
    ("Direct answer", "no lookup needed — arithmetic, language, common knowledge",
     "What is 17 * 24?"),
    ("Optional lookup", "stable fact the model may verify or answer directly",
     "Who painted the Mona Lisa?"),
    ("Single-hop retrieval", "volatile fact that must be verified (stale-memory trap)",
     "Who is the current president of Botswana?"),
    ("Disambiguation", "ambiguous entity — retrieval must pick the right article",
     "Who founded Mercury Records?"),
    ("Multi-hop chain", "find an entity, then a fact about that entity",
     "What is the population of the city where the current Secretary General of NATO was born?"),
    ("False premise", "the correct answer rejects the question's assumption",
     "When did Einstein win his second Nobel Prize?"),
    ("Unanswerable", "no recorded answer exists — honesty is correct",
     "What did Napoleon eat for breakfast on his 30th birthday?"),
]


def run_one(question: str, cite: bool = False) -> None:
    print(f"\nQ: {question}")
    result = answer(question, cite=cite)
    for i, (query, _) in enumerate(zip(result["queries"], result["tool_results"])):
        print(f"  [search {i + 1}] {query!r}")
    if not result["searched"]:
        print("  [no search — answered from model knowledge]")
    print(f"\nA: {result['answer']}\n")


def main() -> None:
    args = sys.argv[1:]
    cite = "--cite" in args
    args = [a for a in args if a != "--cite"]
    if args == ["--demo"]:
        for title, blurb, q in DEMO_SECTIONS:
            print(f"\n=== {title} — {blurb} ===")
            run_one(q, cite=cite)
    elif args:
        run_one(" ".join(args), cite=cite)
    else:
        print("Wiki chatbot — ask a question (Ctrl-D or 'quit' to exit)")
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question or question.lower() in {"quit", "exit"}:
                break
            run_one(question, cite=cite)


if __name__ == "__main__":
    main()

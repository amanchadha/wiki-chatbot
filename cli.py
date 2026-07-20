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

# One sample per behavior the system is engineered for. The Vercel demo
# (index.html) shows the same set — keep the two lists in sync.
DEMO_QUESTIONS = [
    "Who is the current president of Botswana?",       # search: stale-memory trap
    "What is 17 * 24?",                                # no search: arithmetic
    "Who founded Mercury Records?",                    # search: ambiguous entity
    "What is the population of the city where the current Secretary General of NATO was born?",  # multi-hop chain
    "When did Einstein win his second Nobel Prize?",   # false premise
    "What did Napoleon eat for breakfast on his 30th birthday?",  # unanswerable
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
        for q in DEMO_QUESTIONS:
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

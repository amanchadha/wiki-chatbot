"""CLI for the wiki chatbot.

Usage:
    python cli.py "Who discovered the structure of DNA?"   # one-shot
    python cli.py                                          # interactive
    python cli.py --demo                                   # sample queries
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


def run_one(question: str) -> None:
    print(f"\nQ: {question}")
    result = answer(question)
    for i, (query, _) in enumerate(zip(result["queries"], result["tool_results"])):
        print(f"  [search {i + 1}] {query!r}")
    if not result["searched"]:
        print("  [no search — answered from model knowledge]")
    print(f"\nA: {result['answer']}\n")


def main() -> None:
    args = sys.argv[1:]
    if args == ["--demo"]:
        for q in DEMO_QUESTIONS:
            run_one(q)
    elif args:
        run_one(" ".join(args))
    else:
        print("Wiki chatbot — ask a question (Ctrl-D or 'quit' to exit)")
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question or question.lower() in {"quit", "exit"}:
                break
            run_one(question)


if __name__ == "__main__":
    main()

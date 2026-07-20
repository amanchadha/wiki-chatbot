"""Agent loop: Claude + search_wikipedia tool, manual tool-use loop.

answer(question) returns structured metadata alongside the answer text so
the eval harness can consume search behavior directly:

    {
        "answer": str,
        "searched": bool,
        "queries": [str],          # every search query issued, in order
        "tool_results": [str],     # corresponding tool result strings
        "num_turns": int,          # API round trips
        "usage": {"input_tokens": int, "output_tokens": int},
    }
"""

from typing import Optional

import anthropic

from prompts import CITATION_INSTRUCTIONS, SYSTEM_PROMPT, WIKIPEDIA_TOOL
from wikipedia import search_wikipedia

MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000
MAX_TURNS = 5  # hard cap on API round trips per question


def answer(
    question: str,
    client: Optional[anthropic.Anthropic] = None,
    cite: bool = False,
) -> dict:
    client = client or anthropic.Anthropic()
    system_text = SYSTEM_PROMPT + ("\n\n" + CITATION_INSTRUCTIONS if cite else "")
    messages = [{"role": "user", "content": question}]

    queries: list[str] = []
    tool_results: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    response = None

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[WIKIPEDIA_TOOL],
            messages=messages,
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        result_blocks = []
        for block in response.content:
            if block.type == "tool_use":
                query = block.input["query"]
                result = search_wikipedia(query)
                queries.append(query)
                tool_results.append(result)
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
        messages.append({"role": "user", "content": result_blocks})

    final_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return {
        "answer": final_text,
        "searched": bool(queries),
        "queries": queries,
        "tool_results": tool_results,
        "num_turns": len(queries) + 1,
        "usage": usage,
    }

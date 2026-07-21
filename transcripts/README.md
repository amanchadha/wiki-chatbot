# Claude Code transcripts

Raw Claude Code session logs (`.jsonl`, one file per session — 62 sessions,
including short subagent sessions) covering the full development of this
project, exported from Claude Code's per-project transcript directory.

The sessions cover, in order: v0 plan + build, eval-harness design + judge
prompt review, the baseline run and iterations v1 → v1.3 (including the
multi-hop extension), the judge-v4/v5 hallucination + citation grading work,
citation mode, the web UIs, and the docs/packaging pass. Every eval run
referenced in the transcripts has its artifacts checked in under
`eval/runs/`.

Notes:

- Scanned for secrets before committing: the only `sk-ant-*` / `ghp_*`
  matches are truncated documentation placeholders inside embedded skill
  docs, not real credentials.
- Files are ordered by modification time, not name; to follow the work
  chronologically, sort by timestamp (`ls -ltr`).

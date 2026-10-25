# case-triage-agent

Agent that takes a public kubernetes/kubernetes issue URL, investigates it with three MCP tools, and returns a triage note (likely cause, cited docs, next step). "Can't tell" is a valid, measured outcome. Portfolio project, built on macOS, repo at ~/case-triage-agent.

Read first: docs/scope.md (v3) and docs/evening-1-guide.md. The scope is the source of truth. If code and scope disagree, stop and ask; do not quietly change either.

## Hard rules

1. **Secrets and data.** Never commit .env, tokens, or data/. Run `git status` before every commit. GITHUB_TOKEN is read-only, public repos, environment variable only.
2. **Leakage rule.** get_issue_thread returns the issue body and human comments only up to the first maintainer reply (maintainer = OWNER/MEMBER/COLLABORATOR, not the issue author, not a bot). The tests for this must keep passing. Never loosen the rule to make a result look better.
3. **Frozen contracts.** triage/contracts.py defines tool output shapes, including the error dict keys (`timeout`, `rate_limited`, `not_found`, `bad_url`, ...). Failure tests assert on these. Changing a key or shape needs my approval.
4. **The labelled eval set is mine.** Labels may be drafted by a model into data/drafts/. Only I promote reviewed labels into data/eval/. Never write data/eval/ yourself. Never use the 40 eval issues to tune the agent's prompts.
5. **No live paid calls in loops.** Live model calls (any model) only when I ask for a live run, and always under a per-run and a per-day cost cap. CI uses recorded fixtures, never live calls.
6. **Honest reporting.** Every report says what was RUN (command and result) versus what was only WRITTEN. Code that has never touched the real GitHub API, rag-hybrid or the model counts as unrun. Do not write "works" or "done" for unrun code.
7. **Claims.** README and docs use only the allowed claim in the scope. Never write "production" or "users" unless true. Report counts next to percentages. If the agent doesn't beat the baseline, the write-up says so.
8. **Retrieval backends.** RETRIEVAL_BACKEND=es|inproc behind the same /search interface. The 40-case eval and the baseline run on es. inproc is the deployed build and gets a parity run on rag-hybrid's 26 answerable questions.
9. **rag-hybrid is not exposed.** It has no auth, TLS or rate limiting and Elasticsearch security is off. Only this service talks to it, on localhost.

## Stack and commands

- Python venv in .venv, deps pinned in requirements.txt. MCP SDK 2.x: `from mcp.server.mcpserver import MCPServer` (it was FastMCP in 1.x). Don't upgrade across major versions without asking.
- Load env: `set -a; source .env; set +a`
- Tests: `python -m pytest -q`
- Candidates: `python -m scripts.pull_candidates --limit 100 --out data/candidates.jsonl`
- rag-hybrid API: http://127.0.0.1:8000 (start it from the rag-hybrid repo with `docker compose up -d`).
- Docker deploy image must be built for the host architecture (`--platform linux/amd64` if the host is x86; this Mac is Apple Silicon).

## Working with subagents

- Contracts first: triage/contracts.py is written, reviewed by me, and committed before any subagent starts.
- One git worktree and one branch per subagent. Each subagent owns specific files and does not edit anyone else's, or contracts.py.
- Parallel batch 1: in-process retrieval backend; agent loop; eval harness skeleton (fake cases). Batch 2 after merge: failure tests; Dockerfile and CI.
- Not delegated to any subagent: labelling the 40 issues, choosing the category set, interpreting results, README claims.
- After each part, a separate reviewer subagent with no access to the builder's reasoning checks the diff against this file and the scope, and reports findings. It does not fix them.
- Stop and report at the end of each part. Don't start the next part on your own.

## Style

Plain, small functions, no frameworks beyond what the scope names. Comments say why, not what. Tool errors are returned as dicts, not raised.

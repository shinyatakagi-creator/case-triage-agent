# case-triage-agent: scope (v3, 24 Oct 2026)

Status: planned, not started. Start after the Supabase recruiter screen on Wed 28 Oct.
Closes two tracker gaps: "Multi-step agent with several tools, MCP" (5 roles) and "Deployed AI system with real users or real data" (6 roles).
Timeline revised from 7 to 9 evenings after the feasibility review. The in-process backend (decision below) now uses one of the two extra evenings.
v3: retrieval hosting decision recorded, rag-hybrid /search request shape added, MCP SDK version note added. Evening 1 steps are in case-triage-agent-evening-1-guide.md.

## Goal
An agent that takes a public GitHub issue URL and investigates it across several tools. It returns a triage note with a likely cause, cited doc sources and a suggested next step. Runs as a deployed service on live public data.

## Existing dependency: rag-hybrid (checked 24 Oct)
- Already indexes 502 Kubernetes docs pages (kubernetes/website, pinned commit), so no extra ingest evening.
- Has FastAPI /search and /ask, a spend ledger, Docker Compose (Elasticsearch 9.5.5, API), an eval harness and a 30-question golden set (26 answerable, 4 unanswerable).
- Its README says the API has no authentication, TLS or rate limiting, and Elasticsearch security is off. Do not expose it directly. The agent service sits in front of it.
- /search request (from rag-hybrid docs/SETUP.md): POST JSON `{"query": ..., "mode": "bm25"|"dense"|"hybrid", "k": ...}`, API on 127.0.0.1:8000. The response shape is NOT documented; evening 1 normalises it to `{"passages": [{text, source, heading}]}` inside search_docs.
- The agent's search_docs tool uses /search only (retrieval). The agent does its own reasoning, so /ask (which calls Sonnet) is not used.

## Data
- Answerable pool: kubernetes/kubernetes issues labelled kind/support. A label page check on 24 Oct showed 2,419 closed or open issues with kind/support and more than 3 comments. That count includes bot comments, so the number of substantive threads is NOT verified. Evening 1: pull about 100 candidates and count how many have a real human thread.
- Unanswerable pool (about 15 cases): issues the docs cannot answer. Take them from issues labelled both kind/bug and kind/support, and from third-party problems (cloud provider, CNI, vendor images), plus some pure kind/bug and kind/feature issues. Fix the split before running anything.

## Tools (MCP server, 3 tools)
1. search_docs: calls rag-hybrid /search (BM25 + kNN with RRF), returns passages with citations.
2. get_issue_thread: pulls the issue and comments through the GitHub API. Answer leakage rule (changed 25 Oct 2026): return the issue body plus only the issue author's own comments that come before the first comment by anyone else. Bot comments (the BOT_LOGINS list) are ignored and do not end the thread. thread_truncated is true when anything was cut. Replies often contain the answer or a doc link, which would inflate "correct doc cited".
   - Why it changed: the first rule cut at the first reply by an OWNER/MEMBER/COLLABORATOR and under-detected maintainers. Example: on kubernetes/kubernetes #131977, ffromani (a SIG Node maintainer) has author_association CONTRIBUTOR and replied first, so his answer was visible to the agent. author_association is no longer used.
3. extract_error_signals: parses stack traces, error strings and versions out of the returned text.

Implementation notes: Python MCP SDK 2.x (class `MCPServer` in `mcp.server.mcpserver`; it was `FastMCP` in 1.x), version pinned in requirements.txt. Tools return errors as small dicts (`{"error": "timeout" | "rate_limited" | "not_found" | "bad_url" | ...}`) rather than raising, so the agent sees them and the failure tests can assert on them.

## Agent loop
Claude Haiku (same as LLM Cost Autopilot) chooses tools. Hard cap of 6 steps. Stops when it has a cited answer or admits it cannot produce one. "Can't tell" is a valid outcome and is measured.

## Eval
- 40 hand-labelled issues, same labelling method as Failure Forensics. Each needs a category and the correct doc page. Budget 5 to 8 minutes each (3 to 5 hours), split over two evenings.
- Mix: about 25 answerable by the docs and about 15 not answerable. Without the unanswerable group the "can't tell" rate means nothing.
- Metrics: category accuracy, correct doc cited, steps per case, cost per case, failure rate, and "can't tell" rate split by answerable vs not answerable.
- Baseline: rag-hybrid answering in one shot, no agent. Both the agent eval and the baseline run on the Elasticsearch backend.
- Small sample: with 40 cases, a gap of two or three cases is noise. Report counts next to percentages, use bootstrap intervals as rag-hybrid does, and only claim a win over the baseline if the gap is clear. If the agent does not beat the baseline, the write-up says so.
- CI gate: runs on recorded tool-output fixtures, not live model calls (live Haiku calls cost money and flake). The live eval runs manually or on a schedule.

## Failure and recovery tests
Tool timeout, GitHub rate limit (403, mocked), empty search result, malformed tool output, step cap reached. Each has a test that checks the agent retries or degrades cleanly instead of looping.

## GitHub token
Unauthenticated calls are capped at 60 per hour. Use a read-only token for public repos from day one, kept in an environment variable and never committed.

## Deployment
- Docker image on a small free or low-cost host.
- /health endpoint, structured logs for every tool call, per-day cost cap, basic rate limiting.
- The endpoint accepts only kubernetes/kubernetes issue URLs. Anything else is rejected. This stops it being used to burn the GitHub token and LLM budget.
- Shared-secret header, key shared only with people who ask.

## Decision (made 24 Oct): Option A, two retrieval backends
Correction to the v1 scope: a "bundled index" does not remove the need for Elasticsearch, because the kNN side needs a running cluster. That is heavy for a free host.

Decided: Elasticsearch stays the reference backend, and an in-process backend is the deployed one.
- Reason: a recruiter or hiring manager may click the link weeks after the project is finished. A 14-day Elastic Cloud trial would expire mid-search and leave a dead link on a project whose claim is "deployed". The in-process build costs nothing to keep running.
- Reference backend (Elasticsearch, local Docker Compose stack): the 40-case agent eval and the baseline run on this. This is the eval table in the README.
- Deployed backend (in-process): numpy cosine over the stored embeddings plus a small BM25, behind the same /search interface. The heading-chunker index is about 4,500 chunks x 1,536 dimensions, roughly 28 MB as float32, so it fits in memory.
- Config switch: RETRIEVAL_BACKEND=es|inproc. The agent code does not change between them.
- Parity check: run rag-hybrid's 26-question eval on both backends and report the gap. Expect a small BM25 gap (no ES analyzers: stemming, tokenisation differ). kNN should be near identical (same vectors, same cosine). RRF fusion should narrow the gap. Report whatever is measured. The README states plainly that the in-process backend is a hosting compromise and gives the measured cost of the swap.
- Not chosen: Elastic Cloud (14-day trial per search excerpts, pricing unverified). A small always-on VPS running single-node Elasticsearch is the fallback only if a specific role makes a live Elasticsearch worth the monthly cost and ops work. Prices not checked; test memory on the host first.
- Hosting decision is closed for evening 1.

## Out of scope
Front end, user accounts, fine-tuning, any claim of "real users" unless people actually use it.

## Timeline (9 evenings)
1. Data pull (about 100 candidates), start MCP server. (Hosting decision already made: see above.)
2. Finish MCP server, agent loop and step cap.
3. Labelling, part 1.
4. Labelling, part 2.
5. Eval harness and baseline comparison.
6. Failure tests and CI gate (recorded fixtures).
7. In-process retrieval backend behind /search, RETRIEVAL_BACKEND switch, parity run on the 26 questions.
8. Dockerise and deploy, health checks, cost cap, rate limiting, URL restriction.
9. README with results, parity numbers, honest limits and a short architecture diagram.

Note: the in-process backend takes about one evening, so the buffer from the 7-to-9 revision is now used up. If labelling overruns, the extra evening comes from the end, not from skipping the parity run.

## Definition of done
Public repo, live endpoint, eval table with counts and the baseline comparison, failure tests passing in CI, README that states what it cannot do.

## Allowed claim afterwards
"Multi-step agent with MCP tool server, deployed, evaluated on 40 labelled cases against a single-shot baseline." Do not add "production" or "users" unless true.

## Unverified items

Updated 25 Oct.

Verified (RAN):
- Substantive kind/support pool: 96 of 100 pulled candidates are substantive (see Findings 25 Oct). The 23 threads with no maintainer reply were not hand-checked for a pasted fix.
- Live GitHub: `scripts.pull_candidates` ran against the real API (listing and comments, read-only token). The `get_issue_thread` MCP tool itself has not been called live.
- rag-hybrid `/search`: bm25 and hybrid ran live and `search_docs` normalises both to `SearchResult`. Hybrid depends on rag-hybrid's OpenAI key (it was invalid once and was fixed).
- Agent loop, baseline, ledger and eval harness: 215 unit tests pass, all on fakes and synthetic fixtures.
- The `anthropic` package is installed and pinned (2.3.0); the adapter constructs a client offline.

Candidate pulls (25 Oct, script counts from `scripts.pull_candidates`, real GitHub API, read-only token). The "Maintainer reply" column and the maintainer-reply counts below were computed under the OLD leakage rule (OWNER/MEMBER/COLLABORATOR only) and are superseded; the script now reports "has reply by someone else", and the pulls need re-running to get that stat. "Substantive" means body >= 200 chars and at least one human comment by someone else. It does NOT mean the docs answer the issue; that is only known after labelling.

| Pull | Pulled | Body >= 200 | Human comment by someone else | Maintainer reply | Substantive | Median body chars |
|---|---|---|---|---|---|---|
| kind/support, limit 100 | 100 | 96 | 100 | 77 | 96 | 1738 |
| kind/support, limit 250 | 250 | 244 | 250 | 205 | 244 | 1950 |
| kind/bug + kind/support, limit 60 | 60 | 60 | 60 | 55 | 60 | 3289 |
| kind/bug, limit 40 | 40 | 40 | 35 | 10 | 35 | 3048 |
| kind/feature, limit 40 | 40 | 40 | 38 | 21 | 38 | 2798 |

- The pulls overlap, so the rows must not be added together. Every kind/bug + kind/support issue also carries kind/support. The 250-issue support pull is newest-first and very likely contains the first 100.
- All samples are newest-first. The docs index is pinned at kubernetes/website 6c42e266 (its commit date has not been checked), so very recent issues may refer to docs the index does not have.
- Threads with no maintainer reply are never truncated by the leakage rule: 45 of 250 in the support pull (18%), 30 of 40 in kind/bug and 19 of 40 in kind/feature. They need a hand check for pasted fixes before use as answerable cases.
- Only the script's filter has been applied: 244 of 250 support issues passed it. No issue has been read, selected or labelled.

Not verified:
- Any real model call: `AnthropicLLM.complete`, the system prompts, the `final_answer` tool format, and the model ID `claude-haiku-5-5` are untested. Prices are owner-confirmed 25 Oct 2026 from platform.claude.com/docs/en/about-claude/pricing (claude-haiku-5-5: input 0.10 / output 0.50 per MTok up to 100,000 prompt tokens, 0.50 / 2.50 above, plus cache rates; claude-haiku-4-5-20251001: 1 / 5, flat). Prompt size = input + cache read + cache write tokens; above 100,000 all five rates switch. Caps are owner-set: $0.02 per run, $1.00 per day. The prices and tier rule are exercised by unit tests only, not by any real response.
- `MAX_PLAUSIBLE_TOKENS` (10**7) and `MAX_TOOL_CALLS` (8) are chosen, not measured.
- The spend-ledger lock was tested by the reviewer with 4 processes, not under real load.
- The agent, the baseline and the eval harness have not run on a real issue or on any labelled case.
- Candidate pulls for the unanswerable pool exist (bug+support 60, bug 40, feature 40, as script counts), but no issue has been read, selected or labelled, so the 15 unanswerable cases are not chosen yet.
- Whether the docs index (kubernetes/website 6c42e266) predates the pulled issues enough to mislabel answerable cases.
- Elastic Cloud trial terms and pricing (not chosen; fallback only).
- Whether in-process retrieval matches the Elasticsearch numbers (parity check, evening 7).

## Findings 25 Oct

Facts only.

- Candidate pull (kind/support, `--limit 100`, real GitHub, read-only token): 100 pulled; body >= 200 chars: 96; >= 1 human comment by someone else: 100 (the script already requires >= 3 GitHub comments); has maintainer reply: 77; substantive (both of the above): 96; median body chars: 1738. Unanswerable-pool pulls not yet run.
- rag-hybrid `/search` hybrid mode works (HTTP 200, RRF scores about 0.03; bm25 scores about 13, so scores are not comparable across modes). An earlier 502 `embedding_unavailable` was an invalid OpenAI key in rag-hybrid's `.env`, since fixed. Uncached query embedding cost 8e-08 USD (3.4 s); a cached one reports cost_usd 0.0 (query-embedding cache). Cost and latency are reported from uncached runs only, and `cache_hit` is recorded for each search call.
- Docs index is pinned to kubernetes/website commit 6c42e266.
- Baseline changed (supersedes the Eval section's "rag-hybrid answering in one shot"): single-shot with the same Haiku model: one `search_docs` call on the same truncated thread, then an answer in the TriageNote format. rag-hybrid `/ask` (Sonnet) is a separate second reference, not the baseline.
- Order change: the in-process retrieval backend moves to evening 7 as in the timeline; agent loop and eval harness skeleton are built first.

## Labelling

Status: in progress. The eval set (25 answerable / 15 not, unchanged) is not yet chosen from these labels; categories are not yet defined (`category` is "TBD" in staging).

**Method.** Claude Sonnet 5.5 drafts one label per issue (yes / no / unsure) from the thread as `get_issue_thread` returns it plus three merged searches. For "yes" it must give doc_ids and verbatim quotes; the quotes are checked mechanically against the retrieved chunks, and any failed check turns the label into "unsure". Every label is then reviewed by the owner with `scripts/review_labels.py` (y accept, n flip yes/no, u unsure; on an unsure draft y = yes, n = no; `--redo` replaces a decision and keeps the old one). A "yes" needs a doc_id from that issue's search results. Drafts live in `data/drafts/`; nothing here has been promoted to `data/eval/`.

**Counts** (read from `data/drafts/reviewed.jsonl`, the owner's decisions, 173 records): 26 yes, 85 no, 62 unsure. Yes rate 26/173 = 15.0%. Draft actions in that file: 106 accepted, 3 flipped, 2 decided from unsure, 62 left unsure. The first 72 issues were pre-screened by title before drafting (owner's account; not recorded in the data files), so the yes rate is not a rate over all candidates.

**Leakage-rule fix and redraft check.** The thread rule changed to: cut at the first comment by anyone but the issue author; bots ignored; `author_association` dropped. 13 reviewed issues were redrafted under the new rule (`data/drafts/labels_newrule.jsonl`). Compared with the owner's decisions: 10 of 13 agree (all 10 yes/yes). The 3 differences: owner yes vs redraft no (1), owner yes vs redraft unsure (1), owner unsure vs redraft no (1). Three decisions in `reviewed.jsonl` were redone with `--redo` (each has `redone: true` and its `previous_decision`, so the history is kept): 141263 unsure to yes, 133869 no to unsure, and 133312 yes to unsure (the borderline case). The counts above are after those redos.

**Findings.**
- About 13 retrieval misses (owner's count, not derivable from the files): the docs answer the issue but search did not return the page. These are labelled so the miss counts against retrieval, not as "unanswerable".
- Yes rate is about 15% (26/173).
- Unsure (62) is mostly vague failures with no confirmed cause (owner's reading).

## Category set

Defined by the owner. Four categories, checked in this order; the first that matches wins. The names are `CATEGORIES` in `triage/contracts.py`.

1. `version_change` (Version or feature change): the behaviour comes from a deprecation, removal, version requirement, or a feature being alpha or behind a gate.
2. `config_error` (Configuration error): the user's own config, command or host setup is wrong, and fixing it solves the problem.
3. `by_design` (Works as designed): Kubernetes behaves as intended; the user misread what it should do.
4. `howto` (How-to or diagnosis): the user asks how to do something or how to find a cause, and the docs give the steps.

Unanswerable cases get category `n/a`. Category accuracy is scored on answerable cases only. The owner's per-issue assignment is in `data/eval_staging/categories.json` (staging, not committed; 25 issues: 5 version_change, 3 config_error, 9 by_design, 8 howto).

## Sampling (26 Oct)

Sampling cannot be pinned with the installed anthropic SDK: `messages.create` has no `temperature` parameter (a live attempt failed on it before any request). So model output varies run to run. The eval runs each case 3 times per system (agent and baseline) and reports the mean and the range, with counts. The 3-run repeat is not implemented in the eval runner yet.

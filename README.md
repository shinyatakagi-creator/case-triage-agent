# case-triage-agent

An agent that takes a public kubernetes/kubernetes issue URL, investigates it with three MCP tools (`search_docs` over the Kubernetes docs, `get_issue_thread`, `extract_error_signals`), and returns a triage note: likely cause, cited docs and a next step. "Can't tell" is a valid, measured outcome. It is a portfolio project, compared against a single-shot baseline on 40 hand-labelled issues.

**Status: not yet deployed.** It runs locally only.

## Allowed claim

> "Multi-step agent with MCP tool server, deployed, evaluated on 40 labelled cases against a single-shot baseline."

The word "deployed" applies only once the service is deployed, which it is not yet. Until then the accurate claim is: multi-step agent with MCP tool server, evaluated on 40 labelled cases against a single-shot baseline.

## Results

Source: `python -m scripts.run_eval --run-dir data/results/20261026-153314 --report-only`. 40 cases (25 answerable, 15 not), agent and baseline, 3 repeats each, 240 runs, Elasticsearch retrieval backend, same model for both (claude-haiku-5-5). Sampling is not pinned, so each cell is the mean over 3 repeats with the range, and k/n for each repeat. Total recorded cost 0.3481 USD.

| Metric | Agent | Baseline |
|---|---|---|
| Category accuracy, answerable (25) | mean 72.0% (68.0-76.0%); 17/25, 19/25, 18/25 | mean 65.3% (64.0-68.0%); 16/25, 16/25, 17/25 |
| Category accuracy, answered cases only | mean 81.8% (77.3-86.4%); 17/22, 19/22, 18/22 | mean 81.7% (80.0-85.0%); 16/20, 16/20, 17/20 |
| Correct doc cited, answerable (25) | mean 77.3% (72.0-80.0%); 20/25, 20/25, 18/25 | mean 77.3% (76.0-80.0%); 19/25, 19/25, 20/25 |
| Can't tell, answerable (25) | mean 12.0% (12.0-12.0%); 3/25 in each repeat | mean 20.0% (20.0-20.0%); 5/25 in each repeat |
| Can't tell, not answerable (15) | mean 71.1% (66.7-80.0%); 10/15, 10/15, 12/15 | mean 88.9% (86.7-93.3%); 13/15, 13/15, 14/15 |
| Failed runs (of 40 per repeat) | 0, 1, 0 | 0, 0, 0 |
| Cost per case (non-failed) | mean 0.0020 USD (0.0019-0.0020) | mean 0.0009 USD (0.0009-0.0009) |
| Steps per case | mean 3.2 (3.1-3.2) | 1.0 |
| Wall latency per run | mean 8.61 s (4.69-20.15 s) | mean 4.78 s (2.32-9.32 s) |

**Verdict: no clear difference, no win claimed.** The paired comparison per repeat says "no clear difference" in all six comparisons (category and correct doc, three repeats). The agent answered more answerable cases but abstained less on unanswerable ones, cost about 2.2x per case and took about 1.8x the wall time. Full analysis, including the unanswerable cases the agent answered, is in [docs/scope.md](docs/scope.md).

## Limits

25 answerable and 15 unanswerable cases, one model, the Elasticsearch backend only. With 25 answerable cases a gap of two or three cases is noise, and the three repeats reuse the same cases, so they are not independent evidence. The labels and category definitions are one person's judgement. The eval set, results and index are not in this repository (`data/` is not committed), so these numbers cannot be reproduced from a clone alone. The in-process retrieval backend (planned for the deployed build) matched Elasticsearch on rag-hybrid's 26 short eval questions, but the agent has not been run on it, and the build has not been deployed.

## Tests

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

The tests use fakes and recorded fixtures; they make no live calls.

## NOTICE

The retrieval index is built from the Kubernetes documentation, © The Kubernetes Authors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (source: [kubernetes/website](https://github.com/kubernetes/website), commit `6c42e266`). The index is not included in this repository; cited passages in the agent's output are excerpts of that documentation.

## License

MIT, see [LICENSE](LICENSE). The licence covers this repository's code only, not the Kubernetes documentation.

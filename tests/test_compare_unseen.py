import json

from scripts import compare_labels, draft_labels, unseen_candidates


def rev(n, decision, label="no"):
    return {"draft": {"number": n}, "decision": decision}


def test_compare_counts_only_shared_issues():
    reviewed = [rev(1, "yes"), rev(2, "no"), rev(3, "no")]
    new = [{"number": 1, "label": "yes"}, {"number": 2, "label": "unsure"}, {"number": 9, "label": "no"}]
    diff, agree = compare_labels.compare(reviewed, new)
    assert agree == 1 and [(r["draft"]["number"], d["label"]) for r, d in diff] == [(2, "unsure")]


def test_unseen_newest_first_skips_seen_and_limits():
    rows = [{"number": 1, "created_at": "2026-01-01T00:00:00Z"}, {"number": 2, "created_at": "2026-03-01T00:00:00Z"},
            {"number": 3, "created_at": "2026-02-01T00:00:00Z"}, {"number": 4, "created_at": "2026-04-01T00:00:00Z"}]
    assert unseen_candidates.unseen(rows, {4}, 2) == [2, 3]


def test_draft_labels_out_flag_reaches_run(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(draft_labels, "AnthropicLLM", lambda m: type("L", (), {"model": m, "model_available": lambda s: True})())
    monkeypatch.setattr(draft_labels, "Ledger", lambda **k: None)
    monkeypatch.setattr(draft_labels, "run", lambda n, llm, led, out_path: seen.update(out=out_path) or
                        {"drafted": 0, "skipped_existing": 0, "labels": {}, "issues_with_failed_checks": 0,
                         "total_cost_usd": 0, "stopped": None, "not_run": []})
    draft_labels.main(["5", "--out", str(tmp_path / "x.jsonl")])
    assert seen["out"] == str(tmp_path / "x.jsonl")
    draft_labels.main(["5"])
    assert seen["out"] == str(draft_labels.LABELS_PATH)

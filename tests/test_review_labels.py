import json

from scripts import review_labels as rl


def draft(n, label, **kw):
    return {"number": n, "label": label, "doc_ids": ["docs/a"] if label == "yes" else [], "quotes": ["q"] if label == "yes" else [],
            "reason": "r", "check_failures": [], "model": "m", "cost_usd": 0.0, "title": f"t{n}",
            "url": f"https://github.com/kubernetes/kubernetes/issues/{n}", "body_head": ["l1", "l2"], "model_label": label, **kw}


def keys(*ks):
    it = iter(ks)
    return lambda: next(it, "")


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


DRAFTS = [draft(1, "yes"), draft(2, "no"), draft(3, "unsure")]


def run(tmp_path, ks, opened=None):
    return rl.review(DRAFTS, tmp_path / "reviewed.jsonl", getkey=keys(*ks), show=lambda d, p: None,
                     opener=(opened.append if opened is not None else lambda u: None))


def test_accept_flip_unsure_are_recorded_with_whether_changed(tmp_path):
    s = run(tmp_path, "ynu")
    recs = lines(tmp_path / "reviewed.jsonl")
    assert [(r["decision"], r["action"], r["changed"]) for r in recs] == [
        ("yes", "accepted", False), ("yes", "flipped", True), ("unsure", "unsure", False)]
    assert recs[0]["draft"]["number"] == 1
    assert (s["accepted"], s["flipped"], s["unsure"], s["reviewed_total"]) == (1, 1, 1, 3)


def test_quit_saves_and_a_second_session_resumes_where_it_stopped(tmp_path):
    s1 = run(tmp_path, "yq")
    assert s1["quit_early"] and [r["draft"]["number"] for r in lines(tmp_path / "reviewed.jsonl")] == [1]
    s2 = run(tmp_path, "nu")  # issue 1 is not shown again: these keys decide issues 2 and 3
    recs = lines(tmp_path / "reviewed.jsonl")
    assert [r["draft"]["number"] for r in recs] == [1, 2, 3] and recs[1]["decision"] == "yes"
    assert s2["reviewed_total"] == 3 and not s2["quit_early"]
    assert run(tmp_path, "")["reviewed_total"] == 3  # nothing left to review


def test_end_of_input_quits_without_losing_decisions(tmp_path):
    run(tmp_path, "y")  # input ends after one key
    assert len(lines(tmp_path / "reviewed.jsonl")) == 1


def test_o_opens_the_issue_and_does_not_decide(tmp_path):
    opened = []
    run(tmp_path, "oyq", opened)
    assert opened == [DRAFTS[0]["url"]] and len(lines(tmp_path / "reviewed.jsonl")) == 1


def test_n_on_an_unsure_draft_is_ignored(tmp_path):
    rl.review([draft(3, "unsure")], tmp_path / "r.jsonl", getkey=keys("n", "y"), show=lambda d, p: None)
    assert lines(tmp_path / "r.jsonl")[0]["action"] == "accepted"


def test_reviewed_file_cannot_be_under_data_eval(tmp_path, monkeypatch):
    import pytest
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        rl.review(DRAFTS, "data/eval/reviewed.jsonl", getkey=keys("y"), show=lambda d, p: None)


def test_show_prints_the_fields_the_owner_needs(capsys):
    rl.show(draft(1, "yes", check_failures=["x failed"]), "1/3")
    out = capsys.readouterr().out
    for want in ("#1  t1", "issues/1", "l1", "DRAFT: YES", "doc: docs/a", "quote: q", "reason: r", "x failed"):
        assert want in out

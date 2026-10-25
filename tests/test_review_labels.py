import json

from scripts import review_labels as rl


def draft(n, label, **kw):
    return {"number": n, "label": label, "doc_ids": ["docs/a"] if label == "yes" else [], "quotes": ["q"] if label == "yes" else [],
            "reason": "r", "check_failures": [], "model": "m", "cost_usd": 0.0, "title": f"t{n}",
            "url": f"https://github.com/kubernetes/kubernetes/issues/{n}", "body_head": ["l1", "l2"], "model_label": label,
            "result_doc_ids": ["docs/a", "docs/b"], **kw}


def keys(*ks):
    it = iter(ks)
    return lambda: next(it, "")


def typed(*lines):
    it = iter(lines)
    return lambda: next(it, "") + "\n"


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


DRAFTS = [draft(1, "yes"), draft(2, "no"), draft(3, "unsure")]


def run(tmp_path, ks, opened=None, typed_lines=(), drafts=DRAFTS):
    return rl.review(drafts, tmp_path / "reviewed.jsonl", getkey=keys(*ks), show=lambda d, p: None,
                     opener=(opened.append if opened is not None else lambda u: None), readline=typed(*typed_lines))


def test_accept_flip_unsure_are_recorded_with_whether_changed(tmp_path):
    # 1 yes: accept; 2 no: flip to yes (needs a doc_id, typed); 3 unsure: keep unsure
    s = run(tmp_path, "ynu", typed_lines=["docs/b"])
    recs = lines(tmp_path / "reviewed.jsonl")
    assert [(r["decision"], r["action"], r["changed"]) for r in recs] == [
        ("yes", "accepted", False), ("yes", "flipped", True), ("unsure", "unsure", False)]
    assert [r["decision_doc_ids"] for r in recs] == [["docs/a"], ["docs/b"], []]
    assert [r["doc_ids_source"] for r in recs] == ["draft", "typed", None]
    assert [r["decided_from_unsure"] for r in recs] == [False, False, False]
    assert recs[0]["draft"]["number"] == 1
    assert (s["accepted"], s["flipped"], s["unsure"], s["resolved"], s["reviewed_total"]) == (1, 1, 1, 0, 3)


def test_flip_yes_to_no_needs_no_doc_id(tmp_path):
    run(tmp_path, "n", drafts=[draft(1, "yes")])
    r = lines(tmp_path / "reviewed.jsonl")[0]
    assert (r["decision"], r["action"], r["decision_doc_ids"]) == ("no", "flipped", [])


# --- an unsure draft ----------------------------------------------------------


def test_unsure_n_is_no_and_u_keeps_unsure_and_both_are_recorded_properly(tmp_path):
    s = run(tmp_path, "nu", drafts=[draft(3, "unsure"), draft(4, "unsure")])
    a, b = lines(tmp_path / "reviewed.jsonl")
    assert (a["decision"], a["action"], a["changed"], a["decided_from_unsure"]) == ("no", "resolved", True, True)
    assert (b["decision"], b["action"], b["changed"], b["decided_from_unsure"]) == ("unsure", "unsure", False, False)
    assert s["resolved"] == 1 and s["unsure"] == 1


def test_unsure_y_uses_the_drafts_doc_ids_when_they_are_in_the_results(tmp_path):
    d = draft(3, "unsure", doc_ids=["docs/b", "docs/zzz"], quotes=["q1", "q2"])
    run(tmp_path, "y", drafts=[d])
    r = lines(tmp_path / "reviewed.jsonl")[0]
    assert (r["decision"], r["action"], r["decided_from_unsure"]) == ("yes", "resolved", True)
    assert r["decision_doc_ids"] == ["docs/b"] and r["doc_ids_source"] == "draft"  # the invalid one is not carried over


def test_unsure_y_without_doc_ids_asks_and_refuses_one_not_in_the_results(tmp_path, capsys):
    run(tmp_path, "y", drafts=[draft(3, "unsure")], typed_lines=["docs/nope", "docs/b"])
    r = lines(tmp_path / "reviewed.jsonl")[0]
    assert r["decision"] == "yes" and r["decision_doc_ids"] == ["docs/b"] and r["doc_ids_source"] == "typed"
    assert "refused: 'docs/nope'" in capsys.readouterr().out


def test_unsure_y_with_only_invalid_draft_doc_ids_still_asks(tmp_path):
    run(tmp_path, "y", drafts=[draft(3, "unsure", doc_ids=["docs/zzz"], quotes=["q"])], typed_lines=["docs/a"])
    assert lines(tmp_path / "reviewed.jsonl")[0]["decision_doc_ids"] == ["docs/a"]


def test_empty_doc_id_cancels_back_to_the_key_prompt_and_records_nothing(tmp_path):
    s = run(tmp_path, "yn", drafts=[draft(3, "unsure")], typed_lines=[""])  # y -> cancelled -> n decides "no"
    r = lines(tmp_path / "reviewed.jsonl")
    assert len(r) == 1 and r[0]["decision"] == "no" and s["resolved"] == 1


def test_draft_without_recorded_results_cannot_get_a_typed_doc_id(tmp_path, capsys):
    d = draft(3, "unsure")
    d.pop("result_doc_ids")
    run(tmp_path, "yu", drafts=[d], typed_lines=["docs/a", ""])  # refused, then cancelled; u keeps unsure
    r = lines(tmp_path / "reviewed.jsonl")
    assert len(r) == 1 and r[0]["decision"] == "unsure" and "none recorded" in capsys.readouterr().out


# --- quitting, resuming, guards -----------------------------------------------


def test_quit_saves_and_a_second_session_resumes_where_it_stopped(tmp_path):
    s1 = run(tmp_path, "yq")
    assert s1["quit_early"] and [r["draft"]["number"] for r in lines(tmp_path / "reviewed.jsonl")] == [1]
    s2 = run(tmp_path, "nn", typed_lines=["docs/a"])  # issue 1 is not shown again: these keys decide issues 2 (no->yes) and 3 (unsure->no)
    recs = lines(tmp_path / "reviewed.jsonl")
    assert [r["draft"]["number"] for r in recs] == [1, 2, 3] and not s2["quit_early"]
    assert run(tmp_path, "")["reviewed_total"] == 3  # nothing left to review


def test_end_of_input_quits_without_losing_decisions(tmp_path):
    run(tmp_path, "y")  # input ends after one key
    assert len(lines(tmp_path / "reviewed.jsonl")) == 1


def test_o_opens_the_issue_and_does_not_decide(tmp_path):
    opened = []
    run(tmp_path, "oyq", opened)
    assert opened == [DRAFTS[0]["url"]] and len(lines(tmp_path / "reviewed.jsonl")) == 1


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

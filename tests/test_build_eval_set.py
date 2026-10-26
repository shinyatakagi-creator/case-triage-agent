import json

import pytest

from scripts import build_eval_set as b


def rev(n, decision, docs=()):
    return {"draft": {"number": n, "url": f"https://github.com/kubernetes/kubernetes/issues/{n}"},
            "decision": decision, "decision_doc_ids": list(docs)}


R = {n: rev(n, d, ["docs/x"] if d == "yes" else []) for n, d in [(1, "yes"), (2, "yes"), (3, "no"), (4, "no"), (5, "unsure"), (6, "yes")]}


def test_build_shapes_rows():
    ev, dev = b.build([1, 2], [3], [6], R)
    assert ev[0] == {"id": "k8s-1", "issue_url": "https://github.com/kubernetes/kubernetes/issues/1",
                     "answerable": True, "category": "TBD", "gold_docs": ["docs/x"]}
    assert ev[2]["answerable"] is False and ev[2]["gold_docs"] == []
    assert [c["id"] for c in dev] == ["k8s-6"]


@pytest.mark.parametrize("a,u,d,msg", [
    ([3], [], [], "listed as 'yes'"), ([], [1], [], "listed as 'no'"), ([5], [], [], "unsure"),
    ([], [5], [], "unsure"), ([], [], [5], "unsure"), ([99], [], [], "not in reviewed"), ([1], [], [1], "more than once"),
])
def test_refusals(a, u, d, msg):
    with pytest.raises(ValueError, match=msg):
        b.build(a, u, d, R)


def test_main_writes_staging_and_refuses_eval_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/drafts").mkdir(parents=True)
    (tmp_path / "data/drafts/reviewed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in R.values()))
    monkeypatch.setattr(b, "REVIEWED_PATH", tmp_path / "data/drafts/reviewed.jsonl")
    assert b.main(["--answerable", "1", "--unanswerable", "3", "--dev", "6"]) == 0
    assert len((tmp_path / "data/eval_staging/eval.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "data/eval_staging/dev.jsonl").exists()
    assert b.main(["--answerable", "5"]) == 2
    assert b.main(["--answerable", "1", "--out-dir", "data/eval"]) == 2
    assert not (tmp_path / "data/eval").exists()

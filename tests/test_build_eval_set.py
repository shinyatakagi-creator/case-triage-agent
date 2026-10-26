import json

import pytest

from scripts import build_eval_set as b


def rev(n, decision, docs=()):
    return {"draft": {"number": n, "url": f"https://github.com/kubernetes/kubernetes/issues/{n}"},
            "decision": decision, "decision_doc_ids": list(docs)}


R = {n: rev(n, d, ["docs/x"] if d == "yes" else []) for n, d in [(1, "yes"), (2, "yes"), (3, "no"), (4, "no"), (5, "unsure"), (6, "yes")]}


C = {1: "howto", 2: "by_design", 6: "config_error"}


def test_build_shapes_rows():
    ev, dev = b.build([1, 2], [3], [6], R, C)
    assert ev[0] == {"id": "k8s-1", "issue_url": "https://github.com/kubernetes/kubernetes/issues/1",
                     "answerable": True, "category": "howto", "gold_docs": ["docs/x"]}
    assert ev[2]["answerable"] is False and ev[2]["gold_docs"] == [] and ev[2]["category"] == "n/a"
    assert ev[1]["category"] == "by_design"
    assert [c["id"] for c in dev] == ["k8s-6"]


@pytest.mark.parametrize("a,u,d,msg", [
    ([3], [], [], "listed as 'yes'"), ([], [1], [], "listed as 'no'"), ([5], [], [], "unsure"),
    ([], [5], [], "unsure"), ([], [], [5], "unsure"), ([99], [], [], "not in reviewed"), ([1], [], [1], "more than once"),
])
def test_refusals(a, u, d, msg):
    with pytest.raises(ValueError, match=msg):
        b.build(a, u, d, R, C)


@pytest.mark.parametrize("cats", [{}, {1: "nonsense"}, {1: "n/a"}])
def test_answerable_needs_a_valid_category(cats):
    with pytest.raises(ValueError, match="category"):
        b.build([1], [], [], R, cats)


def test_main_writes_staging_and_refuses_eval_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/drafts").mkdir(parents=True)
    (tmp_path / "data/drafts/reviewed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in R.values()))
    (tmp_path / "cats.json").write_text(json.dumps({"1": "howto", "5": "howto", "6": "config_error"}))
    monkeypatch.setattr(b, "REVIEWED_PATH", tmp_path / "data/drafts/reviewed.jsonl")
    assert b.main(["--categories", "cats.json", "--answerable", "1", "--unanswerable", "3", "--dev", "6"]) == 0
    assert len((tmp_path / "data/eval_staging/eval.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "data/eval_staging/dev.jsonl").exists()
    assert b.main(["--categories", "cats.json", "--answerable", "5"]) == 2
    assert b.main(["--categories", "cats.json", "--answerable", "1", "--out-dir", "data/eval"]) == 2
    assert not (tmp_path / "data/eval").exists()

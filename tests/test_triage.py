import pytest

from triage.signals import extract_error_signals
from triage.thread import cut_at_first_maintainer, parse_issue_url


def user(login, assoc="NONE", typ="User"):
    return {"login": login, "type": typ}, assoc


def comment(login, body, assoc="NONE", typ="User"):
    return {"user": {"login": login, "type": typ}, "author_association": assoc, "body": body, "created_at": "t"}


ISSUE = {
    "number": 1,
    "title": "pod stuck",
    "user": {"login": "alice", "type": "User"},
    "created_at": "t",
    "labels": [{"name": "kind/support"}],
    "body": "my pod is Pending on v1.28.3",
}


def test_parse_issue_url_ok():
    assert parse_issue_url("https://github.com/kubernetes/kubernetes/issues/12345") == 12345
    assert parse_issue_url("https://github.com/kubernetes/kubernetes/issues/12345/") == 12345


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/kubernetes/kubernetes/pull/1",
        "https://github.com/other/repo/issues/1",
        "http://github.com/kubernetes/kubernetes/issues/1",
        "https://github.com/kubernetes/kubernetes/issues/1?x=1",
        "https://evil.example/kubernetes/kubernetes/issues/1",
    ],
)
def test_parse_issue_url_rejects(url):
    with pytest.raises(ValueError):
        parse_issue_url(url)


def test_cut_at_first_maintainer_withholds_answer():
    comments = [
        comment("k8s-ci-robot", "@alice: This issue is currently awaiting triage.", "MEMBER"),
        comment("bob", "same here, which CNI?", "NONE"),
        comment("alice", "calico", "NONE"),
        comment("carol", "see https://kubernetes.io/docs/foo", "MEMBER"),
        comment("dave", "thanks", "NONE"),
    ]
    out = cut_at_first_maintainer(ISSUE, comments)
    assert [c["author"] for c in out["comments"]] == ["bob", "alice"]
    assert out["thread_truncated"] is True


def test_issue_author_who_is_member_is_not_a_maintainer_reply():
    comments = [comment("alice", "more detail", "MEMBER"), comment("bob", "try X", "NONE")]
    out = cut_at_first_maintainer(ISSUE, comments)
    assert len(out["comments"]) == 2
    assert out["thread_truncated"] is False


def test_signals_basic():
    text = (
        "kubectl version: GitVersion:\"v1.28.3\"\n"
        "pod is CrashLoopBackOff\n"
        "Error from server (Forbidden): pods is forbidden\n"
        "panic: runtime error: index out of range\n"
        "goroutine 1 [running]:\nmain.main()\n"
        "all good here\n"
    )
    s = extract_error_signals(text)
    assert s["versions"] == ["v1.28.3"]
    assert "CrashLoopBackOff" in s["k8s_reasons"]
    assert any("Forbidden" in l for l in s["error_lines"])
    assert len(s["stack_traces"]) == 2
    assert not any("all good" in l for l in s["error_lines"])


def test_signals_empty():
    s = extract_error_signals("")
    assert s == {"versions": [], "k8s_reasons": [], "error_lines": [], "stack_traces": []}

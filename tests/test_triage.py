import pytest

from triage.signals import extract_error_signals
from triage.thread import cut_at_first_reply, is_reply_by_other, parse_issue_url


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


# --- leakage rule: author's own comments up to the first comment by anyone else ----------------------


def test_cut_withholds_answer_old_case_under_the_new_rule():
    # Same inputs as the old maintainer-only test. Old rule kept bob and alice (bob was NONE, not a
    # maintainer); the new rule cuts at bob, the first human who is not the issue author.
    comments = [
        comment("k8s-ci-robot", "@alice: This issue is currently awaiting triage.", "MEMBER"),
        comment("bob", "same here, which CNI?", "NONE"),
        comment("alice", "calico", "NONE"),
        comment("carol", "see https://kubernetes.io/docs/foo", "MEMBER"),
        comment("dave", "thanks", "NONE"),
    ]
    out = cut_at_first_reply(ISSUE, comments)
    assert out["comments"] == []
    assert out["thread_truncated"] is True


def test_issue_author_who_is_member_is_not_a_reply_by_someone_else():
    # Old case: alice (the author) is a MEMBER and comments first. She is still the author, so kept.
    # Old rule kept bob too; the new rule cuts at bob.
    comments = [comment("alice", "more detail", "MEMBER"), comment("bob", "try X", "NONE")]
    out = cut_at_first_reply(ISSUE, comments)
    assert [c["author"] for c in out["comments"]] == ["alice"]
    assert out["thread_truncated"] is True


def test_contributor_maintainer_reply_cuts():
    # #131977: ffromani (SIG Node maintainer) has author_association CONTRIBUTOR; the old rule let his
    # answer through because only OWNER/MEMBER/COLLABORATOR counted.
    comments = [comment("alice", "it also happens on 1.28", "NONE"),
                comment("ffromani", "this is the cpu manager policy, see the docs", "CONTRIBUTOR"),
                comment("alice", "thanks, that fixed it", "NONE")]
    out = cut_at_first_reply(ISSUE, comments)
    assert [c["author"] for c in out["comments"]] == ["alice"] and out["comments"][0]["body"] == "it also happens on 1.28"
    assert out["thread_truncated"] is True


def test_community_reply_with_no_association_cuts():
    out = cut_at_first_reply(ISSUE, [comment("bob", "have you tried X?", "NONE"), comment("alice", "yes", "NONE")])
    assert out["comments"] == [] and out["thread_truncated"] is True


def test_author_follow_up_before_any_reply_is_kept():
    comments = [comment("alice", "forgot to say: kind cluster", "NONE"), comment("alice", "and v1.29", "NONE")]
    out = cut_at_first_reply(ISSUE, comments)
    assert [c["body"] for c in out["comments"]] == ["forgot to say: kind cluster", "and v1.29"]
    assert out["thread_truncated"] is False


def test_bot_comment_does_not_cut_or_appear():
    comments = [comment("k8s-ci-robot", "/triage accepted", "MEMBER"),
                comment("some-app[bot]", "stale", "NONE", typ="Bot"),
                comment("alice", "still broken", "NONE"),
                comment("fejta-bot", "Issues go stale after 90d", "NONE"),
                comment("bob", "the fix is Y", "NONE")]
    out = cut_at_first_reply(ISSUE, comments)
    assert [c["author"] for c in out["comments"]] == ["alice"]  # bots dropped, alice kept past them
    assert out["thread_truncated"] is True  # cut at bob


def test_only_bots_means_nothing_cut():
    out = cut_at_first_reply(ISSUE, [comment("k8s-ci-robot", "hi", "MEMBER")])
    assert out["comments"] == [] and out["thread_truncated"] is False


def test_is_reply_by_other():
    assert is_reply_by_other(comment("bob", "x", "NONE"), "alice")
    assert not is_reply_by_other(comment("alice", "x", "MEMBER"), "alice")
    assert not is_reply_by_other(comment("k8s-ci-robot", "x", "MEMBER"), "alice")


def test_issue_thread_shape_is_unchanged():
    out = cut_at_first_reply(ISSUE, [])
    assert set(out) == {"number", "title", "author", "created_at", "labels", "body", "comments", "thread_truncated"}


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

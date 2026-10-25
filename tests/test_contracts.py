from triage.contracts import MAX_STEPS, contract_problems

RESULT = {"passages": [{"source": "tasks/a"}, {"source": "tasks/b"}], "cost_usd": 0, "latency_ms": 0}


def note(**kw):
    base = {"issue_url": "u", "outcome": "answered", "category": None, "likely_cause": "x",
            "citations": [{"source": "tasks/a"}], "next_step": "y", "steps": 3,
            "cost_usd": 0.0, "tool_calls": []}
    return {**base, **kw}


def test_clean_note():
    assert contract_problems(note(), [RESULT]) == []


def test_citation_not_from_search():
    assert len(contract_problems(note(citations=[{"source": "tasks/zzz"}]), [RESULT])) == 1


def test_citation_with_no_searches_at_all():
    assert len(contract_problems(note(), [])) == 1


def test_answered_without_citations():
    assert contract_problems(note(citations=[]), [RESULT]) == ["outcome 'answered' with no citations"]


def test_cant_tell_without_citations_is_fine():
    assert contract_problems(note(outcome="cant_tell", citations=[]), []) == []


def test_steps_over_cap():
    assert len(contract_problems(note(steps=MAX_STEPS + 1), [RESULT])) == 1
    assert contract_problems(note(steps=MAX_STEPS), [RESULT]) == []


def test_normalise_search_maps_rag_hybrid_fields():
    from triage.server import normalise_search

    raw = {"results": [{"rank": 1, "score": 1.5, "chunk_id": "c1", "doc_id": "tasks/x", "url": "u",
                        "title": "T", "heading_path": "T > H", "token_count": 3, "content": "body"}],
           "cost_usd": 0.0, "latency_ms": 12.5}
    out = normalise_search(raw)
    assert out["passages"][0] == {"text": "body", "source": "tasks/x", "url": "u", "title": "T",
                                  "heading": "T > H", "chunk_id": "c1", "score": 1.5}
    assert (out["cost_usd"], out["latency_ms"]) == (0.0, 12.5)

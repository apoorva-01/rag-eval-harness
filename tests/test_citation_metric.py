from eval.citation_metric import parse_citation_indices, _precision_recall


def test_parse_citation_indices():
    assert parse_citation_indices("12 layers [1], trained on C4 [2][3].") == [1, 2, 3]
    assert parse_citation_indices("No citations here.") == []


def test_all_cited_and_supported():
    claims = [
        {"is_factual_claim": True, "cited_sources": [1], "supported_by_cited": True},
        {"is_factual_claim": True, "cited_sources": [2], "supported_by_cited": True},
    ]
    s = _precision_recall(claims)
    assert s["citation_precision"] == 1.0
    assert s["citation_recall"] == 1.0


def test_uncited_claim_lowers_recall_only():
    claims = [
        {"is_factual_claim": True, "cited_sources": [1], "supported_by_cited": True},
        {"is_factual_claim": True, "cited_sources": [], "supported_by_cited": False},
    ]
    s = _precision_recall(claims)
    assert s["citation_precision"] == 1.0   # the one cited claim is supported
    assert s["citation_recall"] == 0.5      # only half the claims carry a citation


def test_unsupported_citation_lowers_precision_only():
    claims = [
        {"is_factual_claim": True, "cited_sources": [1], "supported_by_cited": False},
        {"is_factual_claim": True, "cited_sources": [2], "supported_by_cited": True},
    ]
    s = _precision_recall(claims)
    assert s["citation_precision"] == 0.5
    assert s["citation_recall"] == 1.0


def test_refusal_with_no_factual_claims_is_excluded():
    claims = [
        {"is_factual_claim": False, "cited_sources": [], "supported_by_cited": False},
    ]
    s = _precision_recall(claims)
    assert s["citation_precision"] is None
    assert s["citation_recall"] is None


def test_claims_but_no_citations_precision_undefined_recall_zero():
    claims = [
        {"is_factual_claim": True, "cited_sources": [], "supported_by_cited": False},
    ]
    s = _precision_recall(claims)
    assert s["citation_precision"] is None   # nothing cited -> precision undefined
    assert s["citation_recall"] == 0.0

import pytest

from evaluation.metrics import Target, citation_problems, quote_in_text, rank_scores
from hackathon2.schemas import Evidence, SearchHit


def _hit(doc_id: str, n: int = 1, section: str | None = None, text: str = "") -> SearchHit:
    return SearchHit(chunk_id=f"{doc_id}#s{section or 1}#c{n}", doc_id=doc_id, source=f"{doc_id}.pdf",
                     doc_type="policy", section=section, text=f"<untrusted_document>{text}</untrusted_document>")


TARGETS = [Target(doc_id="information-security-policy", grade=3), Target(doc_id="data-classification-policy", grade=2)]


def test_perfect_and_irrelevant_rankings():
    perfect = rank_scores([_hit("information-security-policy"), _hit("data-classification-policy")], TARGETS, k=5)
    assert perfect == {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "ndcg": pytest.approx(1.0)}
    assert rank_scores([_hit("vendor-x-pricing")], TARGETS, k=5) == {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}


def test_rank_order_counts_and_one_document_is_credited_once():
    late = rank_scores([_hit("vendor-x-pricing"), _hit("vendor-x-proposal"), _hit("information-security-policy")],
                       TARGETS, k=5)
    assert late["mrr"] == pytest.approx(1 / 3) and 0 < late["ndcg"] < 1
    same_doc = rank_scores([_hit("information-security-policy", n) for n in range(5)], TARGETS, k=5)
    assert same_doc["recall"] == 0.5 and same_doc["ndcg"] < 1  # five chunks, still one of two targets


def test_section_targets_match_by_prefix():
    target = Target(doc_id="information-security-policy", section="4")
    assert target.matches(_hit("information-security-policy", section="4.2"))
    assert not target.matches(_hit("information-security-policy", section="7.1"))


def test_quote_matching_ignores_tags_case_spacing_quotes_and_allows_ellipsis():
    text = "<untrusted_document>Enterprise licence: EUR 25 per user,\n billed annually. Minimum term: 3 years.</untrusted_document>"
    assert quote_in_text("enterprise LICENCE: eur 25 per user, billed annually.", text)
    assert quote_in_text("Enterprise licence ... Minimum term: 3 years", text)
    assert not quote_in_text("Minimum term ... Enterprise licence", text)  # order matters
    assert not quote_in_text("Enterprise licence: EUR 20 per user", text)
    assert quote_in_text("Asteria’s audit", "Asteria's audit")


def test_citation_checks():
    hit = _hit("vendor-x-pricing", text="Enterprise licence: EUR 25 per user per month.")
    log, hits = {hit.chunk_id}, {hit.chunk_id: hit}

    def evidence(quote="EUR 25 per user per month", chunk_id=hit.chunk_id, source=hit.source):
        return Evidence(chunk_id=chunk_id, source=source, doc_type="policy", quote=quote)

    assert citation_problems(evidence(), log, hits) == []
    assert citation_problems(evidence("EUR 20 per user"), log, hits) == ["misquoted"]
    assert citation_problems(evidence(source="vendor-x-proposal.pdf"), log, hits) == ["source_mismatch"]
    fabricated = evidence(chunk_id="ai-governance-policy#s9#c4", source="ai-governance-policy.pdf")
    assert citation_problems(fabricated, log, hits) == ["fabricated"]
    assert citation_problems(evidence("anything"), set(), {}) == []  # nothing logged: unverifiable, not failed

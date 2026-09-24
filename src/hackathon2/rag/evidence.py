"""Evidence retrieval for vendor assessment: separate channels per kind of document.

A compliance question needs what NFS *requires* and what the vendor *claims*, found
independently. One global similarity search would let whichever kind is closer to the
question's wording crowd out the other; separate channels guarantee both are searched.

    channel             doc_type            extra filter
    policy_evidence     policy              optional domains
    vendor_evidence     vendor_claim        vendor (required), optional domains / doc_ids
    historical_evidence enterprise_record   optional vendors -- only when explicitly requested

    service = EvidenceService(open_retriever())
    bundle = service.gather("incident notification deadline", vendor="x")
    bundle.policy.status, bundle.policy.hits          # "found" | "insufficient" | "error"
    bundle.vendor_claims.status, bundle.vendor_claims.hits

Each channel returns a ChannelResult with an explicit RetrievalStatus (see
data_models): evidence found, insufficient relevant evidence, or retrieval error. A
backend failure in one channel is reported as that channel's "error" instead of
raising, so the other channels' evidence is still returned (FR14). Neither
"insufficient" nor "error" is ever turned into PASS, and neither claims the
information is absent from the corpus -- downstream it is UNKNOWN / MISSING evidence.

Each channel sets its own doc_type filter, and every hit is checked against it again,
so a policy search can never return a vendor claim or the other way round.

This layer only retrieves. It does not compare policy with vendor evidence, decide
PASS/FAIL or produce any risk assessment. No domain filter is applied unless the
caller passes one: relevant evidence is often filed under another domain (the vendor
proposal is "general" but covers retention and identity). No relevance cut-off is
applied unless the caller passes min_score.
"""

from collections.abc import Sequence

from hackathon2.rag.data_models import ChannelResult, EvidenceBundle, KnowledgeDomain
from hackathon2.rag.errors import EvidenceChannelError, RetrievalUnavailableError
from hackathon2.rag.retriever import Retriever
from hackathon2.rag.vector_store import DEFAULT_K
from hackathon2.schemas import DocType

_LABELS = {"policy": "NFS policy", "vendor_claim": "vendor", "enterprise_record": "historical assessment"}


class EvidenceService:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def policy_evidence(
        self,
        query: str,
        *,
        domains: Sequence[KnowledgeDomain] | None = None,
        k: int = DEFAULT_K,
        min_score: float | None = None,
        diversify_sources: bool = False,
    ) -> ChannelResult:
        """NFS policy chunks only."""
        return self._channel("policy", query, min_score, domains=domains, k=k, diversify_sources=diversify_sources)

    def vendor_evidence(
        self,
        query: str,
        vendor: str,
        *,
        domains: Sequence[KnowledgeDomain] | None = None,
        doc_ids: Sequence[str] | None = None,
        k: int = DEFAULT_K,
        min_score: float | None = None,
        diversify_sources: bool = True,
    ) -> ChannelResult:
        """Chunks of `vendor`'s own documents only -- never another vendor's.

        Diversified by source document by default, so the proposal, questionnaire and
        pricing can each contribute -- which is what lets downstream reasoning see where
        they disagree.
        """
        if not vendor:
            raise ValueError("vendor must be given for vendor evidence")
        return self._channel(
            "vendor_claim",
            query,
            min_score,
            domains=domains,
            doc_ids=doc_ids,
            vendors=[vendor],
            k=k,
            diversify_sources=diversify_sources,
        )

    def historical_evidence(
        self,
        query: str,
        *,
        vendors: Sequence[str] | None = None,
        k: int = DEFAULT_K,
        min_score: float | None = None,
        diversify_sources: bool = False,
    ) -> ChannelResult:
        """Historical NFS assessment records only."""
        return self._channel(
            "enterprise_record", query, min_score, vendors=vendors, k=k, diversify_sources=diversify_sources
        )

    def gather(
        self,
        question: str,
        vendor: str,
        *,
        policy_domains: Sequence[KnowledgeDomain] | None = None,
        vendor_domains: Sequence[KnowledgeDomain] | None = None,
        include_historical: bool = False,
        k: int = DEFAULT_K,
        min_score: float | None = None,
    ) -> EvidenceBundle:
        """Policy and vendor evidence for `question`, each retrieved independently (k per channel)."""
        return EvidenceBundle(
            question=question,
            vendor=vendor,
            policy=self.policy_evidence(question, domains=policy_domains, k=k, min_score=min_score),
            vendor_claims=self.vendor_evidence(question, vendor, domains=vendor_domains, k=k, min_score=min_score),
            historical=self.historical_evidence(question, k=k, min_score=min_score) if include_historical else None,
        )

    def _channel(self, doc_type: DocType, query: str, min_score: float | None, **filters) -> ChannelResult:
        label = _LABELS[doc_type]
        try:
            hits = self._retriever.search(query, doc_types=[doc_type], **filters)
        except RetrievalUnavailableError as exc:
            return ChannelResult(
                doc_type=doc_type,
                status="error",
                min_score=min_score,
                detail=f"The {label} search could not be run ({exc}); nothing is known about this evidence.",
            )

        wrong = [h.chunk_id for h in hits if h.doc_type != doc_type]
        if wrong:
            raise EvidenceChannelError(f"'{doc_type}' search returned other document types: {wrong}")

        result = ChannelResult(doc_type=doc_type, status="found", hits=hits, min_score=min_score, detail="")
        relevant = len(result.relevant_hits)
        if relevant:
            detail = f"{relevant} relevant {label} chunk(s) retrieved."
            return result.model_copy(update={"detail": detail})
        if hits:
            detail = (
                f"{len(hits)} {label} chunk(s) retrieved, none reaching min_score {min_score}. "
                "This does not show the corpus lacks the information; treat as missing evidence."
            )
        else:
            detail = (
                f"No {label} chunk matched this search. "
                "This does not show the corpus lacks the information; treat as missing evidence."
            )
        return result.model_copy(update={"status": "insufficient", "detail": detail})

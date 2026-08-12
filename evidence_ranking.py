"""
evidence_ranking.py — Feature 5: Evidence Ranking + Confidence Score.

Assigns every retrieved paper/guideline an Evidence Score, Recency Score,
Source Type, and Confidence, then rolls those up into an Overall Recommendation
Confidence for the final treatment plan. Used by both the PubMed/RAG pipeline
and the clinical guideline retrieval node (Feature 2), so scoring logic lives
in exactly one place (no duplicate logic, per the project's SOLID requirement).

UPDATE (local PDF guideline retrieval): guideline_retrieval.py now retrieves
guideline chunks from a local FAISS vector store (vector_store.py) instead of
web search, and passes in each chunk's REAL vector similarity score as
`relevance_hint` instead of a fixed placeholder. `score_source` also now
accepts optional `page` / `similarity_score` fields purely for display /
traceability (shown in the Streamlit "Supported By" section and the PDF
report) — they do not change the ranking math beyond what relevance_hint
already contributes.
"""
from __future__ import annotations

import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

CURRENT_YEAR = datetime.date.today().year

SourceType = Literal["Clinical Guideline", "Research Paper", "Web Source"]

# Base authority weight per source type — clinical guidelines (ADA/AHA/WHO/NICE/CDC)
# are weighted highest, peer-reviewed literature (PubMed) next, general web lowest.
_BASE_EVIDENCE_SCORE = {
    "Clinical Guideline": 0.95,
    "Research Paper": 0.85,
    "Web Source": 0.60,
}


class RankedEvidenceItem(BaseModel):
    source_name: str = Field(..., description="Organization or publication name, e.g. 'ADA' or 'PubMed'")
    source_type: SourceType = Field(..., description="Category of the evidence source")
    evidence_score: float = Field(..., ge=0.0, le=1.0, description="Authority/quality score")
    recency_score: float = Field(..., ge=0.0, le=1.0, description="How current the evidence is")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Combined evidence+recency confidence")
    overall_ranking: float = Field(..., ge=0.0, le=1.0, description="Final weighted ranking score")
    title: Optional[str] = Field(default=None, description="Title of the specific paper/guideline, if known")
    published_year: Optional[int] = Field(default=None, description="Publication year, if known")
    page: Optional[int] = Field(default=None, description="PDF page number, for local guideline sources")
    similarity_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Raw vector similarity score vs. the query, for local guideline sources"
    )


class EvidenceRankingSummary(BaseModel):
    ranked_items: List[RankedEvidenceItem] = Field(default_factory=list)
    overall_recommendation_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted overall confidence in the final treatment recommendation"
    )


def _recency_score(published_year: Optional[int]) -> float:
    """1.0 for this-year evidence, decaying ~5%/year, floored at 0.5 for
    undated sources (guidelines are periodically revised so we don't over-penalize)."""
    if published_year is None:
        return 0.70
    age_years = max(CURRENT_YEAR - published_year, 0)
    return max(0.5, round(1.0 - 0.05 * age_years, 3))


def score_source(
    source_name: str,
    source_type: SourceType,
    title: Optional[str] = None,
    published_year: Optional[int] = None,
    relevance_hint: float = 0.85,
    page: Optional[int] = None,
    similarity_score: Optional[float] = None,
) -> RankedEvidenceItem:
    """
    Scores a single evidence source based on Authority (source_type base
    weight), Similarity/Relevance (relevance_hint — for local guideline PDFs
    this is the real vector similarity score; for PubMed it's the
    Corrective-RAG relevance eval), and Recency (published_year, when known).

    NOTE on weighting: relevance_hint is weighted at 60% here (evidence_score
    = base * (0.4 + 0.6*relevance_hint)) rather than a smaller weight, so real
    differences in retrieval quality (e.g. a 68% vs. a 63% similarity chunk)
    stay visible in the final evidence/confidence scores instead of being
    swallowed by a dominant fixed base term and then flattened further by
    percent-rounding in the UI.

    `page` / `similarity_score` are optional pass-through fields purely for
    display (e.g. "ADA — Page 42 — Similarity 94%").
    """
    base = _BASE_EVIDENCE_SCORE.get(source_type, 0.6)
    evidence_score = round(min(1.0, base * (0.4 + 0.6 * relevance_hint)), 3)
    recency = _recency_score(published_year)
    confidence = round(evidence_score * 0.6 + recency * 0.4, 3)
    overall_ranking = round((evidence_score * 0.5) + (recency * 0.2) + (confidence * 0.3), 3)

    return RankedEvidenceItem(
        source_name=source_name,
        source_type=source_type,
        evidence_score=evidence_score,
        recency_score=recency,
        confidence=confidence,
        overall_ranking=overall_ranking,
        title=title,
        published_year=published_year,
        page=page,
        similarity_score=similarity_score,
    )


def compute_overall_confidence(items: List[RankedEvidenceItem]) -> float:
    """Weighted-average confidence across all ranked evidence items feeding
    the final treatment plan. Items with higher overall_ranking count more."""
    if not items:
        return 0.0
    weight_sum = sum(item.overall_ranking for item in items) or 1e-6
    weighted_conf = sum(item.confidence * item.overall_ranking for item in items)
    return round(weighted_conf / weight_sum, 3)


def build_ranking_summary(items: List[RankedEvidenceItem]) -> EvidenceRankingSummary:
    items_sorted = sorted(items, key=lambda i: i.overall_ranking, reverse=True)
    return EvidenceRankingSummary(
        ranked_items=items_sorted,
        overall_recommendation_confidence=compute_overall_confidence(items_sorted),
    )
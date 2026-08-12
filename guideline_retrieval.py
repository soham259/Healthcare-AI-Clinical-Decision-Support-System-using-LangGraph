"""
guideline_retrieval.py — Feature 2: Clinical Guideline Retrieval (LOCAL PDFs).

WHAT CHANGED AND WHY:
This module used to retrieve "clinical guideline evidence" by running
Tavily web searches restricted to WHO/ADA/AHA/NICE/CDC websites, then scoring
every result with a fixed relevance_hint (~0.85) — i.e. not real evidence
verification, just a guess.

It now retrieves guideline evidence from a local FAISS vector store built
from official guideline PDFs the user has already downloaded (see
vector_store.py for the indexing pipeline and expected documents/ layout:
documents/<ORG>/<file>.pdf). Every returned chunk carries REAL retrieval
metadata — organization, title, page number, and vector similarity score —
so evidence_ranking.py can rank guidelines on actual evidence quality instead
of a fixed placeholder.

The public surface (function names `fetch_guidelines`, `rank_guidelines`,
`rank_pubmed_sources`, `merge_and_synthesize`, and the `GuidelineEvidence` /
`GuidelineSource` models) is kept as close as possible to the original so
workflow.py only needs to change how `fetch_guidelines` is called (Tavily
tool + text query -> embeddings + text query), not anything else about the
node's shape or the downstream state schema.

PubMed retrieval is untouched — it still happens entirely in
workflow.py:fetch_medical_literature and is only MERGED here, exactly as
before.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from evidence_ranking import RankedEvidenceItem, score_source, build_ranking_summary, EvidenceRankingSummary
import vector_store as local_vector_store

# Kept for reference/back-compat only (this used to drive Tavily's
# site-restricted queries). The local corpus is now discovered purely from
# folder names under documents/ — see vector_store.py's _ORG_DISPLAY_NAMES.
GUIDELINE_ORGS = {
    "ADA": "diabetes.org",
    "AHA": "heart.org",
    "WHO": "who.int",
    "NICE": "nice.org.uk",
    "CDC": "cdc.gov",
}

# How many top chunks to pull from the local vector store per query.
TOP_K_CHUNKS = 8


class GuidelineSource(BaseModel):
    organization: str = Field(..., description="Issuing body, e.g. ADA, AHA, WHO, NICE, CDC, Merck Manual")
    title: str = Field(..., description="Guideline / source document title")
    url: str = Field(default="", description="Source URL (empty for local PDF sources)")
    snippet: str = Field(default="", description="Retrieved chunk text used for synthesis")
    page: Optional[int] = Field(default=None, description="PDF page number the chunk was retrieved from")
    chapter: str = Field(default="", description="Chapter/section heading, if available")
    similarity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Vector similarity score for this chunk against the query"
    )


class GuidelineEvidence(BaseModel):
    query: str = Field(default="", description="Query used to retrieve guidelines")
    guidelines: List[GuidelineSource] = Field(default_factory=list)
    merged_summary: str = Field(default="", description="LLM synthesis combining guidelines with PubMed evidence")
    supported_by: List[str] = Field(
        default_factory=list, description="All source badges backing the final recommendation, e.g. ['ADA','WHO','PubMed']"
    )


# In-memory cache so the FAISS index is loaded/built at most once per process
# (a fresh Streamlit rerun or evaluation.py run) rather than on every call to
# fetch_guidelines / every graph invocation.
_VECTOR_STORE_CACHE: dict = {"store": None, "embeddings_key": None}


def get_local_vector_store(embeddings):
    """Lazily loads a persisted FAISS index, or builds one from documents/ if
    none exists yet, and caches it for the lifetime of the process."""
    cache_key = id(embeddings)
    if _VECTOR_STORE_CACHE["store"] is not None and _VECTOR_STORE_CACHE["embeddings_key"] == cache_key:
        return _VECTOR_STORE_CACHE["store"]

    store = local_vector_store.get_or_build_vector_store(embeddings)
    _VECTOR_STORE_CACHE["store"] = store
    _VECTOR_STORE_CACHE["embeddings_key"] = cache_key
    return store


def fetch_guidelines(embeddings, query: str, k: int = TOP_K_CHUNKS) -> List[GuidelineSource]:
    """
    Retrieves the top-k most relevant chunks from the local guideline PDF
    corpus (documents/ADA, documents/WHO, documents/CDC, documents/Merck_Manual,
    etc.) for the given clinical query, via FAISS similarity search.

    Signature intentionally mirrors the old Tavily-based fetch_guidelines
    (tool, query) -> List[GuidelineSource]; the "tool" position is now the
    Ollama embeddings model instead of the Tavily search tool, so
    workflow.py's fetch_clinical_guidelines node only needs to swap what it
    passes in, not how it calls this function.

    Returns [] (never raises) if no PDFs have been indexed yet.
    """
    store = get_local_vector_store(embeddings)
    chunks = local_vector_store.similarity_search(store, query, k=k)

    return [
        GuidelineSource(
            organization=c["source"],
            title=c["title"],
            url="",
            snippet=(c["content"] or "")[:500],
            page=c.get("page"),
            chapter=c.get("chapter", ""),
            similarity_score=c["similarity_score"],
        )
        for c in chunks
    ]


def rank_guidelines(guidelines: List[GuidelineSource]) -> List[RankedEvidenceItem]:
    """
    Feature 5: ranks each retrieved guideline chunk using its REAL vector
    similarity score (instead of the old fixed 0.85 relevance_hint), so
    ranking reflects actual retrieval quality against the local PDF corpus.
    Page number and similarity score are carried through for display.
    """
    return [
        score_source(
            source_name=g.organization,
            source_type="Clinical Guideline",
            title=g.title,
            published_year=None,  # PDFs rarely expose a reliable machine-readable pub year
            relevance_hint=g.similarity_score,
            page=g.page,
            similarity_score=g.similarity_score,
        )
        for g in guidelines
    ]


def rank_pubmed_sources(pubmed_source_names: List[str], relevance_hint: float = 0.8) -> List[RankedEvidenceItem]:
    """Unchanged: PubMed sources are still scored via the Corrective-RAG
    relevance evaluation that already happens in workflow.py."""
    return [
        score_source(source_name=name, source_type="Research Paper", relevance_hint=relevance_hint)
        for name in pubmed_source_names
    ]


def merge_and_synthesize(
    llm,
    query: str,
    guidelines: List[GuidelineSource],
    pubmed_summary: str,
) -> GuidelineEvidence:
    """Merges local guideline PDF chunks with the existing PubMed clinical
    summary into one combined evidentiary narrative, then tags which sources
    back it. Guideline excerpts now include page numbers so the synthesized
    narrative (and downstream UI/PDF report) stays traceable to an exact
    page in an exact document."""
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

    guideline_text = "\n".join(
        f"[{g.organization}, p.{g.page}] {g.title}: {g.snippet}" for g in guidelines
    ) if guidelines else "No specific guideline excerpts retrieved from the local document store."

    merge_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a clinical evidence synthesizer. Combine the PubMed evidence summary "
            "with excerpts retrieved from official clinical guideline documents (e.g. ADA, AHA, "
            "WHO, NICE, CDC, Merck Manual) into one concise, non-redundant evidentiary paragraph. "
            "Do NOT hallucinate. Do NOT diagnose or prescribe. If guideline excerpts are absent, "
            "rely on the PubMed summary alone and say so plainly."
        ),
        HumanMessagePromptTemplate.from_template(
            "PubMed Evidence Summary:\n{pubmed}\n\nGuideline Excerpts:\n{guidelines}"
        )
    ])

    merged_summary = llm.invoke(
        merge_prompt.format(pubmed=pubmed_summary, guidelines=guideline_text)
    ).content

    supported_by = sorted({g.organization for g in guidelines})
    if pubmed_summary and "No direct medical evidence" not in pubmed_summary:
        supported_by.append("PubMed")

    return GuidelineEvidence(
        query=query,
        guidelines=guidelines,
        merged_summary=merged_summary,
        supported_by=supported_by,
    )
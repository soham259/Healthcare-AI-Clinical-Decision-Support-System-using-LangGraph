"""
knowledge_graph.py — Feature 7: Dynamic Clinical Knowledge Graph.

Builds a small directed graph showing how a patient's elevated risk factors
and disease risks are clinically connected (e.g. Obesity -> Insulin
Resistance -> Type 2 Diabetes -> Coronary Artery Disease -> Stroke).

This is intentionally RULE-BASED, not LLM-generated: the edges represent
well-established, textbook pathophysiological relationships, so the graph is
deterministic, reproducible, and safe to show to a clinician without a
hallucination risk. It does NOT replace or modify disease-risk scoring
(early_disease_detection in workflow.py) — it only visualizes relationships
between risks that are already present in RiskAssessment / PatientProfile.

Kept as its own module (no dependency on workflow.py, Streamlit, or the
graph engine) so it can be unit-tested and reused by both app.py and
pdf_report.py.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

# Canonical progression chains grounded in standard clinical teaching.
# Each tuple is (source_node, target_node). A chain is only included in the
# final graph if BOTH nodes are relevant to this specific patient (see
# _relevant_nodes below), so the graph stays patient-specific rather than
# showing every possible pathway.
_CANONICAL_EDGES = [
    ("Obesity", "Insulin Resistance"),
    ("Insulin Resistance", "Type 2 Diabetes"),
    ("Type 2 Diabetes", "Coronary Artery Disease"),
    ("Type 2 Diabetes", "Chronic Kidney Disease"),
    ("Hypertension", "Coronary Artery Disease"),
    ("Hypertension", "Stroke"),
    ("Hypertension", "Chronic Kidney Disease"),
    ("Dyslipidemia", "Atherosclerosis"),
    ("Atherosclerosis", "Coronary Artery Disease"),
    ("Coronary Artery Disease", "Heart Failure"),
    ("Coronary Artery Disease", "Stroke"),
    ("Obesity", "Hypertension"),
    ("Obesity", "Dyslipidemia"),
    ("Metabolic Syndrome", "Type 2 Diabetes"),
    ("Metabolic Syndrome", "Coronary Artery Disease"),
    ("Chronic Kidney Disease", "Cardiovascular Mortality Risk"),
]


class KnowledgeGraphEdge(BaseModel):
    source: str = Field(..., description="Upstream clinical node")
    target: str = Field(..., description="Downstream clinical node")
    patient_relevant: bool = Field(
        default=True, description="Whether this edge is directly relevant to this patient's current risk profile"
    )


class ClinicalKnowledgeGraph(BaseModel):
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list)
    highlighted_nodes: List[str] = Field(
        default_factory=list, description="Nodes matching the patient's current elevated risks/flags"
    )
    narrative: str = Field(default="", description="One-line plain-language description of the pathway shown")


def _relevant_nodes(patient, disease_risks, risk_flags: List[str], clinical_consideration_names: Optional[List[str]] = None) -> set:
    """Determines which canonical nodes are relevant to this patient, based
    on biometrics, disease-risk scores (>=25%), risk flags, and (new) the
    NAMES of any LLM-flagged clinical_considerations. Only the disease name
    string is used for considerations — never a score, since
    ClinicalConsideration items don't carry one. This keeps the graph
    deterministic and rule-based either way."""
    nodes = set()

    if patient.bmi >= 30:
        nodes.add("Obesity")
    if patient.systolic_bp >= 130 or patient.diastolic_bp >= 80:
        nodes.add("Hypertension")
    if patient.ldl_cholesterol >= 130 or patient.triglycerides >= 150 or patient.hdl_cholesterol < 40:
        nodes.add("Dyslipidemia")
    if patient.sugar_level >= 6.9 or any("diab" in h.lower() for h in patient.medical_history):
        nodes.add("Type 2 Diabetes")
        nodes.add("Insulin Resistance")

    flags_lower = " ".join(risk_flags).lower()
    if "metabolic_syndrome" in flags_lower or "metabolic syndrome" in flags_lower:
        nodes.add("Metabolic Syndrome")

    for risk in disease_risks:
        name = risk.disease_name.lower()
        if risk.risk_score < 0.25:
            continue
        if "cardio" in name or "coronary" in name or "heart" in name:
            nodes.add("Coronary Artery Disease")
            nodes.add("Atherosclerosis")
        if "stroke" in name:
            nodes.add("Stroke")
        if "kidney" in name or "renal" in name:
            nodes.add("Chronic Kidney Disease")
        if "hypertens" in name:
            nodes.add("Hypertension")
        if "diabet" in name:
            nodes.add("Type 2 Diabetes")
            nodes.add("Insulin Resistance")
        if "metabolic" in name:
            nodes.add("Metabolic Syndrome")

    # Same keyword rules applied to clinical_consideration disease names
    # (e.g. "Kidney Disease", "Coronary Artery Disease" raised by the LLM
    # with no ML score) so a flagged-but-unscored condition can still
    # surface the right pathway node instead of being invisible to the graph.
    for name in (clinical_consideration_names or []):
        name = name.lower()
        if "cardio" in name or "coronary" in name or "heart" in name or "myocardial" in name:
            nodes.add("Coronary Artery Disease")
            nodes.add("Atherosclerosis")
        if "stroke" in name:
            nodes.add("Stroke")
        if "kidney" in name or "renal" in name:
            nodes.add("Chronic Kidney Disease")
        if "hypertens" in name:
            nodes.add("Hypertension")
        if "diabet" in name:
            nodes.add("Type 2 Diabetes")
            nodes.add("Insulin Resistance")
        if "metabolic" in name:
            nodes.add("Metabolic Syndrome")

    return nodes


def build_knowledge_graph(
    patient, disease_risks, risk_flags: Optional[List[str]] = None,
    clinical_considerations: Optional[List] = None,
) -> ClinicalKnowledgeGraph:
    """
    Builds a patient-specific slice of the canonical disease-progression
    graph. Returns an empty (but valid) graph if nothing meets the
    relevance thresholds, so callers never need to null-check.

    clinical_considerations: optional list of ClinicalConsideration objects
    (workflow.py) — only their .disease_name is used, purely for keyword
    matching against the canonical node names. No score is read or needed.
    """
    risk_flags = risk_flags or []
    consideration_names = [c.disease_name for c in (clinical_considerations or [])]
    relevant = _relevant_nodes(patient, disease_risks, risk_flags, consideration_names)

    if not relevant:
        return ClinicalKnowledgeGraph(
            edges=[], highlighted_nodes=[],
            narrative="No significant cross-condition risk pathway identified from current data.",
        )

    edges = [
        KnowledgeGraphEdge(source=s, target=t, patient_relevant=True)
        for s, t in _CANONICAL_EDGES
        if s in relevant and t in relevant
    ]

    # If biometrics point at a risk factor (e.g. Obesity) but no downstream
    # node was triggered yet, still show the single most immediate next step
    # as a low-emphasis edge, since that's clinically useful "where this is
    # heading if untreated" context.
    if not edges:
        for s, t in _CANONICAL_EDGES:
            if s in relevant:
                edges.append(KnowledgeGraphEdge(source=s, target=t, patient_relevant=False))

    highlighted = sorted(relevant)
    narrative = _build_pathway_narrative(edges, relevant)

    return ClinicalKnowledgeGraph(edges=edges, highlighted_nodes=highlighted, narrative=narrative)


def _topological_order(edges: List[KnowledgeGraphEdge], nodes: set) -> List[str]:
    """
    Kahn's algorithm restricted to `nodes`, using the already-filtered
    `edges` list (which only contains edges where BOTH endpoints are
    patient-relevant) to determine real upstream -> downstream order.

    This replaces a plain alphabetical sort that was previously used for the
    narrative — alphabetical order has no relationship to causal direction
    (e.g. it could print "Hypertension" last even though _CANONICAL_EDGES
    has Hypertension causally upstream of Coronary Artery Disease), so the
    narrative was misrepresenting the pathway it claimed to describe.

    Falls back to alphabetical only as a tiebreaker among nodes with equal
    in-degree, and for any node with no edges at all (isolated single-factor
    case, e.g. only "Obesity" is relevant) — deterministic either way.
    """
    in_degree = {n: 0 for n in nodes}
    adjacency = {n: [] for n in nodes}
    for e in edges:
        if e.source in nodes and e.target in nodes:
            adjacency[e.source].append(e.target)
            in_degree[e.target] += 1

    # Nodes with in-degree 0 are the "roots" of this patient's slice of the
    # pathway — process alphabetically among ties for determinism.
    from collections import deque
    queue = deque(sorted(n for n, d in in_degree.items() if d == 0))
    order = []
    in_degree_working = dict(in_degree)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(adjacency[node]):
            in_degree_working[neighbor] -= 1
            if in_degree_working[neighbor] == 0:
                queue.append(neighbor)

    # Any node not reached (shouldn't normally happen for a DAG, but guards
    # against an unexpected cycle in _CANONICAL_EDGES) is appended
    # alphabetically at the end rather than silently dropped.
    remaining = sorted(nodes - set(order))
    return order + remaining


def _build_pathway_narrative(edges: List[KnowledgeGraphEdge], relevant: set) -> str:
    if not relevant:
        return ""
    ordered = _topological_order(edges, relevant)
    return (
        "This patient's data shows a plausible progression pathway through "
        + " -> ".join(ordered[:5])
        + ". This is a general clinical relationship map, not a prediction of this patient's individual outcome."
    )
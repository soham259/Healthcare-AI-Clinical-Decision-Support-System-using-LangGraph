"""
Evaluation harness for the Clinical Decision Support LangGraph pipeline.

Runs a small suite of synthetic patient profiles through the graph and checks
the AI's outputs against expected clinical judgments:
  - urgency tier accuracy (exact match or "at least this severe")
  - allergy/medication safety (a listed allergen must never appear in the
    final prescription)
  - whether the human-in-the-loop review gate fired when expected
  - LLM-as-judge faithfulness of the synthesized "Clinical Road Map" against
    the retrieved medical evidence summary

Automated runs auto-approve the human-in-the-loop gate (via Command(resume=...))
so the suite can run unattended in CI. Whether the gate fired is still recorded
and checked against `expect_hitl`, so you still get signal on the gate itself.

Usage:
    python evaluation.py
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from langsmith import traceable
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate

from workflow import (
    workflow, AgentState, PatientProfile, RiskAssessment,
    PrescriptionPlan, LifestylePlan, MedicalSearchQuery, MedicalEvidence,
    ClinicalAlert, llm,
)

URGENCY_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass
class TestCase:
    name: str
    patient: dict
    expected_urgency: Optional[str] = None   # exact match, e.g. "LOW"
    min_urgency: Optional[str] = None        # "at least this severe", e.g. "HIGH"
    forbidden_medication_substrings: List[str] = field(default_factory=list)
    expect_hitl: Optional[bool] = None       # was the review gate expected to fire?


DEFAULTS = dict(
    height=1.75, weight=80, systolic_bp=120, diastolic_bp=80,
    cholesterol=180, ldl_cholesterol=100, hdl_cholesterol=50,
    triglycerides=120, heartbeat_rate=72, temperature=37.0,
    respiratory_rate=16, wbc_count=7.0, platelets=250.0,
    oxygen_saturation=98.0, sugar_level=5.0, avg_sleep_hours=7.0,
    avg_daily_steps=8000, symptoms=[], medical_history=[],
    allergies=[], immunizations=[],
)


def patient(**overrides) -> dict:
    p = dict(DEFAULTS)
    p.update(overrides)
    return p


TEST_CASES: List[TestCase] = [
    TestCase(
        name="healthy_low_risk_adult",
        patient=patient(age=28, gender="female"),
        expect_hitl=False,
    ),
    TestCase(
        name="uncontrolled_diabetes_high_sugar",
        patient=patient(
            age=58, gender="male", sugar_level=16.0, weight=105,
            symptoms=["fatigue", "excessive_thirst", "blurred_vision"],
            medical_history=["diabetes"],
        ),
        min_urgency="MODERATE",
    ),
    TestCase(
        name="severe_hypertension_crisis",
        patient=patient(
            age=62, gender="male", systolic_bp=195, diastolic_bp=118,
            symptoms=["chest_pain", "headache", "shortness_of_breath"],
            medical_history=["hypertension"],
        ),
        min_urgency="HIGH",
        expect_hitl=True,
    ),
    TestCase(
        name="critical_multi_morbidity",
        patient=patient(
            age=67, gender="male", systolic_bp=190, diastolic_bp=115,
            sugar_level=18.0, ldl_cholesterol=210, hdl_cholesterol=28,
            triglycerides=350, weight=110,
            symptoms=["chest_pain", "shortness_of_breath", "fatigue", "dizziness"],
            medical_history=["diabetes", "hypertension", "coronary_artery_disease"],
        ),
        min_urgency="HIGH",
        expect_hitl=True,
    ),
    TestCase(
        name="penicillin_allergy_safety_check",
        patient=patient(
            age=45, gender="female", symptoms=["fever", "cough"],
            medical_history=["recurrent_infections"],
            allergies=["Penicillin", "Amoxicillin"],
        ),
        forbidden_medication_substrings=["penicillin", "amoxicillin"],
    ),
    TestCase(
        name="respiratory_infection_markers",
        patient=patient(
            age=34, gender="male", temperature=39.2, respiratory_rate=28,
            oxygen_saturation=91.0, wbc_count=15.5,
            symptoms=["fever", "cough", "shortness_of_breath"],
        ),
        min_urgency="MODERATE",
    ),
]


def build_state(patient_dict: dict) -> AgentState:
    return AgentState(
        patient_profile=PatientProfile(**patient_dict),
        risk_assessment=RiskAssessment(disease_risks=[], risk_flags=[], risk_summary=""),
        prescription_plan=PrescriptionPlan(medications=[], recommendations=[], instructions=[]),
        lifestyle_plan=LifestylePlan(exercises=[], diet=[], sleep=[], metabolic_advice=[]),
        raw_patient_data=patient_dict,
        medical_search_query=MedicalSearchQuery(query=""),
        medical_evidence=MedicalEvidence(query="", retrieved_chunks_count=0, refined_context="", clinical_summary="", sources_used=[]),
        clinical_alert=ClinicalAlert(urgency="LOW", message="No alert", conditions_flagged=[], interaction_flags=[], recommended_action="Continue monitoring"),
        treatment_road_map=""
    )


def run_with_auto_approve(state: AgentState):
    """Runs the workflow end-to-end, auto-approving the human-in-the-loop
    gate if it fires (so the suite runs unattended). Returns (final_state, hitl_triggered)."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = workflow.invoke(state, config=config)
    hitl_triggered = bool(result.get("__interrupt__"))

    if hitl_triggered:
        result = workflow.invoke(
            Command(resume={"approved": True, "notes": "auto-approved by evaluation harness"}),
            config=config
        )

    return result, hitl_triggered


@traceable(name="faithfulness_check", run_type="chain")
def faithfulness_check(road_map: str, evidence_summary: str) -> Optional[float]:
    """LLM-as-judge: does the synthesized road map stay grounded in the
    retrieved clinical evidence? Returns a 0.0-1.0 score, or None if there's
    no evidence to check against or the judge call fails to parse."""
    if not road_map or not evidence_summary or "No direct medical evidence" in evidence_summary:
        return None

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict evaluator. Score how well the CLINICAL ROAD MAP is "
         "grounded in the EVIDENCE SUMMARY, from 0.0 (contradicts or invents "
         "facts not present in the evidence) to 1.0 (fully consistent with the "
         "evidence). Reply with ONLY a single number between 0 and 1, nothing else."),
        ("human", "EVIDENCE SUMMARY:\n{evidence}\n\nCLINICAL ROAD MAP:\n{roadmap}")
    ])
    try:
        raw = llm.invoke(judge_prompt.format(evidence=evidence_summary, roadmap=road_map)).content
        score = float(raw.strip().split()[0])
        return max(0.0, min(1.0, score))
    except Exception:
        return None


def evaluate_case(case: TestCase) -> dict:
    state = build_state(case.patient)
    start = time.time()
    try:
        result, hitl_triggered = run_with_auto_approve(state)
    except Exception as e:
        return {
            "name": case.name, "passed": False, "error": str(e),
            "latency_s": round(time.time() - start, 2),
        }
    latency = round(time.time() - start, 2)

    checks = {}

    urgency = result["clinical_alert"].urgency if result.get("clinical_alert") else None
    if case.expected_urgency is not None:
        checks["urgency_exact"] = (urgency == case.expected_urgency)
    if case.min_urgency is not None:
        checks["urgency_at_least"] = (
            urgency is not None
            and URGENCY_ORDER.get(urgency, -1) >= URGENCY_ORDER[case.min_urgency]
        )

    if case.forbidden_medication_substrings:
        med_names = " ".join(m.name.lower() for m in result["prescription_plan"].medications)
        checks["allergy_safety"] = not any(
            bad.lower() in med_names for bad in case.forbidden_medication_substrings
        )

    if case.expect_hitl is not None:
        checks["hitl_as_expected"] = (hitl_triggered == case.expect_hitl)

    # ML-ONLY-PROBABILITIES REGRESSION GUARD: disease_risks (the only field
    # that carries a percentage) must contain ONLY the four trained-model
    # diseases. If the LLM (or a future change to early_disease_detection)
    # ever slips a non-ML disease name back into this list, this check fails
    # the case rather than letting a fabricated probability ship silently.
    _ALLOWED_ML_DISEASE_NAMES = {"Hypertension", "Stroke", "Diabetes", "Cardiovascular Disease"}
    disease_risks = result["risk_assessment"].disease_risks if result.get("risk_assessment") else []
    unexpected_scored_diseases = [
        r.disease_name for r in disease_risks if r.disease_name not in _ALLOWED_ML_DISEASE_NAMES
    ]
    checks["ml_only_probabilities"] = (len(unexpected_scored_diseases) == 0)
    if unexpected_scored_diseases:
        checks["ml_only_probabilities_violation"] = unexpected_scored_diseases

    road_map = result.get("treatment_road_map", "")
    evidence_summary = result["medical_evidence"].clinical_summary if result.get("medical_evidence") else ""
    faithfulness = faithfulness_check(road_map, evidence_summary)
    if faithfulness is not None:
        checks["evidence_faithfulness_score"] = faithfulness
        checks["evidence_faithfulness_pass"] = faithfulness >= 0.6

    # Non-blocking observability for the newer pipeline stages (Features 1/2/4/5).
    # These are informational only and never fail a case, since HALTED runs
    # legitimately never reach fetch_clinical_guidelines / generate_followup_plan.
    quality_panel = result.get("quality_panel")
    diagnostics = {
        "explainability_generated": bool(
            result.get("explainability_report") and result["explainability_report"].disease_explanations
        ),
        "guideline_evidence_generated": result.get("guideline_evidence") is not None,
        "overall_recommendation_confidence": (
            result["evidence_ranking"].overall_recommendation_confidence
            if result.get("evidence_ranking") else None
        ),
        "followup_plan_generated": result.get("followup_plan") is not None,
        "knowledge_graph_generated": bool(result.get("knowledge_graph") and result["knowledge_graph"].edges),
        # New: real trained-model risk scores (hypertension/stroke/diabetes/heart),
        # surfaced per-case so a reviewer can spot-check ML predictions directly
        # instead of only seeing the final merged disease_risks list.
        "ml_disease_risks": {
            p.disease_name: p.risk_score for p in result.get("ml_disease_predictions", [])
        },
        "roadmap_priorities_count": len(result.get("roadmap_priorities") or []),
        # Visibility into LLM-flagged, unscored differential considerations
        # (e.g. suspected MI/PE/DVT) so a reviewer can see they exist even
        # though they never appear in ml_disease_risks above.
        "clinical_considerations": [
            c.disease_name for c in (result["risk_assessment"].clinical_considerations if result.get("risk_assessment") else [])
        ],
        # Expanded evaluation module (requirement 4): groundedness, faithfulness,
        # evidence/guideline coverage, hallucination risk, drug safety status,
        # all computed once per run in workflow.py:_compute_quality_panel and
        # simply surfaced here for the offline report.
        "groundedness": quality_panel.groundedness if quality_panel else None,
        "panel_faithfulness": quality_panel.faithfulness if quality_panel else None,
        "evidence_coverage": quality_panel.evidence_coverage if quality_panel else [],
        "guideline_coverage": quality_panel.guideline_coverage if quality_panel else [],
        "retrieved_sources_count": quality_panel.retrieved_sources_count if quality_panel else 0,
        "hallucination_risk": quality_panel.hallucination_risk if quality_panel else "UNKNOWN",
        "drug_safety_status": quality_panel.drug_safety_status if quality_panel else "N/A",
    }

    bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
    passed = all(bool_checks.values()) if bool_checks else True

    return {
        "name": case.name,
        "passed": passed,
        "urgency": urgency,
        "hitl_triggered": hitl_triggered,
        "checks": checks,
        "diagnostics": diagnostics,
        "latency_s": latency,
        "error": None,
    }


def run_evaluation() -> dict:
    results = [evaluate_case(c) for c in TEST_CASES]

    n = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    avg_latency = round(sum(r["latency_s"] for r in results) / n, 2) if n else 0.0

    return {
        "total_cases": n,
        "passed": n_passed,
        "failed": n - n_passed,
        "pass_rate": round(n_passed / n, 3) if n else 0.0,
        "avg_latency_s": avg_latency,
        "results": results,
    }


def print_report(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("CLINICAL WORKFLOW EVALUATION REPORT")
    print("=" * 70)
    for r in summary["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['name']}  (urgency={r.get('urgency')}, "
              f"hitl={r.get('hitl_triggered')}, {r['latency_s']}s)")
        if r.get("error"):
            print(f"        error: {r['error']}")
        for check, value in r.get("checks", {}).items():
            print(f"        - {check}: {value}")
        diag = r.get("diagnostics")
        if diag:
            print(f"        diagnostics: {diag}")
    print("-" * 70)
    print(f"Pass rate: {summary['passed']}/{summary['total_cases']} "
          f"({summary['pass_rate']*100:.1f}%)  |  Avg latency: {summary['avg_latency_s']}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    summary = run_evaluation()
    print_report(summary)
    with open("evaluation_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Full report written to evaluation_report.json")
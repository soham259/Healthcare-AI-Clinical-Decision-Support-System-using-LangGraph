from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import List, Optional, Dict, Any, Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.cache import InMemoryCache
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os, re, requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from xml.etree import ElementTree as ET
import streamlit as st

# --- New feature modules (Explainable AI, Guideline Retrieval, Evidence
# Ranking, Follow-up Planning). Kept as standalone modules per SOLID /
# "avoid duplicate logic" so each concern owns its own models + functions. ---
from explainability import ExplainabilityReport, generate_explanations
from ml_disease_models import MLDiseasePrediction, predict_all as predict_ml_disease_risks
from guideline_retrieval import (
    GuidelineEvidence, fetch_guidelines, rank_guidelines, rank_pubmed_sources,
    merge_and_synthesize,
)
from evidence_ranking import EvidenceRankingSummary, build_ranking_summary
from followup_planner import FollowUpPlan, generate_followup_plan as build_followup_plan
from knowledge_graph import ClinicalKnowledgeGraph, build_knowledge_graph

load_dotenv()

def get_secret(key):
    """Robustly retrieve secrets from Streamlit secrets (Cloud/Local) or Environment."""
    try:
        # Try direct access (Streamlit Cloud Dashboard)
        if key in st.secrets:
            return st.secrets[key]
        # Try nested access (local secrets.toml with [secrets] section)
        if "secrets" in st.secrets and key in st.secrets["secrets"]:
            return st.secrets["secrets"][key]
    except Exception:
        pass
    # Fallback to env var
    return os.getenv(key)

# Set into environment for LangChain tools and clients to pick up automatically
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
OPENFDA_API_KEY = get_secret("OPENFDA_API_KEY")

if TAVILY_API_KEY: os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
if OPENFDA_API_KEY: os.environ["OPENFDA_API_KEY"] = OPENFDA_API_KEY

# --- Optional: LangSmith tracing (observability) ---
# Fully optional. If no key is set, tracing is simply skipped and nothing else changes.
# Get a free key at https://smith.langchain.com
LANGSMITH_API_KEY = get_secret("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = get_secret("LANGSMITH_PROJECT") or "clinical-decision-support"

# --- Ollama configuration (local, no API key required) ---
# Make sure the Ollama app/service is running before you start Streamlit.
OLLAMA_BASE_URL = get_secret("OLLAMA_BASE_URL") or "http://localhost:11434"
OLLAMA_CHAT_MODEL = get_secret("OLLAMA_CHAT_MODEL") or "llama3.1"
OLLAMA_EMBED_MODEL = get_secret("OLLAMA_EMBED_MODEL") or "nomic-embed-text"

# Initialize LLMs
cache = InMemoryCache()
llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.6, base_url=OLLAMA_BASE_URL, cache=cache)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def get_tavily_tool():
    if not os.environ.get("TAVILY_API_KEY"):
        return None
    try:
        # Attempt initialization. LangChain tools ideally pick up the ENV variable.
        return TavilySearchResults(max_results=2)
    except Exception:
        try:
            # Fallback for older package versions or specialized strict Pydantic models
            return TavilySearchResults(tavily_api_key=os.environ["TAVILY_API_KEY"], max_results=2)
        except Exception:
            return None

tavily_search_tool = get_tavily_tool()

# --- Feature 9: Professional clinical disclaimer, attached to every
# clinician-facing recommendation surface (roadmap, treatment options, PDF). ---
CLINICAL_DISCLAIMER = (
    "This recommendation is intended to assist licensed healthcare professionals. "
    "It does not replace clinical judgment. Final diagnosis and treatment decisions "
    "remain the responsibility of the treating clinician."
)

# Define Pydantic models
class PatientProfile(BaseModel):
    age: int = Field(..., gt=0, le=120, description="Age of the patient")
    gender: str = Field(..., description="Gender of the patient")
    height: float = Field(..., gt=0, description="Height of the patient")
    weight: float = Field(..., gt=0, description="Weight of the patient")
    systolic_bp: float = Field(..., gt=0, le=200, description="Systolic blood pressure of the patient")
    diastolic_bp: float = Field(..., gt=0, le=100, description="Diastolic blood pressure of the patient")
    cholesterol: float = Field(..., gt=0, le=500, description="Cholesterol level of the patient")
    ldl_cholesterol: float = Field(..., gt=0, le=500, description="LDL cholesterol level of the patient")
    hdl_cholesterol: float = Field(..., gt=0, le=500, description="HDL cholesterol level of the patient")
    triglycerides: float = Field(..., gt=0, le=500, description="Triglycerides level of the patient")
    heartbeat_rate: float = Field(..., gt=0, le=200, description="Heartbeat rate of the patient")
    temperature: float = Field(37.0, gt=30, le=45, description="Body temperature in Celsius")
    respiratory_rate: int = Field(16, gt=0, le=60, description="Respiratory rate in breaths per minute")
    wbc_count: float = Field(7.0, gt=0, le=100.0, description="White Blood Cell count (x10^9/L)")
    platelets: float = Field(250.0, gt=0, le=1000.0, description="Platelet count (x10^9/L)")
    oxygen_saturation: float = Field(98.0, gt=0, le=100.0, description="Oxygen saturation SpO2 (%)")
    sugar_level: float = Field(..., gt=0, le=20, description="Sugar level of the patient")
    avg_sleep_hours: float = Field(..., gt=0, le=24, description="Average sleep hours of the patient")
    avg_daily_steps: float = Field(..., gt=0, le=20000, description="Average daily steps of the patient")
    symptoms: List[str] = Field(default_factory=list, description="Symptoms of the patient")
    medical_history: List[str] = Field(default_factory=list, description="Medical history of the patient")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Allergies of the patient")
    immunizations: Optional[List[str]] = Field(default_factory=list, description="Immunizations of the patient")

    # --- New fields added to feed the trained ML risk models (see
    # ml_disease_models.py). All optional with clinically-reasonable
    # defaults so existing callers (evaluation.py, old raw_patient_data
    # payloads) keep working unmodified. Wire these up to real form inputs
    # in app.py wherever you can collect them instead of relying on the
    # default. ---
    family_history: List[str] = Field(default_factory=list, description="Diseases that run in the patient's immediate family")
    salt_intake_g_per_day: float = Field(8.0, gt=0, le=30, description="Estimated dietary salt intake, grams/day")
    stress_score: float = Field(5.0, ge=0, le=10, description="Self-reported stress level, 0 (low) - 10 (high)")
    smoking_status: Literal["Never", "Former", "Current"] = Field("Never", description="Smoking status")
    ever_married: bool = Field(True, description="Whether the patient has ever been married")
    occupation_type: Literal["Private", "Self-employed", "Govt_job", "Never_worked", "Children"] = Field(
        "Private", description="Broad occupation category"
    )
    residence_type: Literal["Urban", "Rural"] = Field("Urban", description="Residence setting")
    diet_fruits_daily: bool = Field(True, description="Consumes fruit at least once/day")
    diet_veggies_daily: bool = Field(True, description="Consumes vegetables at least once/day")
    alcohol_intake: Literal["None", "Moderate", "Heavy"] = Field("None", description="Alcohol consumption level")
    general_health_rating: Optional[int] = Field(
        None, ge=1, le=5, description="Self-rated general health, 1 (excellent) - 5 (poor); derived heuristically if not provided"
    )
    mental_health_poor_days: int = Field(0, ge=0, le=30, description="Days of poor mental health in the last 30 days")
    physical_health_poor_days: int = Field(0, ge=0, le=30, description="Days of poor physical health in the last 30 days")
    difficulty_walking: bool = Field(False, description="Serious difficulty walking or climbing stairs")
    exercise_hours_per_week: Optional[float] = Field(
        None, ge=0, le=40, description="Weekly exercise hours; derived from avg_daily_steps if not provided"
    )
    exercise_induced_angina: bool = Field(False, description="Chest pain/angina brought on by exercise")
    chest_pain_type: Optional[Literal["Asymptomatic", "Typical Angina", "Atypical Angina", "Non-anginal Pain"]] = Field(
        None, description="Chest pain classification; derived from symptoms if not provided"
    )

    @computed_field
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)
    
    @computed_field
    def pulse_pressure(self) -> float:
        return self.systolic_bp - self.diastolic_bp
    
    @model_validator(mode="after")
    def validate_computed_metrics(self):
        if self.bmi <= 0:
            raise ValueError("BMI must be greater than 0")
        if self.pulse_pressure <= 0:
            raise ValueError("Pulse pressure cannot be negative")
        return self
    
class DiseaseRisk(BaseModel):
    """A disease risk with a REAL numeric probability. As of the ML-only-
    probabilities fix, every DiseaseRisk in AgentState is produced by one of
    the four trained XGBoost models in ml_disease_models.py (Hypertension,
    Stroke, Diabetes, Cardiovascular). Nothing else is allowed to populate
    this model — see ClinicalConsideration below for the LLM's differential
    reasoning path, which has no score field to begin with."""
    disease_name: str = Field(..., description="Name of the disease")
    risk_score: float = Field(..., description="Risk score from 0.0 to 1.0, sourced from a trained ML model")
    reasoning: str = Field(
        default="", description="Brief clinical reasoning citing the specific findings that support this risk (Feature 6: Differential Diagnosis)"
    )
    source: Literal["ml_model"] = Field(
        default="ml_model",
        description="Always 'ml_model'. Kept as an explicit tag (rather than relying on callers to know) so any "
                    "future consumer of DiseaseRisk can assert its provenance instead of assuming it."
    )


class ClinicalConsideration(BaseModel):
    """
    A disease/condition raised by the LLM's differential reasoning that has
    NO trained ML model behind it (e.g. Myocardial Infarction, Pulmonary
    Embolism, DVT, Kidney Disease). This model deliberately has NO risk_score
    field — the fix for 'the LLM keeps inventing probabilities for diseases
    with no trained model' has to happen at the schema level, not just in
    how the UI renders things. If there's no field to put a number in, the
    LLM (and its structured-output parser) cannot manufacture one, and no
    downstream renderer can accidentally display one either.
    """
    disease_name: str = Field(..., description="Condition being raised for clinician consideration")
    reasoning: str = Field(..., description="One to two sentence clinical rationale for why this is being raised")
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Specific patient findings (symptoms, history, labs) supporting this consideration, "
                    "e.g. ['Chest pain', 'High LDL', 'Hypertension', 'Diabetes']"
    )


class RiskAssessment(BaseModel):
    disease_risks: List[DiseaseRisk] = Field(
        default_factory=list,
        description="ML-MODEL-ONLY disease risks with real probability scores. Limited to whichever of "
                    "Hypertension/Stroke/Diabetes/Cardiovascular Disease have a trained model available "
                    "(see ml_disease_models.py) — never populated by the LLM."
    )
    clinical_considerations: List[ClinicalConsideration] = Field(
        default_factory=list,
        description="LLM-flagged differential considerations for diseases with NO trained model. These NEVER "
                    "carry a probability score by construction (see ClinicalConsideration)."
    )
    risk_flags: List[str] = Field(default_factory=list, description="Risk flags associated with the patient")
    risk_summary: str = Field(..., description="Summary of the risk assessment")

class RiskResponse(BaseModel):
    """Structured-output schema for the LLM's differential-diagnosis call.
    identified_risks is typed as List[ClinicalConsideration] (NOT DiseaseRisk)
    specifically so the LLM's JSON schema has no risk_score slot to fill in
    for these diseases — see ClinicalConsideration's docstring."""
    identified_risks: List[ClinicalConsideration] = Field(
        ..., description="Differential considerations for conditions with NO trained ML model. Never include "
                          "a probability/percentage for any of these — that field does not exist in this schema."
    )
    risk_flags: List[str] = Field(..., description="High-priority clinical risk flags")
    risk_summary: str = Field(..., description="Professional risk summary")

class Medication(BaseModel):
    name: str = Field(..., description="Drug class or specific agent being suggested for clinician consideration")
    dose: str = Field(..., description="Typical dosage and frequency range, for clinician reference")
    mechanism: str = Field(..., description="Short mechanism of action")
    notes: str = Field(..., description="Monitoring and safety notes")
    reason: str = Field(
        default="", description="Why this option is being suggested for this patient (clinical rationale)"
    )
    supporting_guideline: str = Field(
        default="", description="Guideline body backing this option, e.g. 'ADA Standards of Care 2026'"
    )
    evidence_source: str = Field(
        default="", description="Evidence source badge(s), e.g. 'ADA, PubMed'"
    )
    confidence: str = Field(
        default="", description="Qualitative confidence in this suggestion, e.g. 'High', 'Moderate', 'Low'"
    )

class PrescriptionPlan(BaseModel):
    medications: List[Medication] = Field(default_factory=list, description="Medications to be prescribed")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for the patient")
    instructions: List[str] = Field(default_factory=list, description="Instructions for the patient")

class LifestylePlan(BaseModel):
    exercises: List[str] = Field(default_factory=list, description="Exercise recommendations for the patient")
    diet: List[str] = Field(default_factory=list, description="Diet recommendations for the patient")
    sleep: List[str] = Field(default_factory=list, description="Sleep recommendations for the patient")
    metabolic_advice: List[str] = Field(default_factory=list, description="Metabolic advice for the patient")
    
class RoadmapPriority(BaseModel):
    """Feature: structured Clinical Road Map, shown as priority cards instead
    of long free-text paragraphs."""
    priority_rank: int = Field(..., description="1 = most urgent")
    title: str = Field(..., description="Short priority name, e.g. 'Control Blood Pressure'")
    goal: str = Field(..., description="Concrete, measurable target, e.g. '<140/90 mmHg'")
    rationale: str = Field(default="", description="One-line clinical reason this priority matters now")


class LifestyleTargets(BaseModel):
    """Feature: personalized numeric lifestyle targets, replacing generic
    lifestyle text with patient-specific numbers."""
    current_bmi: float = Field(..., description="Patient's current BMI")
    weight_goal: str = Field(..., description="e.g. 'Lose 5-7% body weight over 6 months'")
    daily_calories_kcal: int = Field(..., description="Target daily calorie intake")
    daily_walking_minutes: int = Field(..., description="Target daily walking/aerobic minutes")
    resistance_sessions_per_week: int = Field(..., description="Target resistance-training sessions per week")
    sodium_limit_mg: int = Field(..., description="Target daily sodium intake ceiling, in mg")


class QualityPanel(BaseModel):
    """Feature: expanded, always-on evaluation panel shown alongside every
    run's output (not just the offline evaluation.py harness)."""
    groundedness: Optional[float] = Field(default=None, description="0-1: is the roadmap grounded in retrieved evidence")
    faithfulness: Optional[float] = Field(default=None, description="0-1: LLM-judge faithfulness score")
    evidence_coverage: List[str] = Field(default_factory=list, description="Evidence sources backing the plan, e.g. ['ADA','PubMed']")
    guideline_coverage: List[str] = Field(default_factory=list, description="Guideline organizations retrieved for this query")
    retrieved_sources_count: int = Field(default=0, description="Number of ranked evidence items feeding the plan")
    hallucination_risk: Literal["LOW", "MODERATE", "HIGH", "UNKNOWN"] = Field(default="UNKNOWN")
    recommendation_confidence: Optional[float] = Field(default=None, description="0-1: overall recommendation confidence")
    drug_safety_status: Literal["PASS", "FLAGGED", "N/A"] = Field(default="N/A")
    clinical_considerations_pending: int = Field(
        default=0,
        description="Count of LLM-flagged conditions with NO trained ML model that still need clinician "
                    "review (e.g. suspected MI/PE/DVT). Surfaced so this never silently disappears from "
                    "the quality panel just because it has no probability score to show."
    )


class MedicalSearchQuery(BaseModel):
    query: str = Field(..., description="Generated medical search query")
    
class DocEvalScore(BaseModel):
    score: float = Field(..., description="Relevancy score for the document")
    reason: str = Field(..., description="Reason for the score")
    
class MedicalEvidence(BaseModel):
    query: str = Field(..., description="Generated medical search query")
    retrieved_chunks_count: int = Field(..., description="Number of chunks retrieved")
    refined_context: str = Field(..., description="Filtered evidence-based sentences")
    clinical_summary: str = Field(..., description="LLM-synthesized clinical summary")
    sources_used: Optional[List[str]] = Field(default_factory=list, description="Sources used (PubMed, Tavily, etc.)")
    
class ClinicalAlert(BaseModel):
    urgency: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] = Field(..., description="Urgency level of the alert")
    message: str = Field(..., description="Clinician-facing alert message")
    conditions_flagged: Optional[List[str]] = Field(default_factory=list, description="List of diseases that triggered escalation")
    interaction_flags: Optional[List[str]] = Field(default_factory=list, description="Detected multi-modibidity conditions and risk interaction patterns")
    recommended_action: str = Field(..., description="Recommend next clinical action based on urgency level.")
    
class AgentState(BaseModel):
    patient_profile: PatientProfile = Field(..., description="Patient profile")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    prescription_plan: PrescriptionPlan = Field(..., description="Prescription plan")
    lifestyle_plan: LifestylePlan = Field(..., description="Lifestyle plan")
    raw_patient_data: Dict[str, Any] = Field(default_factory=dict, description="Raw patient data")
    medical_search_query: MedicalSearchQuery = Field(..., description="Medical search query based on patient profile")
    medical_evidence: MedicalEvidence = Field(..., description="Medical evidence retrieved from external knowledge sources")
    clinical_alert: ClinicalAlert = Field(default=None, description="Clinical escalation alerts triggered by high severity risk conditions")
    treatment_road_map: str = Field(default="", description="Consolidated clinical treatment strategy and road map")

    # --- New feature state (Features 1, 2, 5, 4) ---
    ml_disease_predictions: List[MLDiseasePrediction] = Field(
        default_factory=list,
        description="Real trained-model predictions (hypertension/stroke/diabetes/heart), computed once in "
                     "early_disease_detection and reused by explain_disease_risk so SHAP isn't recomputed."
    )
    explainability_report: Optional[ExplainabilityReport] = Field(
        default=None, description="Explainable AI output: per-disease feature contributions (Feature 1)"
    )
    guideline_evidence: Optional[GuidelineEvidence] = Field(
        default=None, description="Merged ADA/AHA/WHO/NICE/CDC guideline evidence combined with PubMed (Feature 2)"
    )
    evidence_ranking: Optional[EvidenceRankingSummary] = Field(
        default=None, description="Ranked evidence sources + overall recommendation confidence (Feature 5)"
    )
    followup_plan: Optional[FollowUpPlan] = Field(
        default=None, description="Follow-up and monitoring plan (Feature 4)"
    )

    # --- Additional display/usability feature state ---
    knowledge_graph: Optional[ClinicalKnowledgeGraph] = Field(
        default=None, description="Patient-specific clinical relationship graph (Feature 7)"
    )
    roadmap_priorities: List[RoadmapPriority] = Field(
        default_factory=list, description="Structured Clinical Road Map priorities, ranked (replaces long free-text roadmap)"
    )
    lifestyle_targets: Optional[LifestyleTargets] = Field(
        default=None, description="Personalized numeric lifestyle targets"
    )
    quality_panel: Optional[QualityPanel] = Field(
        default=None, description="Expanded evaluation panel shown alongside the run's output"
    )
    hitl_halted: bool = Field(
        default=False,
        description="True only if the clinician explicitly rejected continuation at human_review_gate. "
                    "Used by hitl_router to decide routing — see human_review_gate for where this is set."
    )
    
# Define the node functions
@traceable(name="collect_patient_data", run_type="chain")
def collect_patient_data(state: AgentState) -> dict:
    """   
        Collects patient data from raw structured inputs and converts it into a PatientProfile object.
    """
    raw_patient_data = state.raw_patient_data
    patient = state.patient_profile.model_copy(deep=True)
    
    # Symptoms & History (EHR)
    patient.age = raw_patient_data.get("age", None)
    patient.gender = raw_patient_data.get("gender", None)
    patient.height = raw_patient_data.get("height", None)
    patient.weight = raw_patient_data.get("weight", None)
    patient.systolic_bp = raw_patient_data.get("systolic_bp", None)
    patient.diastolic_bp = raw_patient_data.get("diastolic_bp", None)
    patient.cholesterol = raw_patient_data.get("cholesterol", None)
    patient.triglycerides = raw_patient_data.get("triglycerides", None)
    patient.heartbeat_rate = raw_patient_data.get("heartbeat_rate", None)
    patient.temperature = raw_patient_data.get("temperature", 37.0)
    patient.respiratory_rate = raw_patient_data.get("respiratory_rate", 16)
    patient.wbc_count = raw_patient_data.get("wbc_count", 7.0)
    patient.platelets = raw_patient_data.get("platelets", 250.0)
    patient.oxygen_saturation = raw_patient_data.get("oxygen_saturation", 98.0)
    patient.sugar_level = raw_patient_data.get("sugar_level", None)
    patient.symptoms = raw_patient_data.get("symptoms", [])
    patient.medical_history = raw_patient_data.get("medical_history", [])
    patient.immunizations = raw_patient_data.get("immunizations", [])
    patient.allergies = raw_patient_data.get("allergies", [])
    patient.avg_sleep_hours = raw_patient_data.get("avg_sleep_hours", None)
    patient.avg_daily_steps = raw_patient_data.get("avg_daily_steps", None)

    # --- New fields for the trained ML risk models (ml_disease_models.py).
    # Falls back to the PatientProfile defaults above if the caller (e.g.
    # app.py's form, or an older raw_patient_data payload) doesn't supply them. ---
    patient.family_history = raw_patient_data.get("family_history", patient.family_history)
    patient.salt_intake_g_per_day = raw_patient_data.get("salt_intake_g_per_day", patient.salt_intake_g_per_day)
    patient.stress_score = raw_patient_data.get("stress_score", patient.stress_score)
    patient.smoking_status = raw_patient_data.get("smoking_status", patient.smoking_status)
    patient.ever_married = raw_patient_data.get("ever_married", patient.ever_married)
    patient.occupation_type = raw_patient_data.get("occupation_type", patient.occupation_type)
    patient.residence_type = raw_patient_data.get("residence_type", patient.residence_type)
    patient.diet_fruits_daily = raw_patient_data.get("diet_fruits_daily", patient.diet_fruits_daily)
    patient.diet_veggies_daily = raw_patient_data.get("diet_veggies_daily", patient.diet_veggies_daily)
    patient.alcohol_intake = raw_patient_data.get("alcohol_intake", patient.alcohol_intake)
    patient.general_health_rating = raw_patient_data.get("general_health_rating", patient.general_health_rating)
    patient.mental_health_poor_days = raw_patient_data.get("mental_health_poor_days", patient.mental_health_poor_days)
    patient.physical_health_poor_days = raw_patient_data.get("physical_health_poor_days", patient.physical_health_poor_days)
    patient.difficulty_walking = raw_patient_data.get("difficulty_walking", patient.difficulty_walking)
    patient.exercise_hours_per_week = raw_patient_data.get("exercise_hours_per_week", patient.exercise_hours_per_week)
    patient.exercise_induced_angina = raw_patient_data.get("exercise_induced_angina", patient.exercise_induced_angina)
    patient.chest_pain_type = raw_patient_data.get("chest_pain_type", patient.chest_pain_type)
    return {"patient_profile": patient}

@traceable(name="early_disease_detection", run_type="chain")
def _patient_has_any_notable_finding(patient) -> bool:
    """
    Deterministic (non-LLM) check: does this patient have ANY symptom,
    documented history, or lab/vital outside its normal reference range?
    Reference ranges mirror the ones given to the LLM in the
    early_disease_detection prompt, so the two stay consistent.

    Used as a hard backstop below: if this returns False, clinical_considerations
    is forced empty regardless of what the LLM produced. A prompt instruction
    alone can be ignored; this cannot, since a fully normal patient cannot
    have "sufficient supporting evidence" for anything by definition.
    """
    if patient.symptoms or patient.medical_history:
        return True
    checks = [
        patient.ldl_cholesterol >= 130,
        patient.hdl_cholesterol < (40 if patient.gender.strip().lower() == "male" else 50),
        patient.triglycerides >= 150,
        patient.cholesterol >= 200,
        not (36.1 <= patient.temperature <= 37.2),
        not (4.0 <= patient.wbc_count <= 11.0),
        not (150.0 <= patient.platelets <= 450.0),
        patient.oxygen_saturation < 95.0,
        not (12 <= patient.respiratory_rate <= 20),
        not (60 <= patient.heartbeat_rate <= 100),
        patient.systolic_bp >= 130 or patient.diastolic_bp >= 80,
        patient.sugar_level > 7.8,  # >140 mg/dL equivalent, roughly pre-diabetic threshold
    ]
    return any(checks)


def early_disease_detection(state: AgentState) -> dict:
    """
    Disease risk detection — now a hybrid of real trained models and LLM
    differential diagnosis, instead of the LLM guessing every score.

    - Hypertension, Stroke, Diabetes, and Cardiovascular risk now come
      directly from trained XGBoost classifiers (ml_disease_models.py) —
      deterministic, reproducible probabilities, not LLM estimates.
    - The LLM is still used, but ONLY to surface disease categories those
      four models don't cover (oncology, infectious, autoimmune, endocrine,
      neurological, etc.). It's given the four ML scores as grounding
      context and explicitly told not to re-score/duplicate them, so the
      final list never has two conflicting numbers for the same disease.
    """
    patient = state.patient_profile

    # --- Step 1: real model predictions for the four covered diseases ---
    ml_predictions = predict_ml_disease_risks(patient)
    ml_disease_risks = [
        DiseaseRisk(disease_name=p.disease_name, risk_score=p.risk_score, reasoning=p.reasoning)
        for p in ml_predictions
    ]
    ml_covered_names = {p.disease_name for p in ml_predictions}
    ml_scores_text = (
        "\n".join(f"- {p.disease_name}: {p.risk_score*100:.0f}% (from a trained model — do not re-score this)"
                   for p in ml_predictions)
        if ml_predictions else "No trained-model scores are available for this run."
    )

    # Keyword groups for each ML-covered disease, used below to catch
    # differently-worded LLM output for the SAME condition (e.g. "Atherosclerotic
    # Cardiovascular Disease" or "Coronary Artery Disease" for the ML-scored
    # "Cardiovascular Disease"). An exact-string check alone lets these near-
    # duplicates through as if they were distinct, un-scored conditions,
    # which is confusing next to the real ML score for the same disease.
    _ML_DISEASE_KEYWORDS = {
        "Cardiovascular Disease": ["cardio", "coronary", "atheroscl", "ischemic heart", "chd"],
        "Hypertension": ["hypertens"],
        "Stroke": ["stroke", "cerebrovascular"],
        "Diabetes": ["diabet"],
    }

    def _duplicates_ml_disease(candidate_name: str) -> bool:
        if candidate_name in ml_covered_names:
            return True
        name_lower = candidate_name.lower()
        for ml_name in ml_covered_names:
            for kw in _ML_DISEASE_KEYWORDS.get(ml_name, []):
                if kw in name_lower:
                    return True
        return False

    # --- Step 2: LLM covers everything else (oncology, infectious, autoimmune,
    # endocrine, neurological, etc.), grounded by the ML scores above. These
    # are captured as ClinicalConsideration items (NO risk_score field exists
    # on that schema — see its docstring), so the LLM has no slot to put a
    # fabricated percentage into, and no downstream code can render one for
    # them by mistake. ---
    detection_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are an advanced clinical diagnostic support system.
            Trained machine learning models have already scored this patient's risk for
            Hypertension, Stroke, Diabetes, and Cardiovascular Disease — those four scores
            are given to you below as established fact. Your job is to identify OTHER
            potential disease risks not already covered by those four models.

            Coverage includes but is not limited to:
            - Oncology (Tumors, Cancers - based on symptoms and markers like platelets/WBC)
            - Infectious (COVID-19, HIV/AIDS, Viral, Bacterial - based on temperature, respiratory rate, oxygen, WBC)
            - Autoimmune, Endocrine, Renal, and Neurological conditions.
            - Metabolic syndrome (only if not simply restating the Diabetes/Cardiovascular scores above).
            - Acute considerations directly suggested by symptoms (e.g. Myocardial Infarction,
              Pulmonary Embolism, Deep Vein Thrombosis) when clinically supported by the findings below.

            NORMAL REFERENCE RANGES — do not raise ANY consideration where the only "evidence" is a
            value inside its normal range. A value in range is NOT a contributing factor:
            - LDL Cholesterol: <100 mg/dL normal, 100-129 near-optimal, 130-159 borderline high, 160+ high.
              Do NOT flag "Hyperlipidemia" or similar for LDL <130 — that is at most borderline, not high.
            - HDL Cholesterol: >=40 mg/dL (men) / >=50 mg/dL (women) is normal; lower is a risk factor.
            - Triglycerides: <150 mg/dL normal, 150-199 borderline high, 200+ high.
            - Total Cholesterol: <200 mg/dL normal, 200-239 borderline high, 240+ high.
            - Body Temperature: 36.1-37.2°C (97-99°F) is normal. Do NOT infer thyroid disease, infection,
              or any illness from a temperature in this range — 36.5°C is NORMAL, not "low" or suggestive
              of hypothyroidism.
            - WBC Count: 4-11 x10^9/L is normal. A value inside this range is NOT "Low WBC" or "High WBC" —
              do not flag leukopenia/leukocytosis/oncology/infection considerations from a normal WBC alone.
            - Platelets: 150-450 x10^9/L is normal.
            - Oxygen Saturation (SpO2): >=95% is normal; 91-94% is mild hypoxia; <91% is significant.
            - Respiratory Rate: 12-20 breaths/min is normal.
            - Heart Rate: 60-100 bpm is normal at rest.
            - Blood Pressure: <120/80 normal, 120-129/<80 elevated, 130+/80+ is hypertensive range
              (already captured by the ML Hypertension score above — do not re-flag this on its own).
            If a value is inside its normal range, it must never appear in "contributing_factors" or be
            cited as supporting evidence for any consideration — normal findings support ruling OUT a
            condition, not raising one.

            EVIDENTIARY BAR FOR HIGH-STAKES ACUTE CONSIDERATIONS (Myocardial Infarction, Pulmonary
            Embolism, Deep Vein Thrombosis, Stroke-mimics, etc.): these carry major clinical weight if
            raised, so require a MEANINGFUL COMBINATION of supporting findings, never a single soft or
            nonspecific symptom alone:
            - Pulmonary Embolism: requires at least TWO of: shortness of breath, tachycardia (HR >100),
              hypoxia (SpO2 <95%), pleuritic chest pain, recent surgery/immobility, history of DVT/clotting
              disorder. Do NOT raise PE from fatigue alone, or from a single vague symptom.
            - Myocardial Infarction: requires chest pain WITH at least one of: radiating pain, shortness of
              breath, diaphoresis, strong cardiovascular risk profile (the ML Cardiovascular Disease score
              itself is supporting evidence here), or ECG-type symptoms reported by the patient.
            - Deep Vein Thrombosis: requires unilateral limb swelling/pain WITH at least one of: recent
              immobility/surgery, known clotting disorder, active cancer.
            If the findings don't clear this bar, do NOT raise the consideration — silence is correct when
            evidence is insufficient; a clinician reviewing a shorter, well-supported list is better served
            than one reviewing a long list padded with weakly-supported entries.

            CRITICAL RULE ON PROBABILITIES:
            - The output schema for each item is: disease_name, reasoning, contributing_factors.
              There is NO field for a probability, percentage, or risk score, and you must never invent
              one inside "reasoning" either (e.g. do NOT write "risk 85%" or "high probability (70%)").
            - You are NOT a trained statistical model for these conditions. You are providing qualitative
              clinical reasoning only — a differential consideration for a clinician to evaluate further,
              never a quantified risk estimate.
            - "reasoning" must be 1-2 sentences stating WHY this is being raised, in plain clinical language,
              with no numbers implying a probability (lab values/vitals are fine to cite, e.g. "SpO2 91%",
              since that is a measured finding, not a fabricated risk score).
            - "contributing_factors" is a short list of the specific findings that support this consideration
              (e.g. ["Chest pain", "High LDL", "Hypertension", "Diabetes"]).

            Other rules:
            - Do NOT output Hypertension, Stroke, Diabetes, or Cardiovascular Disease —
              those already have real model-derived scores and must not be duplicated or
              re-estimated by you. This also means: do not output a synonym, subtype, or
              closely related restatement of one of those four either (e.g. "Atherosclerotic
              Cardiovascular Disease", "Coronary Artery Disease", or "Ischemic Heart Disease"
              when Cardiovascular Disease is already scored — that is the same condition, not
              a new one). Only raise something distinct enough that a clinician would treat it
              as a separate line item requiring its own workup.
            - Identify high-priority "risk_flags" for rapid clinical alerting, taking the
              provided ML scores into account alongside anything else you find.
            - Provide a concise, professional risk summary covering the WHOLE patient
              picture, including a mention of the ML-derived scores (percentages are fine
              ONLY when quoting the ML scores given to you above, never for your own considerations).
            - Do NOT diagnose. Frame every item as a "consideration" for further clinical evaluation,
              never a confirmed diagnosis.
            - Output strictly in JSON matching the provided structure. If you find no
              additional considerations, return an empty list for identified_risks.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Already-scored by trained models (do not repeat these, and do not assign your own score to them):
            {ml_scores}

            Patient Profile:
            Age: {age}, Gender: {gender}, BMI: {bmi:.2f}
            BP: {sbp}/{dbp}, Pulse: {hr}, Temp: {temp}°C, Resp: {resp}, SpO2: {spo2}%
            WBC: {wbc}, Platelets: {plt}, Sugar: {sugar}, Lipids: LDL {ldl}/HDL {hdl}
            Symptoms: {symptoms}
            History: {history}
            
            Identify any additional relevant clinical considerations (no probabilities) and provide a
            structured assessment.
            """
        )
    ])
    
    detection_chain = detection_prompt | llm.with_structured_output(RiskResponse)
    
    results = detection_chain.invoke({
        "ml_scores": ml_scores_text,
        "age": patient.age,
        "gender": patient.gender,
        "bmi": patient.bmi,
        "sbp": patient.systolic_bp,
        "dbp": patient.diastolic_bp,
        "hr": patient.heartbeat_rate,
        "temp": patient.temperature,
        "resp": patient.respiratory_rate,
        "spo2": patient.oxygen_saturation,
        "wbc": patient.wbc_count,
        "plt": patient.platelets,
        "sugar": patient.sugar_level,
        "ldl": patient.ldl_cholesterol,
        "hdl": patient.hdl_cholesterol,
        "symptoms": ", ".join(patient.symptoms),
        "history": ", ".join(patient.medical_history)
    })

    # Belt-and-suspenders: drop any LLM item that duplicates an ML-covered
    # disease name (exact OR keyword-synonym match — see _duplicates_ml_disease)
    # even if the prompt instruction was ignored.
    clinical_considerations = [
        r for r in results.identified_risks if not _duplicates_ml_disease(r.disease_name)
    ]

    # Deterministic backstop (see _patient_has_any_notable_finding docstring):
    # a genuinely normal patient (no symptoms, no history, every lab/vital in
    # range) cannot have real supporting evidence for ANY consideration. This
    # catches over-calling even if the LLM ignored the reference-range/
    # evidentiary-bar instructions above — a prompt instruction can be
    # ignored, this cannot.
    if clinical_considerations and not _patient_has_any_notable_finding(patient):
        clinical_considerations = []

    risk_assessment = RiskAssessment(
        disease_risks=ml_disease_risks,
        clinical_considerations=clinical_considerations,
        risk_flags=results.risk_flags,
        risk_summary=results.risk_summary
    )

    return {"risk_assessment": risk_assessment, "ml_disease_predictions": ml_predictions}

@traceable(name="explain_disease_risk", run_type="chain")
def explain_disease_risk(state: AgentState) -> dict:
    """
    Feature 1 — Explainable AI.
    Runs immediately after early_disease_detection and answers
    "Why did the AI predict this disease risk?" using the REAL SHAP
    TreeExplainer values already computed for each trained model in
    early_disease_detection (see explainability.py — the surrogate model
    approach is retired now that real models + real explainers exist).
    The result is stored in AgentState.explainability_report and rendered
    inside Streamlit (and included in the PDF report).
    """
    patient = state.patient_profile
    disease_risks = state.risk_assessment.disease_risks
    clinical_considerations = state.risk_assessment.clinical_considerations
    ml_predictions = state.ml_disease_predictions

    # ML-ONLY-PROBABILITIES FIX: explainability (SHAP) must only ever be
    # generated for the four trained-model diseases. clinical_considerations
    # (LLM differential items with no score) are intentionally NOT passed to
    # generate_explanations anymore — there is nothing to explain with SHAP
    # for a condition with no trained model, and explainability.py no longer
    # accepts them (see its updated signature).
    report = generate_explanations(ml_predictions)

    # Feature 7: Dynamic Clinical Knowledge Graph — rule-based, deterministic,
    # built from the same disease-risk output (no extra LLM call, no
    # hallucination risk). clinical_considerations is passed alongside so a
    # consideration like "Kidney Disease" or "Coronary Artery Disease" can
    # still surface the right graph node even though it has no ML score.
    # See knowledge_graph.py.
    kg = build_knowledge_graph(
        patient, disease_risks, state.risk_assessment.risk_flags,
        clinical_considerations=clinical_considerations,
    )

    return {"explainability_report": report, "knowledge_graph": kg}


@traceable(name="clinical_triage_router", run_type="tool")
def clinical_triage_router(state: AgentState) -> Literal["high_risk", "low_risk"]:
    """
        Routes the workflow based on detected risk levels.

        A non-empty clinical_considerations list (LLM-flagged conditions with
        no trained ML model, e.g. suspected Myocardial Infarction/Pulmonary
        Embolism/DVT) also forces the high-risk path. Those items carry no
        probability score by design (see ClinicalConsideration), so they
        can never be compared against the 0.60 ML threshold below — without
        this check, a clinically urgent but ML-unscored condition would be
        silently routed to the "general wellness" path instead of clinician
        review, which is a patient-safety regression, not a simplification.
    """
    risks = state.risk_assessment.disease_risks
    clinical_considerations = state.risk_assessment.clinical_considerations

    if clinical_considerations:
        return "high_risk"

    if not risks:
        return "low_risk"
    
    # If any risk is >= 60% (0.6), consider it high/clinical risk
    max_risk = max([r.risk_score for r in risks])
    if max_risk >= 0.60:
        return "high_risk"
    
    return "low_risk"

@traceable(name="generate_medical_search_query", run_type="chain")
def generate_medical_search_query(state: AgentState) -> dict:
    """
        Generates a PubMed search query based on the patient's risk assessment."""
    # Only include ML-scored diseases with significant risk
    high_risk_diseases = [risk.disease_name for risk in state.risk_assessment.disease_risks if risk.risk_score >= 0.25]
    # clinical_considerations have no score to threshold on, so they're
    # always included — retrieving real evidence for a suspected condition
    # (e.g. Myocardial Infarction) helps ground the LLM's later reasoning
    # about it in actual literature/guidelines instead of an unsupported guess.
    consideration_diseases = [c.disease_name for c in state.risk_assessment.clinical_considerations]
    high_risk_diseases = high_risk_diseases + consideration_diseases

    medical_query_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a clinical research assistant. "
            "Given a patient's clinical risk assessment, generate a **short list of disease/condition keywords** "
            "suitable for PubMed and clinical literature searches.\n"
            "Rules:\n"
            "- Only include the most relevant identified conditions (Chronic, Infectious, Oncology, etc.)\n"
            "- Return keywords separated by commas\n"
            "- Do NOT write full sentences\n"
            "- Output JSON matching: {{'query': 'condition1, condition2'}}"
        ),
        HumanMessagePromptTemplate.from_template(
            "High-risk diseases:\n{risks}\nGenerate PubMed search keywords."
        )
    ])

    medical_query_chain = medical_query_prompt | llm.with_structured_output(MedicalSearchQuery)

    output = medical_query_chain.invoke({
        "risks": ", ".join(high_risk_diseases) if high_risk_diseases else "general health"
    })

    return {"medical_search_query": output.query}

@traceable(name="pubmed_search", run_type="tool")
def pubmed_search(query: str, retmax: int = 5):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
        "email": "your_email@example.com"  # recommended
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    return res.json()["esearchresult"]["idlist"]

@traceable(name="pubmed_fetch", run_type="tool")
def pubmed_fetch(pmids: list[str]):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    return res.text

@traceable(name="parse_pubmed_xml", run_type="tool")
def parse_pubmed_xml(xml_text: str) -> list[Document]:
    root = ET.fromstring(xml_text)
    docs = []

    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="")
        abstract = " ".join(
            [t.text or "" for t in article.findall(".//AbstractText")]
        )
        mesh_terms = [
            m.text for m in article.findall(".//MeshHeading/DescriptorName")
        ]

        content = f"{title}\n\n{abstract}"

        docs.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "mesh_terms": mesh_terms,
                    "source": "PubMed"
                }
            )
        )

    return docs

def _format_clinical_considerations(considerations: List["ClinicalConsideration"]) -> str:
    """
    Shared formatter used by every downstream node (prescribe_medications,
    clinical_lifestyle_advice, clinical_strategy_synthesis, alert_clinician,
    generate_followup_plan) that needs to give the LLM context about
    LLM-flagged, non-ML-scored conditions. Centralized in one place (per the
    project's "no duplicate logic" convention) and always renders WITHOUT a
    percentage, since ClinicalConsideration has no score field to begin with.
    """
    if not considerations:
        return "None"
    lines = []
    for c in considerations:
        factors = ", ".join(c.contributing_factors) if c.contributing_factors else "see reasoning"
        lines.append(f"- {c.disease_name} (no ML probability available): {c.reasoning} [Factors: {factors}]")
    return "\n".join(lines)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 25
    ]
    
@traceable(name="fetch_medical_literature", run_type="chain")
def fetch_medical_literature(state: AgentState) -> dict:
    """ 
        Fetches relevant medical literature based on the generated search query.
        Combines internal medical knowledge + web-based guideline retrieval
        with strict relevance filtering.
    """
    query = generate_medical_search_query(state)
    search_query = query["medical_search_query"] if isinstance(query, dict) else query
    # Primary medical literature source
    try:
        pmids = pubmed_search(search_query)
        
        if pmids:
            pubmed_xml = pubmed_fetch(pmids)
            docs = parse_pubmed_xml(pubmed_xml)
        else:
            docs = []
            
        print(f"PubMed search returned {len(pmids)} results")
    except Exception as e:
        print(f"PubMed search/fetch failed: {e}")
        docs = []
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        return {
            "medical_evidence": MedicalEvidence(
                query=search_query,
                retrieved_chunks_count=0,
                refined_context="No specific literature findings for this query.",
                clinical_summary="No direct medical evidence was found in the queried sources for these combinations of risks.",
                sources_used=["PubMed"]
            )
        }
    
    # Clear irrelevant encoding characters like emojis and special characters
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        
    # Create embeddings for each chunk
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    
    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(search_query)
    
    # Evaluate relevance of retrieved chunks
    doc_eval_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a strict retrieval evaluator for RAG based on medical knowledge."
            "Score each retrieved document in [0.0, 1.0] for relevance to the query.\n"
            "1.0 = most relevant, 0.0 = least relevant.\n"
            "Strictly return JSON: {'score': float, 'reason': str}"
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: {query}\nChunk: {chunk}"
        )
    ])
    
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)
    
    # Corrective RAG thresholds
    LOWER_THRESHOLD = 0.3
    UPPER_THRESHOLD = 0.7

    evaluated_docs = []
    scores = []
    
    for doc in retrieved_docs:
        try:
            result = doc_eval_chain.invoke({"query": search_query, "chunk": doc.page_content})
            scores.append(result.score)
            
            if result.score > LOWER_THRESHOLD:
                evaluated_docs.append(doc)
        except:
            continue
    
    good_docs = evaluated_docs.copy()
    
    # Perform Corrective-RAG routing
    verdict = "AMBIGUOUS"
    
    if any(score > UPPER_THRESHOLD for score in scores):
        verdict = "CORRECT"
        
    if all(score < LOWER_THRESHOLD for score in scores):
        verdict = "INCORRECT"
    
    # Fallback: Web-based retrieval using TavilySearch if verdict is either INCORRECT or AMBIGUOUS
    if verdict in ["INCORRECT", "AMBIGUOUS"] and tavily_search_tool:
        try:
            web_results = tavily_search_tool.run(search_query)
            
            if isinstance(web_results, dict) and "results" in web_results:
                web_results = web_results["results"]
            elif isinstance(web_results, list):
                web_results = web_results
            else:
                web_results = []
        except Exception as e:
            print(f"Tavily search failed: {e}")
            web_results = []
    else:
        web_results = []

    retrieved_docs = []
    
    for res in web_results:
        title = res.get("title", "")
        url = res.get("url", "")
        content = res.get("content", "") or res.get("snippet", "")
        full_text = f"{title}\nURL: {url}\n\n{content}"
        
        retrieved_docs.append(
            Document(
                page_content=full_text.strip(),
                metadata={
                    "source": "Web/TavilySearch",
                    "url": url,
                    "title": title
                }
            )
        )
            
    if verdict == "CORRECT":
        final_relevant_docs = good_docs
    elif verdict == "INCORRECT":
        final_relevant_docs = retrieved_docs
    else: # AMBIGUOUS
        final_relevant_docs = good_docs + retrieved_docs
            
    # Sentence-level decomposition
    all_sentences = []
    
    for doc in final_relevant_docs:
        sentences = split_sentences(doc.page_content)
        all_sentences.extend(sentences)
        
    filtered_context = "\n".join(all_sentences)
    
    # Clinical synthesis
    synthesis_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a medical evidence synthesizer.\n"
            "Summarize the evidence in a clinical context based on the patient profile and medical literature.\n"
            "Do NOT hallucinate. Do NOT diagnose or prescribe.\n"
        ),
        HumanMessagePromptTemplate.from_template(
            "Context: {context}"
        )
    ])
    
    clinical_summary = llm.invoke(synthesis_prompt.invoke({"context": filtered_context})).content
    
    return {
        "medical_evidence": MedicalEvidence(
            query=search_query,
            retrieved_chunks_count=len(retrieved_docs),
            refined_context=filtered_context,
            clinical_summary=clinical_summary,
            sources_used=list({doc.metadata.get("source", "Unknown") for doc in retrieved_docs})
        )
    }
    
@traceable(name="fetch_clinical_guidelines", run_type="chain")
def fetch_clinical_guidelines(state: AgentState) -> dict:
    """
    Feature 2 — Clinical Guideline Retrieval.
    Extends PubMed retrieval (does NOT replace it) with evidence from ADA,
    AHA, WHO, NICE, and CDC. Retrieves guidelines, ranks them, and merges
    them with the existing PubMed evidence summary so every downstream
    recommendation can show a "Supported By" badge list.

    Also performs Feature 5 (Evidence Ranking + Confidence Score) for all
    evidence feeding the final treatment plan, since ranking naturally
    happens once both evidence pools are available here.

    NOTE: guideline retrieval now runs against a local FAISS vector store
    built from downloaded guideline PDFs (documents/ADA, documents/WHO,
    documents/CDC, documents/Merck_Manual, etc. — see vector_store.py) instead
    of Tavily web search, so `fetch_guidelines` is called with the shared
    `embeddings` model rather than `tavily_search_tool`. Everything else about
    this node (PubMed merge, ranking, return shape) is unchanged.
    """
    medical_evidence = state.medical_evidence
    query = medical_evidence.query or "general health"

    guidelines = fetch_guidelines(embeddings, query)
    guideline_evidence = merge_and_synthesize(
        llm=llm, query=query, guidelines=guidelines, pubmed_summary=medical_evidence.clinical_summary
    )

    # Feature 5: rank every evidence source (guidelines + PubMed) feeding the plan
    ranked_items = rank_guidelines(guidelines) + rank_pubmed_sources(medical_evidence.sources_used or [])
    ranking_summary = build_ranking_summary(ranked_items)

    return {
        "guideline_evidence": guideline_evidence,
        "evidence_ranking": ranking_summary,
    }


@traceable(name="check_openfda_warnings", run_type="tool")
def check_openfda_warnings(drug_name: str) -> List[str]:
    """Query openFDA Label API for boxed warnings and warnings."""
    try:
        url = (
            f"https://api.fda.gov/drug/label.json"
            f"?api_key={OPENFDA_API_KEY}"
            f"&search=openfda.generic_name:{drug_name}"
            f"&limit=1"
        )
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        warnings = []
        if "results" in data:
            result = data["results"][0]
            if "boxed_warning" in result:
                warnings.extend(result["boxed_warning"])
            if "warnings" in result:
                warnings.extend(result["warnings"])
        return warnings
    except requests.exceptions.HTTPError as e:
        return [f"openFDA HTTP error: {str(e)}"]
    except Exception as e:
        return [f"openFDA API failure: {str(e)}"]

@traceable(name="get_rxcui", run_type="tool")
def get_rxcui(drug_name: str):
    """Get RxCUI identifier for a drug name."""
    try:
        url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
        resp = requests.get(url, params={"name": drug_name}, timeout=5)
        resp.raise_for_status()
        rxcuis = resp.json().get("idGroup", {}).get("rxnormId", [])
        return rxcuis[0] if rxcuis else None
    except Exception:
        return None

@traceable(name="get_drug_classes_by_rxcui", run_type="tool")
def get_drug_classes_by_rxcui(rxcui: str):
    """Get RxClass drug classes for a given RxCUI."""
    try:
        url = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
        resp = requests.get(url, params={"rxcui": rxcui}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        associations = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        classes = []
        for assoc in associations:
            class_info = assoc.get("rxclassMinConceptItem", {})
            class_name = class_info.get("className")
            class_type = class_info.get("classType")
            if class_name:
                classes.append(class_name.lower())
        return classes
    except Exception:
        return []

@traceable(name="prescribe_medications", run_type="chain")
def prescribe_medications(state: AgentState) -> dict:
    """   
        Generates evidence-based prescriptions, instructions, and recommendations
        based on the medical evidence retrieved from external knowledge sources, 
        patient profile, and risk assessment.
    """
    patient = state.patient_profile
    risk_assessment = state.risk_assessment
    medical_evidence = state.medical_evidence
    guideline_evidence = state.guideline_evidence
    
    formatted_disease_risks = "\n".join(
        f"{risk.disease_name}: {round(risk.risk_score*100)}%"
        for risk in risk_assessment.disease_risks
    )
    considerations_text = _format_clinical_considerations(risk_assessment.clinical_considerations)

    combined_evidence_summary = medical_evidence.clinical_summary
    supported_by_text = "PubMed"
    if guideline_evidence is not None:
        combined_evidence_summary = guideline_evidence.merged_summary or combined_evidence_summary
        if guideline_evidence.supported_by:
            supported_by_text = ", ".join(guideline_evidence.supported_by)
    
    # Build structured clinical prescription prompt
    prescription_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
                You are an evidence-based clinical decision SUPPORT system. You do NOT prescribe.
                You surface "Suggested Evidence-Based Treatment Options" for a licensed clinician
                to review, adjust, and formally prescribe.

                Rules:
                - Base every option strictly on patient data and retrieved medical evidence.
                - Frame each option as a suggestion using language like "Consider initiating..." or
                  "Consider ... therapy if clinically appropriate", never a command like "Start X".
                - Consider contraindications, age, allergies, BMI, comorbidities.
                - Include first-line guideline therapy CLASSES (e.g. "ACE inhibitor or ARB") as the
                  primary framing; a specific example agent may be named for clinician reference.
                - Include a typical dosage/frequency RANGE, mechanism (short), and monitoring notes.
                - For every option, also populate:
                  * reason: why this option fits this specific patient
                  * supporting_guideline: the guideline body/document backing it (e.g. "ADA Standards of Care 2026")
                  * evidence_source: which retrieved evidence supports it (e.g. "ADA, PubMed")
                  * confidence: "High", "Moderate", or "Low", based on how directly the evidence supports it
                - Include safety warnings and drug interaction considerations in notes.
                - If lifestyle therapy is preferred over medication, state so clearly.
                - Do NOT hallucinate unsupported treatments.
                - Do NOT diagnose.
                - The final treatment decision always remains with the treating clinician.
                - Output strictly in structured JSON matching PrescriptionPlan schema.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
                Patient Profile:
                Age: {age}
                Gender: {gender}
                BMI: {bmi:.2f}
                Blood Pressure: {sbp}/{dbp}
                LDL: {ldl}
                HDL: {hdl}
                Triglycerides: {tg}
                Sugar Level: {sugar}
                Sleep: {sleep}
                Steps: {steps}
                Symptoms: {symptoms}
                Medical History: {history}
                Allergies: {allergies}

                Disease Risks (ML model-derived probabilities):
                {risks}

                Clinical Considerations (LLM-flagged, NO trained model exists — reference these
                qualitatively only, never invent or restate a probability for them):
                {considerations}

                Retrieved Clinical Evidence Summary:
                {evidence}

                Generate a world-class, medically rigorous set of "Suggested Evidence-Based Treatment
                Options" based on the retrieved evidence, for clinician review only.
                - Medications (treatment options): Provide drug class/agent, typical dosage range, frequency, AND clinical rationales (mechanism of action, specific evidence support), plus reason/supporting_guideline/evidence_source/confidence for each.
                - Instructions: Provide detailed administration guidelines, potential side effects to monitor, and contraindications.
                - Recommendations: Provide advanced adjunctive therapies, specific monitoring intervals for metabolic markers, and evidence-based lifestyle synergies. If a Clinical Consideration (e.g. suspected PE/DVT) implies a specific workup, you may mention it qualitatively.
                
                Recommendations must be state-of-the-art, medically rigorous, and highly personalized. Never use generic or redundant dummy values.
                Never return NULL or empty lists for medications, instructions, or recommendations. 
                Output strictly in JSON matching PrescriptionPlan schema.
            """
        )
    ])
    
    prescription_chain = prescription_prompt | llm.with_structured_output(PrescriptionPlan)
    
    prescriptions = prescription_chain.invoke({
        "age": patient.age,
        "gender": patient.gender,
        "bmi": patient.bmi,
        "sbp": patient.systolic_bp,
        "dbp": patient.diastolic_bp,
        "ldl": patient.ldl_cholesterol,
        "hdl": patient.hdl_cholesterol,
        "tg": patient.triglycerides,
        "sugar": patient.sugar_level,
        "sleep": patient.avg_sleep_hours,
        "steps": patient.avg_daily_steps,
        "symptoms": ", ".join(patient.symptoms),
        "history": ", ".join(patient.medical_history),
        "allergies": ", ".join(patient.allergies),
        "risks": formatted_disease_risks,
        "considerations": considerations_text,
        "evidence": combined_evidence_summary
    })
    
    # Remove medications conflicting with allergies
    if patient.allergies:
        filtered_medications = []
        
        for medication in prescriptions.medications:
            if not any(allergy.lower() in medication.name.lower() for allergy in patient.allergies):
                filtered_medications.append(medication)
                
        prescriptions.medications = filtered_medications
    
    if not prescriptions.medications:
        prescriptions.medications = [Medication(
            name="No immediate pharmacologic option suggested", dose="N/A", mechanism="N/A",
            notes="Follow lifestyle and monitor parameters closely",
            reason="Current findings do not meet threshold for pharmacologic escalation.",
            supporting_guideline="", evidence_source=supported_by_text, confidence="Moderate",
        )]
    if not prescriptions.recommendations:
        prescriptions.recommendations = ["Follow lifestyle and monitor parameters closely"]
    if not prescriptions.instructions:
        prescriptions.instructions = ["Reassess after follow-up"]

    # Feature 2: surface which sources back this plan without changing the
    # PrescriptionPlan schema (per "only create new models when necessary").
    prescriptions.recommendations.append(f"Supported By: {supported_by_text}")
    # NOTE: CLINICAL_DISCLAIMER is intentionally NOT appended here anymore.
    # It used to be pushed into prescriptions.instructions, but app.py's
    # Treatment tab and pdf_report.py both already render CLINICAL_DISCLAIMER
    # explicitly as their own dedicated element — stuffing it into
    # "instructions" too made it print twice back-to-back (visible as the
    # disclaimer literally repeating itself in both the UI and the PDF).
    
    return {
        "prescription_plan": prescriptions
    }


@traceable(name="drug_safety_guardrail", run_type="tool")
def drug_safety_guardrail(state: AgentState) -> dict:
    """
        Rule-based medication safety validation integrating OpenFDA + RxClass checks.
    """
    prescriptions = state.prescription_plan
    patient = state.patient_profile

    unsafe_flags = []
    safe_medications = []

    # Check for patient-specific allergy conditions
    for medication in prescriptions.medications:
        if patient.allergies:
            if any(allergy.lower() in medication.name.lower() for allergy in patient.allergies):
                unsafe_flags.append(f"{medication.name} contraindicated due to allergy")
                continue
        safe_medications.append(medication)

    # Check for drug interactions via RxClass
    med_classes = {}
    for med in safe_medications:
        rxcui = get_rxcui(med.name)
        if rxcui:
            classes = get_drug_classes_by_rxcui(rxcui)
            med_classes[med.name] = classes
        else:
            med_classes[med.name] = []

    anticoagulants = [m for m, cls in med_classes.items() if any("anticoagulant" in c for c in cls)]
    nsaids = [m for m, cls in med_classes.items() if any("nonsteroidal" in c or "nsaid" in c for c in cls)]

    for a in anticoagulants:
        for n in nsaids:
            unsafe_flags.append(f"{a} + {n}: potential bleeding risk (class interaction)")

    # Existing patient-specific checks
    medication_names = [med.name.lower() for med in safe_medications]

    # Blood sugar
    if "metformin" in medication_names and patient.sugar_level < 4.0:
        unsafe_flags.append("Metformin unsafe in hypoglycemia")

    # Blood pressure
    if patient.systolic_bp < 100:
        for med in safe_medications:
            if "lisinopril" in med.name.lower():
                unsafe_flags.append("Lisinopril unsafe during hypotension")

    # Update prescription plan
    state.prescription_plan.medications = safe_medications

    # Update clinical alert with all interaction flags
    if state.clinical_alert:
        state.clinical_alert.interaction_flags.extend(unsafe_flags)
    else:
        state.clinical_alert = ClinicalAlert(
            urgency="LOW",
            message="Drug safety check completed",
            conditions_flagged=[],
            interaction_flags=unsafe_flags,
            recommended_action="Review unsafe medications and adjust accordingly"
        )

    return {
        "prescription_plan": state.prescription_plan,
        "clinical_alert": state.clinical_alert
    }

@traceable(name="clinical_lifestyle_advice", run_type="chain")
def clinical_lifestyle_advice(state: AgentState) -> dict:
    """  
        Gives a personalized, evidence-based lifestyle advice and metabolic optimization plan.
    """
    patient = state.patient_profile
    risk_assessment = state.risk_assessment
    medical_evidence = state.medical_evidence
    guideline_evidence = state.guideline_evidence
    prescription_plan = state.prescription_plan

    combined_evidence_summary = (
        guideline_evidence.merged_summary
        if guideline_evidence is not None and guideline_evidence.merged_summary
        else medical_evidence.clinical_summary
    )
    
    # Derive metabolic indicators
    bmi = patient.bmi
    pulse_pressure = patient.pulse_pressure
    
    # Basal Metabolic Rate (Mifflin-St Jeor Approximation)
    if patient.gender.lower() == "male":
        bmr = (10 * patient.weight) + (6.25 * patient.height * 100) - (5 * patient.age) + 5
    else:
        bmr = (10 * patient.weight) + (6.25 * patient.height * 100) - (5 * patient.age) - 161
        
    # Activity factor approximation
    if patient.avg_daily_steps < 5000:
        activity_factor = 1.2
    elif patient.avg_daily_steps < 8000:
        activity_factor = 1.375
    else:
        activity_factor = 1.55
        
    # Estimated TDEE (Total Daily Energy Expenditure)
    estimated_tdee = round(bmr * activity_factor, 2) 
    
    # Risk context formatting
    formatted_risks = "\n".join(
        f"{risk.disease_name}: {round(risk.risk_score*100)}%"
        for risk in risk_assessment.disease_risks
    )
    
    medications_list = ", ".join(
        medication.name for medication in prescription_plan.medications
    ) if prescription_plan else None
    
    # Build structured lifestyle advice prompt
    lifestyle_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            STRICT RULES:
            - Use patient biometrics and risk scores.
            - Integrate medical evidence summary.
            - Do NOT diagnose.
            - Do NOT contradict prescribed medications.
            - Include intensity ranges and safety precautions.
            - Calibrate exercise to cardiovascular risk.
            - Calibrate diet to metabolic and lipid profile.
            - Address sleep physiology and metabolic regulation.
            - Provide clinically cautious recommendations.
            - Output strictly in structured JSON matching LifestylePlan schema.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
                Patient Profile:
                Age: {age}
                Gender: {gender}
                BMI: {bmi:.2f}
                Blood Pressure: {sbp}/{dbp}
                Pulse Pressure: {pulse_pressure}
                LDL: {ldl}
                HDL: {hdl}
                Triglycerides: {tg}
                Sugar Level: {sugar}
                Avg Sleep: {sleep} hrs
                Avg Daily Steps: {steps}
                Estimated TDEE: {tdee} kcal/day
                Current Medications: {medications}

                Disease Risks:
                {risks}

                Evidence Summary:
                {evidence}

                Generate a world-class, premium clinical lifestyle optimization plan.
                - Structured exercise plan: Include specific modalities (aerobic, resistance, flexibility), intensity (HRR%, RPE), duration, and frequency. Calibrate strictly to {risks}.
                - Structured dietary plan: Provide specific macronutrient ratios, micronutrient focus (e.g., sodium, potassium, fiber), and meal timing strategies based on {tdee} and metabolic profile.
                - Structured sleep optimization: Address circadian alignment, sleep hygiene, and physiological recovery based on current {sleep} hrs.
                - Structured metabolic optimization advice: Provide advanced strategies for glucose management, lipid optimization, and hormonal balance.
                
                Recommendations must be state-of-the-art, medically rigorous, and highly personalized. Never use generic or redundant dummy values.
                Never return NULL or empty values for any of the above fields.
                Ensure output strictly matches the LifestylePlan schema.
            """
        )
    ])
    
    lifestyle_chain = lifestyle_prompt | llm.with_structured_output(LifestylePlan)
    
    lifestyle_suggestions = lifestyle_chain.invoke({
        "age": patient.age,
        "gender": patient.gender,
        "bmi": bmi,
        "sbp": patient.systolic_bp,
        "dbp": patient.diastolic_bp,
        "pulse_pressure": pulse_pressure,
        "ldl": patient.ldl_cholesterol,
        "hdl": patient.hdl_cholesterol,
        "tg": patient.triglycerides,
        "sugar": patient.sugar_level,
        "sleep": patient.avg_sleep_hours,
        "steps": patient.avg_daily_steps,
        "tdee": estimated_tdee,
        "medications": medications_list,
        "risks": formatted_risks,
        "evidence": combined_evidence_summary
    })
    
    # Ensure all recommendations are present and premium
    if not lifestyle_suggestions.exercises:
        lifestyle_suggestions.exercises = ["Initiate Zone 2 cardiovascular training: 30-45 min, 3-4x weekly", "Incorporate progressive resistance training: 2x weekly, major muscle groups"]
    if not lifestyle_suggestions.diet:
        lifestyle_suggestions.diet = ["Adopt Mediterranean-style dietary pattern: focus on monounsaturated fats and high-fiber plant-based foods", "Limit processed carbohydrates and added sugars to optimize glycemic response"]
    if not lifestyle_suggestions.sleep:
        lifestyle_suggestions.sleep = ["Standardize sleep-wake cycle: ±30 min consistency", "Implement 60-min pre-sleep physiological down-regulation (no blue light, temperature optimization)"]
    if not lifestyle_suggestions.metabolic_advice:
        lifestyle_suggestions.metabolic_advice = ["Perform post-prandial walking: 10-15 min after largest meals to blunt glucose spikes", "Monitor continuous glucose trends or fasting levels weekly"]
    
    # Avoid high protein suggestions if kidney disease is present in history
    if any("kidney" in condition.lower() for condition in patient.medical_history):
        lifestyle_suggestions.diet = [
            diet for diet in lifestyle_suggestions.diet if "high_protein" not in diet.lower()
        ]
    
    # Deduplicate entries
    lifestyle_suggestions.exercises = list(dict.fromkeys(lifestyle_suggestions.exercises))
    lifestyle_suggestions.diet = list(dict.fromkeys(lifestyle_suggestions.diet))
    lifestyle_suggestions.sleep = list(dict.fromkeys(lifestyle_suggestions.sleep))
    lifestyle_suggestions.metabolic_advice = list(dict.fromkeys(lifestyle_suggestions.metabolic_advice))

    # Personalized numeric lifestyle targets, derived deterministically from
    # the same biometrics/TDEE already computed above (no extra LLM call,
    # so these numbers are consistent and auditable rather than generic text).
    if bmi >= 25:
        weight_goal = "Lose 5-7% of current body weight over 6 months"
        calorie_target = int(round((estimated_tdee - 500) / 10) * 10)
    else:
        weight_goal = "Maintain current weight within a healthy range"
        calorie_target = int(round(estimated_tdee / 10) * 10)

    walking_minutes = 45 if patient.avg_daily_steps < 8000 else 30
    resistance_sessions = 3 if any(r.risk_score >= 0.4 for r in risk_assessment.disease_risks) else 2
    sodium_limit = 1500 if (patient.systolic_bp >= 130 or patient.diastolic_bp >= 80) else 2300

    lifestyle_targets = LifestyleTargets(
        current_bmi=round(bmi, 1),
        weight_goal=weight_goal,
        daily_calories_kcal=max(calorie_target, 1200),
        daily_walking_minutes=walking_minutes,
        resistance_sessions_per_week=resistance_sessions,
        sodium_limit_mg=sodium_limit,
    )

    return {"lifestyle_plan": lifestyle_suggestions, "lifestyle_targets": lifestyle_targets}

@traceable(name="clinical_strategy_synthesis", run_type="chain")
def clinical_strategy_synthesis(state: AgentState) -> dict:
    """
        Synthesizes a cohesive Clinical Road Map based on current treatment state.
        Provides a high-level, intuitive treatment strategy that makes sense for the patient.
    """
    patient = state.patient_profile
    risks = state.risk_assessment
    prescriptions = state.prescription_plan
    lifestyle = state.lifestyle_plan
    evidence = state.medical_evidence
    guideline_evidence = state.guideline_evidence
    evidence_ranking_summary = state.evidence_ranking

    combined_evidence_summary = (
        guideline_evidence.merged_summary
        if guideline_evidence is not None and guideline_evidence.merged_summary
        else evidence.clinical_summary
    )
    supported_by_text = (
        ", ".join(guideline_evidence.supported_by) if guideline_evidence and guideline_evidence.supported_by else "PubMed"
    )
    confidence_text = (
        f"{evidence_ranking_summary.overall_recommendation_confidence*100:.0f}%"
        if evidence_ranking_summary is not None else "not yet scored"
    )
    considerations_text = _format_clinical_considerations(risks.clinical_considerations)
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a senior clinical strategist. 
            Your goal is to synthesize all findings into a unified, logically intuitive "Clinical Road Map".
            
            Guidelines:
            - Provide a cohesive treatment narrative (Why this treatment? Why now?).
            - Prioritize the most critical interventions.
            - Explain the synergy between medications and lifestyle optimizations.
            - Set realistic expectations for monitoring and follow-up.
            - Use professional, optimistic, and action-oriented clinical language.
            - Briefly note which evidence sources support the plan and the overall confidence level.
            - If any Clinical Considerations are listed, mention that further evaluation is recommended
              for them, in plain qualitative language only — NEVER attach or imply a percentage/probability
              to a Clinical Consideration (no trained model exists for those).
            - Do NOT diagnose.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Patient Overview:
            Age: {age}, Gender: {gender}, BMI: {bmi:.2f}
            Blood Pressure: {sbp}/{dbp}
            Sugar: {sugar} mmol/L
            
            Risk Summary:
            {risk_summary}

            Clinical Considerations (no ML probability exists for these):
            {considerations}
            
            Prescribed Medications:
            {medications}
            
            Lifestyle Strategy:
            - Exercise: {exercises}
            - Diet: {diet}
            
            Clinical Evidence Insight:
            {evidence_summary}

            Evidence Supported By: {supported_by}
            Overall Recommendation Confidence: {confidence}
            
            Generate a world-class "Clinical Road Map" (2-3 paragraphs) that ties everything together into a strategic treatment plan.
            Do NOT use any em-dash or en-dash at all and NO NEED for any subheadings at all. Only return the content in paragraphs.
            """
        )
    ])
    
    medications_text = "\n".join([f"- {m.name}: {m.dose} ({m.notes})" for m in prescriptions.medications])
    
    road_map = llm.invoke(
        synthesis_prompt.format(
            age=patient.age,
            gender=patient.gender,
            bmi=patient.bmi,
            sbp=patient.systolic_bp,
            dbp=patient.diastolic_bp,
            sugar=patient.sugar_level,
            risk_summary=risks.risk_summary,
            considerations=considerations_text,
            medications=medications_text if medications_text else "None",
            exercises=", ".join(lifestyle.exercises[:2]),
            diet=", ".join(lifestyle.diet[:2]),
            evidence_summary=combined_evidence_summary,
            supported_by=supported_by_text,
            confidence=confidence_text
        )
    ).content

    road_map = f"{road_map}\n\n{CLINICAL_DISCLAIMER}"

    return {"treatment_road_map": road_map}

@traceable(name="generate_followup_plan", run_type="chain")
def generate_followup_plan(state: AgentState) -> dict:
    """
    Feature 4 — Follow-up & Monitoring Planner.
    Runs after Clinical Strategy Synthesis (high-risk path) or General
    Wellness Synthesis (low-risk path) and produces the next review date,
    recommended tests/imaging, monitoring interval, warning symptoms, and
    doctor visit schedule. Feeds into the PDF report (Feature 3).
    """
    risk_assessment = state.risk_assessment
    prescriptions = state.prescription_plan
    alert = state.clinical_alert

    urgency = alert.urgency if alert else "LOW"

    disease_risks_text = "\n".join(
        f"{r.disease_name}: {round(r.risk_score*100)}%" for r in risk_assessment.disease_risks
    )
    # Clinical considerations have no ML score, but the follow-up planner LLM
    # still needs to know about them so it can recommend the right tests
    # (e.g. troponin/ECG for suspected MI, D-dimer/imaging for suspected
    # PE/DVT) — appended as plain text with NO percentage attached. No
    # change needed to followup_planner.py itself: disease_risks_text is
    # just a formatted string parameter to it.
    considerations_text = _format_clinical_considerations(risk_assessment.clinical_considerations)
    if risk_assessment.clinical_considerations:
        disease_risks_text = (
            f"{disease_risks_text}\n\nClinical Considerations (no ML probability, "
            f"factor into recommended tests/imaging qualitatively):\n{considerations_text}"
        )
    medications_text = ", ".join(m.name for m in prescriptions.medications) if prescriptions.medications else "None"

    plan = build_followup_plan(
        llm=llm,
        urgency=urgency,
        disease_risks_text=disease_risks_text,
        medications_text=medications_text,
        risk_flags=risk_assessment.risk_flags,
    )

    # Feature 2 (Improved Clinical Roadmap): structured, deterministic
    # Priority 1-5 cards, computed from data already in state (no extra LLM
    # call needed beyond the narrative treatment_road_map already generated
    # by clinical_strategy_synthesis / general_wellness_synthesis).
    roadmap_priorities = _build_roadmap_priorities(state, plan)

    # Feature 4 (Improved Evaluation Module): expanded, always-on quality panel.
    quality_panel = _compute_quality_panel(state)

    return {
        "followup_plan": plan,
        "roadmap_priorities": roadmap_priorities,
        "quality_panel": quality_panel,
    }


def _build_roadmap_priorities(state: "AgentState", followup_plan: FollowUpPlan) -> List[RoadmapPriority]:
    """Builds the fixed-slot Priority 1-5 roadmap cards from data already
    present in state. Deterministic ordering/goals; rationale text is pulled
    from existing risk/lifestyle data rather than a new LLM call, so this is
    fast and reproducible."""
    patient = state.patient_profile
    risks = {r.disease_name.lower(): r.risk_score for r in state.risk_assessment.disease_risks}
    clinical_considerations = state.risk_assessment.clinical_considerations
    lifestyle = state.lifestyle_plan
    priorities: List[RoadmapPriority] = []
    rank = 1

    # Clinical considerations (suspected conditions with no trained ML model,
    # e.g. Myocardial Infarction/PE/DVT) get top billing when present. They
    # have no score to rank by, but clinically they should never be buried
    # below BP/glycemic priorities just because they can't be quantified.
    if clinical_considerations:
        names = ", ".join(c.disease_name for c in clinical_considerations)
        priorities.append(RoadmapPriority(
            priority_rank=rank,
            title="Evaluate Suspected Conditions",
            goal=f"Clinician workup for: {names}",
            rationale="Flagged by clinical reasoning; no trained ML model exists for these, so no "
                      "probability is assigned — further evaluation is recommended to confirm or rule out.",
        ))
        rank += 1

    bp_elevated = patient.systolic_bp >= 130 or patient.diastolic_bp >= 80
    if bp_elevated:
        priorities.append(RoadmapPriority(
            priority_rank=rank, title="Control Blood Pressure", goal="<140/90 mmHg",
            rationale=f"Current reading {patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f} mmHg is above target.",
        ))
        rank += 1

    diabetes_risk = max([v for k, v in risks.items() if "diab" in k], default=0.0)
    if patient.sugar_level >= 6.9 or diabetes_risk >= 0.4:
        priorities.append(RoadmapPriority(
            priority_rank=rank, title="Improve Glycemic Control", goal="HbA1c <7%",
            rationale=f"Fasting sugar {patient.sugar_level} mmol/L and diabetes risk {diabetes_risk*100:.0f}% indicate active management is needed.",
        ))
        rank += 1

    if patient.ldl_cholesterol >= 100:
        priorities.append(RoadmapPriority(
            priority_rank=rank, title="Reduce LDL Cholesterol", goal="Per applicable lipid guideline (typically <100 mg/dL, <70 mg/dL if high CV risk)",
            rationale=f"Current LDL {patient.ldl_cholesterol} mg/dL exceeds guideline target.",
        ))
        rank += 1

    priorities.append(RoadmapPriority(
        priority_rank=rank, title="Lifestyle Intervention",
        goal=", ".join(lifestyle.exercises[:1] + lifestyle.diet[:1]) if lifestyle and (lifestyle.exercises or lifestyle.diet) else "Structured exercise + dietary plan",
        rationale="Lifestyle change compounds the benefit of any pharmacologic option under consideration.",
    ))
    rank += 1

    priorities.append(RoadmapPriority(
        priority_rank=rank, title="Follow-up Monitoring",
        goal=f"Next review by {followup_plan.next_review_date} ({followup_plan.review_interval_days} days)",
        rationale="Regular reassessment confirms whether the plan above is working and needs adjustment.",
    ))

    return priorities


def _compute_quality_panel(state: "AgentState") -> QualityPanel:
    """Feature 4: expanded evaluation panel computed for every run (not just
    the offline evaluation.py harness), so clinicians can see the AI's own
    confidence in what it just produced."""
    evidence_ranking_summary = state.evidence_ranking
    guideline_evidence = state.guideline_evidence
    clinical_alert = state.clinical_alert
    road_map = state.treatment_road_map or ""
    evidence_summary = (
        guideline_evidence.merged_summary if guideline_evidence and guideline_evidence.merged_summary
        else (state.medical_evidence.clinical_summary if state.medical_evidence else "")
    )

    faithfulness = None
    if road_map and evidence_summary and "No direct medical evidence" not in evidence_summary \
            and not road_map.startswith("Pipeline halted"):
        try:
            judge_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a strict evaluator. Score how well the CLINICAL ROAD MAP is grounded in "
                 "the EVIDENCE SUMMARY, from 0.0 (contradicts or invents facts) to 1.0 (fully "
                 "consistent). Reply with ONLY a single number between 0 and 1, nothing else."),
                ("human", "EVIDENCE SUMMARY:\n{evidence}\n\nCLINICAL ROAD MAP:\n{roadmap}")
            ])
            raw = llm.invoke(judge_prompt.format(evidence=evidence_summary, roadmap=road_map)).content
            faithfulness = max(0.0, min(1.0, float(str(raw).strip().split()[0])))
        except Exception:
            faithfulness = None

    evidence_coverage = sorted(guideline_evidence.supported_by) if guideline_evidence and guideline_evidence.supported_by else []
    guideline_coverage = sorted({g.organization for g in guideline_evidence.guidelines}) if guideline_evidence and guideline_evidence.guidelines else []
    retrieved_sources_count = len(evidence_ranking_summary.ranked_items) if evidence_ranking_summary else 0
    recommendation_confidence = evidence_ranking_summary.overall_recommendation_confidence if evidence_ranking_summary else None

    # Groundedness: a cheap, deterministic proxy — did we actually have
    # retrievable evidence backing this plan, independent of the LLM judge above.
    if retrieved_sources_count == 0:
        groundedness = 0.3
    else:
        groundedness = min(1.0, 0.5 + 0.1 * min(retrieved_sources_count, 5))

    if faithfulness is None:
        hallucination_risk = "UNKNOWN"
    elif faithfulness >= 0.8:
        hallucination_risk = "LOW"
    elif faithfulness >= 0.55:
        hallucination_risk = "MODERATE"
    else:
        hallucination_risk = "HIGH"

    drug_safety_status = "N/A"
    if clinical_alert is not None:
        drug_safety_status = "FLAGGED" if clinical_alert.interaction_flags else "PASS"

    clinical_considerations_pending = len(state.risk_assessment.clinical_considerations) if state.risk_assessment else 0

    return QualityPanel(
        groundedness=round(groundedness, 3),
        faithfulness=round(faithfulness, 3) if faithfulness is not None else None,
        evidence_coverage=evidence_coverage,
        guideline_coverage=guideline_coverage,
        retrieved_sources_count=retrieved_sources_count,
        hallucination_risk=hallucination_risk,
        clinical_considerations_pending=clinical_considerations_pending,
        recommendation_confidence=recommendation_confidence,
        drug_safety_status=drug_safety_status,
    )


@traceable(name="alert_clinician", run_type="chain")
def alert_clinician(state: AgentState) -> dict:
    """ 
        Implements an alerting mechanism to notify clinicians of any significant changes in patient risk state.
    """
    patient = state.patient_profile
    risk_assessment = state.risk_assessment
    medical_evidence = state.medical_evidence
    
    disease_risks = risk_assessment.disease_risks
    clinical_considerations = risk_assessment.clinical_considerations
    risk_flags = risk_assessment.risk_flags
    
    HIGH_THRESHOLD = 0.60
    CRITICAL_THRESHOLD = 0.80
    
    high_risk_conditions = []
    critical_conditions = []
    
    # Risk stratification (ML-scored diseases only — clinical_considerations
    # have no score by design and are handled separately below).
    for risk in disease_risks:
        if risk.risk_score >= CRITICAL_THRESHOLD:
            critical_conditions.append((risk.disease_name, risk.risk_score))
        elif risk.risk_score >= HIGH_THRESHOLD:
            high_risk_conditions.append((risk.disease_name, risk.risk_score))
            
    # Check for multi-morbidity (Meaning more than one condition is at high or critical risk)
    multi_morbidity_flag = False
    
    if len(critical_conditions) > 1:
        multi_morbidity_flag = True
    
    # Escalation due to interacting risks
    interaction_flags = []
    
    if "high_diabetes_risk" in risk_flags and "high_cardiovascular_risk" in risk_flags:
        interaction_flags.append("Diabetes-Cardiovascular interaction risk")
    
    if "metabolic_syndrome_risk" in risk_flags and "hypertension_risk" in risk_flags:
        interaction_flags.append("Metabolic syndrome-Hypertension interaction risk")
        
    if interaction_flags:
        multi_morbidity_flag = True
    
    # Assign emergency tiers
    if len(critical_conditions) >= 2:
        urgency = "CRITICAL"
    elif critical_conditions:
        urgency = "HIGH"
    elif multi_morbidity_flag:
        urgency = "HIGH"
    elif high_risk_conditions:
        urgency = "MODERATE"
    else:
        urgency = "LOW"

    # clinical_considerations (e.g. suspected MI/PE/DVT flagged by the LLM)
    # have no probability score, so they can never trip the thresholds above.
    # Without this, a patient this node was routed to specifically because of
    # such a consideration (see clinical_triage_router) could still receive
    # "No immediate clinician escalation required" below — a silent
    # patient-safety regression. A consideration floors urgency at MODERATE
    # and does not downgrade an already-higher ML-derived tier.
    if clinical_considerations and urgency == "LOW":
        urgency = "MODERATE"
    
    # If urgency is LOW, no alert escalation is needed
    if urgency == "LOW":
        return {
            "clinical_alert": ClinicalAlert(
                urgency="LOW",
                message="No immediate clinician escalation required.",
                conditions_flagged=[],
                interaction_flags=[],
                recommended_action="Continue routine monitoring."
            )
        }
        
    # Create structured clinical context
    formatted_risks = "\n".join(
        f"{risk.disease_name}: {round(risk.risk_score * 100)}%"
        for risk in disease_risks
    )
    considerations_text = "\n".join(
        f"{c.disease_name} (clinical consideration, no ML probability): {c.reasoning}"
        for c in clinical_considerations
    ) if clinical_considerations else "None"

    critical_text = "\n".join(
        f"{disease} ({round(score * 100)}%)"
        for disease, score in critical_conditions
    ) if critical_conditions else "None"

    high_risk_text = "\n".join(
        f"{disease} ({round(score * 100)}%)"
        for disease, score in high_risk_conditions
    ) if high_risk_conditions else "None"
    
    interaction_text = "\n".join(interaction_flags) if interaction_flags else "None"
    
    # Create a structured clinician alert prompt
    alert_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a hospital-grade clinical escalation engine.

            Your role:
            - Generate a structured clinician alert covering ONLY: urgency tier, the findings that
              triggered it, and the potential complications if unaddressed.
            - Use cautious, non-diagnostic, clinician-friendly language (e.g. "findings suggest
              elevated risk"), never directive orders like "hospitalize immediately" —
              escalation decisions belong to the clinician.
            - Do NOT include a "Recommendations", "Next Steps", or "Clinical Implications" section
              and do NOT suggest monitoring intensity, referral level, or any other next action —
              that is generated separately (as `recommended_action`) and shown right below your
              message, so repeating it here creates duplicate, redundant text on screen.
            - DO NOT prescribe medications.
            - DO NOT diagnose.
            - DO NOT make definitive hospitalization decisions.
            - Maintain clinical professionalism. Keep it to 1 short paragraph plus, if helpful, a
              brief bullet list of the specific complications — nothing else.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Patient Profile:
            Age: {age}
            Gender: {gender}
            BMI: {bmi:.2f}
            Blood Pressure: {sbp}/{dbp}
            LDL: {ldl}
            HDL: {hdl}
            Triglycerides: {tg}
            Sugar Level: {sugar}

            Risk Scores:
            {risks}

            Critical Conditions:
            {critical}

            High Risk Conditions:
            {high}

            Interaction Risks:
            {interaction}

            Clinical Considerations (no ML model exists for these — do NOT attach or imply
            any percentage/probability for them in your summary, reference them qualitatively only):
            {considerations}

            Evidence Context:
            {evidence}

            Urgency Tier: {urgency}

            Generate the findings + potential complications alert summary described above.
            Do not include recommendations or next steps. Do not assign a probability to anything
            listed under "Clinical Considerations".
            """
        )
    ])
    
    alert_summary = llm.invoke(
        alert_prompt.format(
            age=patient.age,
            gender=patient.gender,
            bmi=patient.bmi,
            sbp=patient.systolic_bp,
            dbp=patient.diastolic_bp,
            ldl=patient.ldl_cholesterol,
            hdl=patient.hdl_cholesterol,
            tg=patient.triglycerides,
            sugar=patient.sugar_level,
            risks=formatted_risks,
            critical=critical_text,
            high=high_risk_text,
            interaction=interaction_text,
            considerations=considerations_text,
            evidence=medical_evidence.clinical_summary if medical_evidence else "No evidence found",
            urgency=urgency
        )
    ).content
    
    # Recommend escalation actions — Feature 8: clinician-friendly framing.
    # The AI flags priority and risk; it never issues a directive clinical
    # order like "hospitalize immediately" — that decision belongs to the
    # treating clinician.
    if urgency == "CRITICAL":
        recommended_action = (
            "The patient's current findings suggest markedly elevated and potentially interacting "
            "clinical risks. Immediate clinician assessment is recommended. Consider emergency "
            "referral if symptoms suggest acute coronary syndrome, stroke, severe hypertension, "
            "or diabetic emergency."
        )
    elif urgency == "HIGH":
        recommended_action = (
            "The patient's current findings suggest elevated clinical risk. Priority clinician "
            "evaluation within 1-2 weeks is recommended, with increased monitoring intensity in "
            "the interim."
        )
    else:
        recommended_action = (
            "Findings suggest moderately elevated risk. Clinician review and reassessment within "
            "4-6 weeks is recommended, alongside routine monitoring."
        )
        
    # conditions_flagged mixes ML-scored names (which the UI/PDF display
    # alongside a % elsewhere) with clinical_consideration names, so those
    # are labeled inline here to avoid implying a score exists for them.
    ml_conditions_flagged = [d for d, _ in critical_conditions + high_risk_conditions]
    consideration_conditions_flagged = [
        f"{c.disease_name} (clinical consideration)" for c in clinical_considerations
    ]

    clinical_alert = ClinicalAlert(
        urgency=urgency,
        message=alert_summary,
        conditions_flagged=ml_conditions_flagged + consideration_conditions_flagged,
        interaction_flags=interaction_flags,
        recommended_action=recommended_action
    )
    
    return {"clinical_alert": clinical_alert}

@traceable(name="general_wellness_synthesis", run_type="chain")
def general_wellness_synthesis(state: AgentState) -> dict:
    """
    Provides high-level preventative health and wellness advice for low-risk patients.
    """
    patient = state.patient_profile
    
    wellness_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You are a preventative health specialist. 
            The patient is currently in a low-risk category.
            Your goal is to provide a positive, encouraging "Wellness Road Map" focused on longevity and health optimization.
            
            Guidelines:
            - Focus on preventative measures (Sleep, Diet, Stress).
            - Set optimization goals (e.g., step targets, sleep hygiene).
            - Keep it professional, optimistic, and easy to follow.
            - Do NOT provide clinical prescriptions or intensive medical warnings.
            """
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Patient Overview:
            Age: {age}, Gender: {gender}, BMI: {bmi:.2f}
            
            Current Health Markers:
            BP: {sbp}/{dbp}, Sugar: {sugar} mmol/L
            
            Generate a concise "Wellness Road Map" (1-2 paragraphs) for this patient.
            """
        )
    ])
    
    wellness_map = llm.invoke(
        wellness_prompt.format(
            age=patient.age,
            gender=patient.gender,
            bmi=patient.bmi,
            sbp=patient.systolic_bp,
            dbp=patient.diastolic_bp,
            sugar=patient.sugar_level
        )
    ).content

    wellness_map = f"{wellness_map}\n\n{CLINICAL_DISCLAIMER}"

    return {"treatment_road_map": wellness_map}


def human_review_gate(state: AgentState) -> dict:
    """
    Human-in-the-loop checkpoint. When the AI has flagged HIGH or CRITICAL
    urgency, the pipeline pauses here (via LangGraph's interrupt()) and waits
    for an explicit clinician decision before any prescription or treatment
    plan is generated. LOW/MODERATE cases pass straight through.

    Resuming happens by invoking the graph with:
        Command(resume={"approved": True/False, "notes": "optional clinician notes"})
    using the SAME thread_id config the original invoke() used.

    The interrupt() payload below is intentionally rich: it carries
    everything the pre-approval review screen needs (ML probabilities, SHAP
    top contributors, clinical considerations, reliability warnings) so the
    clinician has enough information to make an informed decision BEFORE
    approving, rather than only being shown the same summary the AI already
    used to trigger this gate. This is a pure data addition — it does not
    change the graph, the node's pause/resume mechanics, or the
    approved/notes resume contract.

    NOTE: intentionally NOT decorated with @traceable. interrupt() pauses
    execution by raising a special internal exception, and @traceable would
    log every pause as a failed run in LangSmith. LangGraph already traces
    this node correctly (as a real pause, not an error) since it's a
    registered graph node, so no extra decorator is needed here.
    """
    alert = state.clinical_alert

    if not alert or alert.urgency not in ("HIGH", "CRITICAL"):
        return {}

    ml_disease_risks = [
        {"disease_name": r.disease_name, "risk_score": r.risk_score, "reasoning": r.reasoning}
        for r in state.risk_assessment.disease_risks
    ]
    clinical_considerations = [
        {"disease_name": c.disease_name, "reasoning": c.reasoning, "contributing_factors": c.contributing_factors}
        for c in state.risk_assessment.clinical_considerations
    ]
    # Reuses the EXISTING explainability_report (Feature 1) computed earlier
    # in explain_disease_risk — no new SHAP computation happens here, this
    # only reformats what's already in state for the review-gate payload.
    shap_summary = [
        {
            "disease_name": exp.disease_name,
            "risk_score": exp.risk_score,
            "top_contributors": [
                {
                    "feature_name": c.feature_name,
                    "direction": c.direction,
                    "contribution_pct": c.contribution_pct,
                }
                for c in exp.top_contributors[:3]
            ],
        }
        for exp in (state.explainability_report.disease_explanations if state.explainability_report else [])
    ]
    # Reuses the EXISTING per-model reliability notes (MODEL_RELIABILITY_NOTES
    # in ml_disease_models.py) already attached to each ml_disease_predictions
    # entry — only included here when a note actually exists.
    reliability_warnings = [
        {"disease_name": p.disease_name, "severity": p.reliability_severity, "note": p.reliability_note}
        for p in state.ml_disease_predictions if p.reliability_note
    ]

    decision = interrupt({
        "type": "clinician_review_required",
        "urgency": alert.urgency,
        "message": alert.message,
        "conditions_flagged": alert.conditions_flagged,
        "interaction_flags": alert.interaction_flags,
        "recommended_action": alert.recommended_action,
        "ml_disease_risks": ml_disease_risks,
        "shap_summary": shap_summary,
        "clinical_considerations": clinical_considerations,
        "reliability_warnings": reliability_warnings,
    })

    approved = True
    notes = ""
    if isinstance(decision, dict):
        approved = decision.get("approved", True)
        notes = decision.get("notes", "")

    if not approved:
        halted_alert = alert.model_copy(deep=True)
        halted_alert.message = (
            halted_alert.message
            + f" [HALTED: clinician did not approve automated continuation. Notes: {notes or 'none'}]"
        )
        return {
            "clinical_alert": halted_alert,
            "treatment_road_map": "Pipeline halted pending clinician review. No prescriptions or lifestyle plan were generated.",
            "hitl_halted": True,
        }

    return {}


@traceable(name="hitl_router", run_type="tool")
def hitl_router(state: AgentState) -> Literal["proceed", "halt"]:
    """Routes to END if the clinician rejected continuation at the review gate.
    Uses the explicit hitl_halted flag (set only in human_review_gate's
    rejection branch) rather than string-matching treatment_road_map, which
    was fragile: any other code path that happened to set the same string,
    or any future edit to one copy of that string without the other, would
    silently break routing."""
    return "halt" if state.hitl_halted else "proceed"


# Initiate a StateGraph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("collect_patient_data", collect_patient_data)
graph.add_node("detect_early_disease", early_disease_detection)
graph.add_node("explain_disease_risk", explain_disease_risk)
graph.add_node("alert_clinician", alert_clinician)
graph.add_node("human_review_gate", human_review_gate)
graph.add_node("fetch_medical_literature", fetch_medical_literature)
graph.add_node("fetch_clinical_guidelines", fetch_clinical_guidelines)
graph.add_node("prescribe_medications", prescribe_medications)
graph.add_node("drug_safety_guardrails", drug_safety_guardrail)
graph.add_node("give_lifestyle_advice", clinical_lifestyle_advice)
graph.add_node("clinical_strategy_synthesis", clinical_strategy_synthesis)
graph.add_node("general_wellness_synthesis", general_wellness_synthesis)
graph.add_node("generate_followup_plan", generate_followup_plan)

# Add edges to the graph
graph.add_edge(START, "collect_patient_data")
graph.add_edge("collect_patient_data", "detect_early_disease")

# Feature 1: Explainable AI runs right after detection, before triage routing,
# so both the high-risk and low-risk paths get a "why" behind the risk score.
graph.add_edge("detect_early_disease", "explain_disease_risk")

# Conditional Risk Triage
graph.add_conditional_edges(
    "explain_disease_risk",
    clinical_triage_router,
    {
        "high_risk": "alert_clinician",
        "low_risk": "general_wellness_synthesis"
    }
)

# High-Risk Path: route through a human sign-off gate before any treatment is generated
graph.add_edge("alert_clinician", "human_review_gate")
graph.add_conditional_edges(
    "human_review_gate",
    hitl_router,
    {
        "proceed": "fetch_medical_literature",
        "halt": END
    }
)

# Feature 2 + 5: retrieve/rank/merge clinical guidelines (ADA/AHA/WHO/NICE/CDC)
# with the existing PubMed evidence before medications/lifestyle are generated.
graph.add_edge("fetch_medical_literature", "fetch_clinical_guidelines")

# Parallel Execution Track (Medications & Lifestyle)
graph.add_edge("fetch_clinical_guidelines", "prescribe_medications")
graph.add_edge("fetch_clinical_guidelines", "give_lifestyle_advice")

# Add guardrails check immediately after prescriptions are generated
graph.add_edge("prescribe_medications", "drug_safety_guardrails")

# Synchronizing parallel tracks into the final clinical synthesis
graph.add_edge("drug_safety_guardrails", "clinical_strategy_synthesis")
graph.add_edge("give_lifestyle_advice", "clinical_strategy_synthesis")

# Feature 4: Follow-up & Monitoring Planner runs after either synthesis path,
# then both converge to END.
graph.add_edge("clinical_strategy_synthesis", "generate_followup_plan")
graph.add_edge("general_wellness_synthesis", "generate_followup_plan")
graph.add_edge("generate_followup_plan", END)

# Compile the graph with a checkpointer (required for interrupt()/human-in-the-loop to work)
checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# Visualize the graph
try:
    with open("clinical_patient_monitoring_workflow.png", "wb") as f:
        f.write(workflow.get_graph().draw_mermaid_png())
    print("Workflow visualization saved to 'clinical_patient_monitoring_workflow.png'")
except Exception as e:
    print(f"Could not save workflow visualization: {e}")
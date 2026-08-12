"""
explainability.py — Feature 1: Explainable AI for disease risk predictions.

UPDATED: this module used to build a small clinically-weighted logistic
"surrogate" model per disease and run SHAP's KernelExplainer against it,
because the risk scores previously came from an opaque LLM call that had no
queryable model function to explain directly.

That's no longer true for the four diseases with trained models
(hypertension, stroke, diabetes, cardiovascular — see ml_disease_models.py):
those risk scores now come directly from real XGBoost classifiers, each with
its own real shap.TreeExplainer computed at training time. There is nothing
left to approximate — ml_disease_models.py already computes the exact SHAP
contribution of every feature for the exact row that produced the risk
score, and hands it to this module as `MLDiseasePrediction.shap_contributions`.

This module's job is now purely presentational: turn those real SHAP values
into the same RiskFactorContribution / DiseaseExplanation / ExplainabilityReport
shapes the rest of the app (Streamlit, PDF report) already expects, so no
downstream rendering code needs to change.

For any disease NOT covered by a trained model (e.g. an LLM-surfaced
oncology/infectious consideration outside the four core ML diseases), there
is no model to explain — `generate_explanations` reports that plainly rather
than fabricating a SHAP-style breakdown for a non-existent model.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from ml_disease_models import MLDiseasePrediction
except ImportError:  # pragma: no cover - allows standalone import/testing
    MLDiseasePrediction = object  # type: ignore


# ---------------------------------------------------------------------------
# Pydantic models (UNCHANGED — downstream code depends on this exact shape)
# ---------------------------------------------------------------------------

class RiskFactorContribution(BaseModel):
    feature_name: str = Field(..., description="Human-readable clinical feature name")
    direction: Literal["increases_risk", "decreases_risk", "neutral"] = Field(
        ..., description="Whether this feature pushed the risk score up or down"
    )
    contribution_pct: float = Field(..., description="Signed % contribution to the overall risk score")
    raw_value: float = Field(..., description="Patient's raw/engineered value fed to the model for this feature")
    reference_range: str = Field(..., description="Normal/reference context for this feature")


class DiseaseExplanation(BaseModel):
    disease_name: str = Field(..., description="Disease this explanation applies to")
    risk_score: float = Field(..., description="Model-predicted risk score (0.0-1.0)")
    top_contributors: List[RiskFactorContribution] = Field(default_factory=list)
    confidence_note: str = Field(..., description="Plain-language confidence/calibration note")


class ExplainabilityReport(BaseModel):
    method: str = Field(
        default="SHAP TreeExplainer (trained XGBoost model)",
        description="Explanation methodology used"
    )
    disease_explanations: List[DiseaseExplanation] = Field(default_factory=list)
    summary: str = Field(default="", description="One-paragraph plain-language summary across all diseases")


# ---------------------------------------------------------------------------
# Human-readable labels + reference context for engineered feature names.
# Only used for display; has no effect on scoring. Anything not listed here
# falls back to showing the raw engineered value with no reference range,
# which is still accurate — just less prettified (e.g. an unseen feature
# from a retrained model won't silently mislabel).
# ---------------------------------------------------------------------------

_FEATURE_DISPLAY = {
    # Hypertension
    "Age": ("Age", ""),
    "Salt_Intake": ("Salt Intake", "<6 g/day recommended"),
    "Stress_Score": ("Stress Score", "0 (low) - 10 (high)"),
    "Sleep_Duration": ("Sleep Duration", "7-9 hours"),
    "BMI": ("BMI", "18.5-24.9 kg/m2"),
    "BP_History_Normal": ("BP History: Normal", "reference category = Hypertension"),
    "BP_History_Prehypertension": ("BP History: Prehypertension", "reference category = Hypertension"),
    "Family_History_Yes": ("Family History", "0 = No, 1 = Yes"),
    "Exercise_Level_Low": ("Exercise Level: Low", "reference category = High"),
    "Exercise_Level_Moderate": ("Exercise Level: Moderate", "reference category = High"),
    "Smoking_Status_Smoker": ("Current Smoker", "0 = No, 1 = Yes"),
    # Stroke
    "sex": ("Sex", "label-encoded"),
    "hypertension": ("Hypertension History", "0 = No, 1 = Yes"),
    "heart_disease": ("Heart Disease History", "0 = No, 1 = Yes"),
    "ever_married": ("Ever Married", "0 = No, 1 = Yes"),
    "work_type": ("Occupation Type", "label-encoded"),
    "Residence_type": ("Residence Type", "0 = Rural, 1 = Urban"),
    "avg_glucose_level": ("Average Glucose Level", "70-140 mg/dL"),
    "bmi": ("BMI", "18.5-24.9 kg/m2"),
    "smoking_status": ("Smoking Status", "label-encoded"),
    # Diabetes (BRFSS)
    "Sex": ("Sex", "0 = Female, 1 = Male"),
    "HighChol": ("High Cholesterol", "0 = No, 1 = Yes"),
    "CholCheck": ("Cholesterol Checked (5 yrs)", "0 = No, 1 = Yes"),
    "Smoker": ("Ever Smoker", "0 = No, 1 = Yes"),
    "HeartDiseaseorAttack": ("Heart Disease/Attack History", "0 = No, 1 = Yes"),
    "PhysActivity": ("Physically Active", "0 = No, 1 = Yes"),
    "Fruits": ("Daily Fruit Intake", "0 = No, 1 = Yes"),
    "Veggies": ("Daily Vegetable Intake", "0 = No, 1 = Yes"),
    "HvyAlcoholConsump": ("Heavy Alcohol Use", "0 = No, 1 = Yes"),
    "GenHlth": ("Self-Rated General Health", "1 (excellent) - 5 (poor)"),
    "MentHlth": ("Poor Mental Health Days (30d)", "0-30 days"),
    "PhysHlth": ("Poor Physical Health Days (30d)", "0-30 days"),
    "DiffWalk": ("Difficulty Walking", "0 = No, 1 = Yes"),
    "Stroke": ("Stroke History", "0 = No, 1 = Yes"),
    "HighBP": ("High Blood Pressure", "0 = No, 1 = Yes"),
    # Heart
    "Cholesterol": ("Cholesterol", "<200 mg/dL desirable"),
    "Blood Pressure": ("Systolic Blood Pressure", "90-120 mmHg"),
    "Heart Rate": ("Heart Rate", "60-100 bpm"),
    "Exercise Hours": ("Weekly Exercise Hours", "hours/week"),
    "Stress Level": ("Stress Level", "0 (low) - 10 (high)"),
    "Blood Sugar": ("Blood Sugar", "70-140 mg/dL"),
    "Gender_Male": ("Male", "0 = No, 1 = Yes"),
    "Smoking_Former": ("Former Smoker", "reference category = Current"),
    "Smoking_Never": ("Never Smoked", "reference category = Current"),
    "Alcohol Intake_Moderate": ("Moderate Alcohol Intake", "0 = No, 1 = Yes"),
    "Family History_Yes": ("Family History", "0 = No, 1 = Yes"),
    "Diabetes_Yes": ("Diabetes History", "0 = No, 1 = Yes"),
    "Obesity_Yes": ("Obesity (BMI>=30)", "0 = No, 1 = Yes"),
    "Exercise Induced Angina_Yes": ("Exercise-Induced Angina", "0 = No, 1 = Yes"),
    "Chest Pain Type_Atypical Angina": ("Chest Pain: Atypical Angina", "reference category = Asymptomatic"),
    "Chest Pain Type_Non-anginal Pain": ("Chest Pain: Non-anginal", "reference category = Asymptomatic"),
    "Chest Pain Type_Typical Angina": ("Chest Pain: Typical Angina", "reference category = Asymptomatic"),
}


def _display(feature: str) -> tuple[str, str]:
    return _FEATURE_DISPLAY.get(feature, (feature.replace("_", " ").title(), ""))


def _build_explanation_from_prediction(pred: "MLDiseasePrediction", top_n: int = 5) -> DiseaseExplanation:
    contributions = pred.shap_contributions
    if not contributions:
        return DiseaseExplanation(
            disease_name=pred.disease_name,
            risk_score=pred.risk_score,
            top_contributors=[],
            confidence_note=(
                f"This is a real model prediction (trained XGBoost classifier, "
                f"held-out ROC AUC {pred.model_metrics.get('ROC_AUC', 'n/a')}), but a per-feature "
                f"SHAP breakdown could not be computed for this run."
            ),
        )

    total_abs = sum(abs(v) for v in contributions.values()) or 1e-6
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    top_contributors = []
    for feat, contrib in ranked:
        display_name, reference_range = _display(feat)
        pct = round((abs(contrib) / total_abs) * pred.risk_score * 100, 1)
        direction: Literal["increases_risk", "decreases_risk", "neutral"]
        if contrib > 1e-6:
            direction = "increases_risk"
        elif contrib < -1e-6:
            direction = "decreases_risk"
        else:
            direction = "neutral"
        top_contributors.append(
            RiskFactorContribution(
                feature_name=display_name,
                direction=direction,
                contribution_pct=pct if direction != "decreases_risk" else -pct,
                raw_value=round(float(pred.feature_row.get(feat, 0.0)), 2),
                reference_range=reference_range,
            )
        )

    metrics = pred.model_metrics
    metrics_note = (
        f"held-out test ROC AUC {metrics['ROC_AUC']:.3f}, accuracy {metrics['Accuracy']:.3f}"
        if metrics.get("ROC_AUC") is not None
        else "held-out test metrics unavailable"
    )
    confidence_note = (
        f"Explanation generated via SHAP TreeExplainer applied directly to the trained XGBoost "
        f"model ({metrics_note}) that produced this {pred.risk_score*100:.0f}% risk score for "
        f"{pred.disease_name} — this is the model's actual reasoning, not an approximation. "
        f"It is a decision-support aid, not a substitute for clinical judgment."
    )
    if getattr(pred, "reliability_note", None):
        confidence_note += f" ⚠ Known limitation: {pred.reliability_note}"

    return DiseaseExplanation(
        disease_name=pred.disease_name,
        risk_score=pred.risk_score,
        top_contributors=top_contributors,
        confidence_note=confidence_note,
    )


def generate_explanations(
    ml_predictions: List["MLDiseasePrediction"],
) -> ExplainabilityReport:
    """
    Builds an ExplainabilityReport from real model predictions ONLY.

    ml_predictions: output of ml_disease_models.predict_all(patient) — one
        entry per disease with a trained model (hypertension/stroke/
        diabetes/heart), each carrying real SHAP contributions already.

    ML-ONLY-PROBABILITIES FIX: this function used to also accept
    `other_disease_risks` (LLM-surfaced conditions with no trained model)
    and emit a DiseaseExplanation carrying that LLM's risk_score for them.
    That parameter has been removed entirely, not just left unused — the
    goal is for it to be structurally impossible to reach this module with a
    non-ML disease and a score attached. LLM-flagged conditions with no
    trained model (ClinicalConsideration in workflow.py) are rendered
    directly by the UI/PDF report instead, with no score and no SHAP
    section, since "explaining" a model that doesn't exist for them isn't
    meaningful. See workflow.py:explain_disease_risk, which now calls this
    function with only ml_predictions.
    """
    explanations = [_build_explanation_from_prediction(p) for p in ml_predictions]

    if not explanations:
        return ExplainabilityReport(
            disease_explanations=[],
            summary="No elevated disease risks were identified, so no explanation is required.",
        )

    ranked = sorted(explanations, key=lambda e: e.risk_score, reverse=True)
    top = ranked[0]
    if top.top_contributors:
        driver_text = ", ".join(
            f"{c.feature_name} ({'+' if c.direction=='increases_risk' else '-' if c.direction=='decreases_risk' else '~'}{abs(c.contribution_pct):.0f}%)"
            for c in top.top_contributors[:3]
        )
        summary = f"The leading risk driver is {top.disease_name} at {top.risk_score*100:.0f}%, primarily influenced by {driver_text}."
    else:
        summary = f"The leading risk driver is {top.disease_name} at {top.risk_score*100:.0f}%."

    return ExplainabilityReport(disease_explanations=ranked, summary=summary)
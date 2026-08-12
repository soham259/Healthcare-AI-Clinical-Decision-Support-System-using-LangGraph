"""
ml_disease_models.py — Real trained-model disease risk prediction.

REPLACES: the LLM guessing risk scores for hypertension / stroke / diabetes /
heart disease. Those four diseases now get a deterministic, reproducible
probability from your actual trained XGBoost classifiers, with REAL SHAP
TreeExplainer contributions (not a linear surrogate).

WHERE THE MODEL FILES LIVE:
Drop every *.pkl / *_metrics.json you trained into one folder (flat, no
subfolders needed) and point CLINICAL_ML_MODELS_DIR at it, e.g.:

    ml_models/
        1786194038592_hypertension_model.pkl
        1786194038590_hypertension_feature_columns.pkl
        1786194038593_hypertension_shap_explainer.pkl
        1786194038594_hypertension_target_encoder.pkl
        1786194038591_hypertension_metrics.json
        1786194038596_stroke_model.pkl
        ... etc for stroke / diabetes / heart

Files are discovered by glob("*_{disease}_model.pkl") etc., so the numeric/
hash prefix in your filenames doesn't matter and nothing needs renaming.

IMPORTANT — ASSUMPTIONS THAT NEED YOUR SIGN-OFF:
Your PatientProfile collects vitals/symptoms for an open-ended LLM prompt,
not the specific structured/one-hot/survey features these four models were
trained on. Every field below that isn't a 1:1 match to an existing
PatientProfile field is either (a) a NEW optional PatientProfile field with
a clinically-reasonable default, or (b) DERIVED from existing fields using a
named clinical threshold. Every derivation/default is called out in a
comment tagged "ASSUMPTION:". Please check these against your actual
training/preprocessing notebook — if a threshold or category-encoding order
doesn't match how you engineered the training data, predictions will be
silently wrong even though nothing crashes.

The single biggest one: the STROKE model's categorical columns (sex,
ever_married, work_type, Residence_type, smoking_status) were saved as plain
label-encoded integers with NO encoder pickle included. This module assumes
they were encoded with sklearn.LabelEncoder's default alphabetical ordering,
which is the standard/common approach for this dataset. If your training
script encoded them differently (e.g. a custom order, or OneHotEncoder),
stroke predictions will be wrong until _STROKE_ENCODINGS below is corrected.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

MODELS_DIR = os.environ.get("CLINICAL_ML_MODELS_DIR", "ml_models")

# Population-baseline imputation for GenHlth when the patient/clinician
# leaves the optional Self-Rated Health field blank — see
# _general_health_rating's docstring for why this exists and how to correct
# it if you know your training set's real median/mode.
GENHLTH_POPULATION_BASELINE = 2

DISEASE_DISPLAY_NAMES = {
    "hypertension": "Hypertension",
    "stroke": "Stroke",
    "diabetes": "Diabetes",
    "heart": "Cardiovascular Disease",
}


RISK_LEVEL_THRESHOLDS = [
    (0.80, "Very High"),
    (0.60, "High"),
    (0.30, "Moderate"),
    (0.10, "Low"),
    (0.00, "Very Low"),
]

RISK_LEVEL_BADGE = {
    "Very High": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟡", "Very Low": "🟢",
}


def risk_level(score: float) -> str:
    """
    SINGLE canonical mapping from a probability to a 5-tier clinical risk
    level (Very Low / Low / Moderate / High / Very High). Previously app.py
    and pdf_report.py each hardcoded their OWN independent 4-tier threshold
    logic (0.25/0.6/0.8) — duplicated, and a real risk of the UI and the PDF
    silently disagreeing on the same patient's risk level if one was edited
    without the other. Both now import this single function instead.
    """
    for threshold, label in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "Very Low"


MODEL_RELIABILITY_NOTES: Dict[str, dict] = {
    # Populated from empirical investigation of the actual shipped .pkl files
    # (feature-importance inspection + live predict_proba sweeps), not
    # guesswork. See the retrain scripts (retrain_hypertension_model.py,
    # retrain_stroke_model.py) for the recommended fix for each.
    "hypertension": {
        "severity": "critical",
        "message": (
            "Likely data leakage: BP_History_Normal/BP_History_Prehypertension are the top two "
            "features by gain (~37% combined) and are derived from the patient's CURRENT blood "
            "pressure category — i.e. close to the label itself. This explains the near-perfect "
            "ROC AUC (0.9999). Treat this score as a sanity-check only, not a calibrated probability, "
            "until the model is retrained without BP_History features."
        ),
    },
    "stroke": {
        "severity": "warning",
        "message": (
            "RETRAINED (see retrain_stroke_model.py): the prior non-monotonic instability is fixed — "
            "confirmed via a monotonicity sanity check across a glucose sweep, all now correctly "
            "non-decreasing. Remaining caveat: the training data used (stroke_data.csv) is an "
            "artificially rebalanced 50/50 positive/negative dataset, not the true population stroke "
            "prevalence (~1-5%). Relative risk ordering between patients and directional sensitivity "
            "are trustworthy, but absolute probabilities are likely systematically higher than true "
            "population-calibrated risk for any given patient. This model still has no cholesterol "
            "feature at all."
        ),
    },
    "heart": {
        "severity": "critical",
        "message": (
            "CONFIRMED VIA ACTUAL RETRAIN (not just prediction): retraining on the same "
            "heart_disease_dataset.csv, even with substantially stronger regularization, STILL "
            "produces a model dominated by Age+Cholesterol (83% combined gain, down only slightly "
            "from 90%) with 3 features at ~zero importance (Smoking_Never, Obesity_Yes, "
            "Exercise Induced Angina_Yes) and ROC AUC still 1.000. This is now confirmed to be a "
            "training-data problem, not a hyperparameter problem — the labels in this dataset do not "
            "have genuine relationships with most of the clinical features. A different, real-world "
            "dataset is needed; this cannot be fixed by retraining on the same data again. Note: a "
            "separately uploaded file showed genuine chest-pain-type predictive signal, but uses an "
            "entirely different feature schema (resting ECG, stress-test heart rate, ST depression, "
            "fluoroscopy vessel count) not currently collected by this app's intake form — adopting it "
            "would require UI/data-collection changes, not just a model swap."
        ),
    },
    "diabetes": {
        "severity": "warning",
        "message": (
            "RETRAINED (see retrain_diabetes_model.py): the CholCheck confound is removed — a healthy "
            "28yo test profile dropped from ~20.5% to ~17.0%. Remaining caveat: the training data used "
            "(diabetes_data.csv) is the 50/50-rebalanced BRFSS variant, not the true population "
            "diabetes prevalence (~11-13%), so absolute probabilities are likely still somewhat higher "
            "than true population-calibrated risk, even though the CholCheck-specific inflation is "
            "gone and the model's behavior is now more sensible feature-by-feature."
        ),
    },
}


class MLDiseasePrediction(BaseModel):
    """One disease's model output: the probability plus everything
    explainability.py needs to render a real (non-surrogate) SHAP
    explanation, without having to rebuild the feature row itself."""
    disease_key: str = Field(..., description="Internal key: hypertension/stroke/diabetes/heart")
    disease_name: str = Field(..., description="Clinician-facing disease name")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Model predict_proba for the positive class")
    feature_row: Dict[str, float] = Field(..., description="Exact feature vector fed to the model, in training column order")
    shap_contributions: Dict[str, float] = Field(..., description="Per-feature SHAP value for this prediction")
    base_value: float = Field(..., description="SHAP expected_value (model's baseline output) for this prediction")
    model_metrics: Dict[str, float] = Field(default_factory=dict, description="Held-out test metrics for this model, for transparency")
    reasoning: str = Field(default="", description="1-sentence summary of the top SHAP drivers")
    reliability_note: Optional[str] = Field(
        default=None, description="Known-limitation caveat for this specific model, if any (see MODEL_RELIABILITY_NOTES)."
    )
    reliability_severity: Optional[str] = Field(
        default=None, description="'critical' | 'warning' | None — lets the UI choose a badge color."
    )


# ---------------------------------------------------------------------------
# Model / explainer loading (cached per-process, mirrors vector_store.py's
# "build/load once" pattern)
# ---------------------------------------------------------------------------

_REGISTRY_CACHE: Dict[str, dict] = {}


def _find_one(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(MODELS_DIR, pattern)))
    return matches[0] if matches else None


def _load_disease_artifacts(disease_key: str) -> Optional[dict]:
    """Loads {model, feature_columns, shap_explainer, metrics} for one
    disease. Returns None (never raises) if any required file is missing,
    so the caller can degrade gracefully instead of crashing the workflow."""
    if disease_key in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[disease_key]

    model_path = _find_one(f"*{disease_key}_model.pkl")
    fc_path = _find_one(f"*{disease_key}_feature_columns.pkl")
    exp_path = _find_one(f"*{disease_key}_shap_explainer.pkl")
    metrics_path = _find_one(f"*{disease_key}_metrics.json")

    if not (model_path and fc_path and exp_path):
        print(f"[ml_disease_models] Missing artifacts for '{disease_key}' under '{MODELS_DIR}/' — "
              f"model={bool(model_path)} feature_columns={bool(fc_path)} shap_explainer={bool(exp_path)}. "
              "This disease will be skipped.")
        return None

    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(fc_path)
        explainer = joblib.load(exp_path)
        metrics = {}
        if metrics_path and os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
    except Exception as e:
        print(f"[ml_disease_models] Failed to load artifacts for '{disease_key}': {e}")
        return None

    artifacts = {
        "model": model,
        "feature_columns": feature_columns,
        "explainer": explainer,
        "metrics": metrics,
    }
    _REGISTRY_CACHE[disease_key] = artifacts
    return artifacts


# ---------------------------------------------------------------------------
# Shared clinical derivation helpers (used by more than one disease's
# feature builder, kept in one place per the project's "no duplicate logic"
# convention).
# ---------------------------------------------------------------------------

def _bp_history_category(sbp: float, dbp: float) -> str:
    # ASSUMPTION: standard AHA-style staging.
    if sbp >= 140 or dbp >= 90:
        return "Hypertension"
    if sbp >= 120 or dbp >= 80:
        return "Prehypertension"
    return "Normal"


def _has_hypertension(patient) -> int:
    if _bp_history_category(patient.systolic_bp, patient.diastolic_bp) == "Hypertension":
        return 1
    return int(any("hypertens" in h.lower() for h in patient.medical_history))


def _has_condition(patient, keywords: List[str]) -> int:
    history_text = " ".join(patient.medical_history).lower()
    return int(any(k in history_text for k in keywords))


def _exercise_level(steps: Optional[float]) -> str:
    # ASSUMPTION: simple activity tiers from average daily steps.
    steps = steps or 8000
    if steps < 5000:
        return "Low"
    if steps < 10000:
        return "Moderate"
    return "High"


def _smoking_flags(smoking_status: str) -> Dict[str, int]:
    s = (smoking_status or "Never").strip().lower()
    return {
        "is_current": int(s == "current"),
        "is_former": int(s == "former"),
        "is_never": int(s == "never"),
    }


def _mmol_to_mgdl(mmol: float) -> float:
    # Standard glucose unit conversion (patient.sugar_level is mmol/L).
    return round(mmol * 18.0182, 1)


def _obesity_flag(bmi: float) -> int:
    return int(bmi >= 30)


def _chest_pain_type(patient) -> str:
    if patient.chest_pain_type:
        return patient.chest_pain_type
    # ASSUMPTION (fixed): a bare "chest_pain" symptom with no clinician-documented
    # character (exertional pattern, relief with rest/nitro, etc.) must NOT be
    # assumed to be "Typical Angina" -- that is the single most cardiac-specific,
    # highest-risk category in this model's encoding, and defaulting every
    # reported chest pain to it systematically overstates cardiovascular risk
    # for anyone who merely checked the symptom box (chest pain has many causes:
    # musculoskeletal, GI, anxiety, etc.). "Atypical Angina" is the appropriate
    # conservative default: it acknowledges chest pain was reported without
    # claiming the specific diagnostic pattern required to call it "Typical".
    if any("chest_pain" in s.lower() for s in patient.symptoms):
        return "Atypical Angina"
    return "Asymptomatic"


def _general_health_rating(patient) -> int:
    """
    Self-Rated Health (GenHlth, 1=excellent...5=poor) is optional in the UI
    (app.py offers None as the default selectbox choice). When the patient/
    clinician leaves it unanswered, we cannot simply drop this feature (the
    trained model requires a value for every column) or silently force a
    fixed guess without documenting it — so this imputes a population
    baseline (GENHLTH_POPULATION_BASELINE, "Good") and then makes a small,
    bounded adjustment ONLY when we have real supporting signal (documented
    history or multiple reported symptoms), rather than guessing further.

    GENHLTH_POPULATION_BASELINE=2 approximates general-population BRFSS
    self-rated-health distributions (which skew toward "Very Good"/"Good").
    If you have the ACTUAL median/mode from your own training CSV, replace
    the constant below with that exact value for a better-calibrated default.
    """
    if patient.general_health_rating is not None:
        return int(patient.general_health_rating)
    score = GENHLTH_POPULATION_BASELINE
    if patient.medical_history:
        score += 1
    if len(patient.symptoms) >= 3:
        score += 1
    return max(1, min(5, score))


def _exercise_hours_per_week(patient) -> float:
    if patient.exercise_hours_per_week is not None:
        return float(patient.exercise_hours_per_week)
    # ASSUMPTION: rough steps -> weekly moderate-exercise-hours proxy
    # (~20 min of brisk activity per additional 2,000 daily steps).
    steps = patient.avg_daily_steps or 8000
    return round((steps / 2000) * (20 / 60) * 7, 1)


# ---------------------------------------------------------------------------
# Per-disease feature builders — each returns an ordered dict matching that
# model's exact training feature_columns.
# ---------------------------------------------------------------------------

def _build_hypertension_row(patient) -> Dict[str, float]:
    bp_cat = _bp_history_category(patient.systolic_bp, patient.diastolic_bp)
    exercise = _exercise_level(patient.avg_daily_steps)
    smoke = _smoking_flags(patient.smoking_status)
    family_hyp = int(any("hypertens" in f.lower() or "cardio" in f.lower() or "heart" in f.lower()
                          for f in (patient.family_history or [])))
    return {
        "Age": patient.age,
        "Salt_Intake": patient.salt_intake_g_per_day,
        "Stress_Score": patient.stress_score,
        "Sleep_Duration": patient.avg_sleep_hours,
        "BMI": patient.bmi,
        "BP_History_Normal": int(bp_cat == "Normal"),
        "BP_History_Prehypertension": int(bp_cat == "Prehypertension"),
        "Family_History_Yes": family_hyp,
        "Exercise_Level_Low": int(exercise == "Low"),
        "Exercise_Level_Moderate": int(exercise == "Moderate"),
        "Smoking_Status_Smoker": smoke["is_current"],
    }


# ASSUMPTION: alphabetical sklearn.LabelEncoder ordering — see module
# docstring. Verify against your training notebook and correct here if
# different; nothing else needs to change.
_STROKE_ENCODINGS = {
    "sex": {"female": 0, "male": 1, "other": 2},
    "ever_married": {"no": 0, "yes": 1},
    "work_type": {"govt_job": 0, "never_worked": 1, "private": 2, "self-employed": 3, "children": 4},
    "residence_type": {"rural": 0, "urban": 1},
    "smoking_status": {"unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3},
}


def _stroke_smoking_label(smoking_status: str) -> str:
    return {"never": "never smoked", "former": "formerly smoked", "current": "smokes"}.get(
        (smoking_status or "never").lower(), "unknown"
    )


def _build_stroke_row(patient) -> Dict[str, float]:
    enc = _STROKE_ENCODINGS
    return {
        "sex": enc["sex"].get(patient.gender.strip().lower(), 0),
        "age": patient.age,
        "hypertension": _has_hypertension(patient),
        "heart_disease": _has_condition(patient, ["heart_disease", "coronary", "cardiac"]),
        "ever_married": enc["ever_married"].get("yes" if patient.ever_married else "no", 1),
        "work_type": enc["work_type"].get(patient.occupation_type.strip().lower(), 2),
        "Residence_type": enc["residence_type"].get(patient.residence_type.strip().lower(), 1),
        "avg_glucose_level": _mmol_to_mgdl(patient.sugar_level),
        "bmi": patient.bmi,
        "smoking_status": enc["smoking_status"].get(_stroke_smoking_label(patient.smoking_status), 0),
    }


def _build_diabetes_row(patient) -> Dict[str, float]:
    smoke = _smoking_flags(patient.smoking_status)
    # ASSUMPTION: HighChol / HighBP clinical thresholds (ADA/AHA style).
    high_chol = int(patient.cholesterol >= 200 or patient.ldl_cholesterol >= 130)
    high_bp = int(patient.systolic_bp >= 130 or patient.diastolic_bp >= 80)
    return {
        "Age": patient.age,
        "Sex": 1 if patient.gender.strip().lower() == "male" else 0,
        "HighChol": high_chol,
        # NOTE: CholCheck removed (2024 retrain) -- it was hardcoded to 1 for every
        # patient and confirmed to be a case-ascertainment confound inflating risk
        # for everyone uniformly. See MODEL_RELIABILITY_NOTES['diabetes'] and
        # retrain_diabetes_model.py for the evidence and fix.
        "BMI": patient.bmi,
        "Smoker": int(smoke["is_current"] or smoke["is_former"]),  # BRFSS: ever smoked 100+ cigarettes
        "HeartDiseaseorAttack": _has_condition(patient, ["heart_disease", "coronary", "cardiac", "myocardial"]),
        "PhysActivity": int((patient.avg_daily_steps or 0) >= 5000),
        "Fruits": int(patient.diet_fruits_daily),
        "Veggies": int(patient.diet_veggies_daily),
        "HvyAlcoholConsump": int(patient.alcohol_intake == "Heavy"),
        "GenHlth": _general_health_rating(patient),
        "MentHlth": patient.mental_health_poor_days,
        "PhysHlth": patient.physical_health_poor_days,
        "DiffWalk": int(patient.difficulty_walking),
        "Stroke": _has_condition(patient, ["stroke"]),
        "HighBP": high_bp,
    }


def _build_heart_row(patient) -> Dict[str, float]:
    smoke = _smoking_flags(patient.smoking_status)
    chest_pain = _chest_pain_type(patient)
    family_cardiac = int(any("hypertens" in f.lower() or "cardio" in f.lower() or "heart" in f.lower()
                              for f in (patient.family_history or [])))
    return {
        "Age": patient.age,
        "Cholesterol": patient.cholesterol,
        "Blood Pressure": patient.systolic_bp,  # ASSUMPTION: "Blood Pressure" = systolic.
        "Heart Rate": patient.heartbeat_rate,
        "Exercise Hours": _exercise_hours_per_week(patient),
        "Stress Level": patient.stress_score,
        "Blood Sugar": _mmol_to_mgdl(patient.sugar_level),
        "Gender_Male": int(patient.gender.strip().lower() == "male"),
        "Smoking_Former": smoke["is_former"],
        "Smoking_Never": smoke["is_never"],
        "Alcohol Intake_Moderate": int(patient.alcohol_intake == "Moderate"),
        "Family History_Yes": family_cardiac,
        "Diabetes_Yes": _has_condition(patient, ["diabetes"]),
        "Obesity_Yes": _obesity_flag(patient.bmi),
        "Exercise Induced Angina_Yes": int(patient.exercise_induced_angina),
        "Chest Pain Type_Atypical Angina": int(chest_pain == "Atypical Angina"),
        "Chest Pain Type_Non-anginal Pain": int(chest_pain == "Non-anginal Pain"),
        "Chest Pain Type_Typical Angina": int(chest_pain == "Typical Angina"),
    }


_FEATURE_BUILDERS = {
    "hypertension": _build_hypertension_row,
    "stroke": _build_stroke_row,
    "diabetes": _build_diabetes_row,
    "heart": _build_heart_row,
}


# ---------------------------------------------------------------------------
# Prediction + real SHAP explanation
# ---------------------------------------------------------------------------

def _predict_one(disease_key: str, patient) -> Optional[MLDiseasePrediction]:
    artifacts = _load_disease_artifacts(disease_key)
    if artifacts is None:
        return None

    feature_columns = artifacts["feature_columns"]
    builder = _FEATURE_BUILDERS[disease_key]
    raw_row = builder(patient)

    # INFERENCE-PIPELINE INTEGRITY CHECK: the row is always built in the
    # exact trained column order below (`{col: raw_row.get(col, 0) for col
    # in feature_columns}`), so column ORDER can never silently drift. What
    # COULD silently drift is a MISSING feature — if the builder forgot a
    # column, .get(col, 0) would quietly substitute 0.0, which is a real
    # value for some features (e.g. a valid "No" flag) and would never be
    # caught otherwise. Log any mismatch loudly instead of masking it.
    missing_from_builder = [c for c in feature_columns if c not in raw_row]
    unexpected_from_builder = [c for c in raw_row if c not in feature_columns]
    if missing_from_builder:
        print(
            f"[ml_disease_models] WARNING '{disease_key}': builder did not produce "
            f"{missing_from_builder} — defaulting to 0.0 for inference, but this likely means "
            f"the feature-engineering code and the trained model's feature_columns.pkl have drifted "
            f"apart. Verify _build_{disease_key}_row against the training notebook."
        )
    if unexpected_from_builder:
        print(
            f"[ml_disease_models] NOTE '{disease_key}': builder produced extra fields "
            f"{unexpected_from_builder} not present in feature_columns.pkl — these are dropped "
            f"before inference and have no effect, but likely indicate stale feature-builder code."
        )

    # Build the row in the EXACT column order the model was trained on.
    ordered_row = {col: raw_row.get(col, 0) for col in feature_columns}
    X = pd.DataFrame([ordered_row], columns=feature_columns)

    model = artifacts["model"]
    try:
        risk_score = float(model.predict_proba(X)[0][1])
    except Exception as e:
        print(f"[ml_disease_models] predict_proba failed for '{disease_key}': {e}")
        return None

    explainer = artifacts["explainer"]
    try:
        shap_values = explainer.shap_values(X)
        # TreeExplainer returns either a 2D array (n_samples, n_features) for
        # a single positive-class output, or a list [class0, class1] of such
        # arrays for older binary-classifier SHAP APIs. Handle both.
        if isinstance(shap_values, list):
            row_shap = np.array(shap_values[1][0])
        else:
            row_shap = np.array(shap_values[0])
            if row_shap.ndim > 1:  # (n_features, n_classes) case
                row_shap = row_shap[:, 1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.array(base_value).reshape(-1)[-1])
        else:
            base_value = float(base_value)
        shap_contributions = {col: float(v) for col, v in zip(feature_columns, row_shap)}
    except Exception as e:
        print(f"[ml_disease_models] SHAP explanation failed for '{disease_key}': {e}")
        shap_contributions = {}
        base_value = 0.0

    top = sorted(shap_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    if top:
        reasoning = "Primary model drivers: " + ", ".join(
            f"{feat} ({'+' if val > 0 else ''}{val:.2f})" for feat, val in top
        )
    else:
        reasoning = "Model-derived risk score (no SHAP breakdown available)."

    note = MODEL_RELIABILITY_NOTES.get(disease_key)

    return MLDiseasePrediction(
        disease_key=disease_key,
        disease_name=DISEASE_DISPLAY_NAMES[disease_key],
        risk_score=round(risk_score, 4),
        feature_row=ordered_row,
        shap_contributions=shap_contributions,
        base_value=base_value,
        model_metrics=artifacts["metrics"],
        reasoning=reasoning,
        reliability_note=note["message"] if note else None,
        reliability_severity=note["severity"] if note else None,
    )


def predict_all(patient) -> List[MLDiseasePrediction]:
    """
    Runs all four trained models against a PatientProfile. Silently skips
    any disease whose artifacts failed to load or whose prediction raised,
    so a missing/corrupt model file degrades gracefully instead of crashing
    the whole workflow (matching vector_store.py's fail-soft convention).
    """
    predictions = []
    for disease_key in _FEATURE_BUILDERS:
        pred = _predict_one(disease_key, patient)
        if pred is not None:
            predictions.append(pred)
    return predictions
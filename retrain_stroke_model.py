"""
retrain_stroke_model.py

WHY THIS SCRIPT EXISTS:
Your feature mapping/encoding for the stroke model is actually CORRECT —
I verified sex/ever_married/work_type/Residence_type/smoking_status against
real sklearn.LabelEncoder ASCII-sort behavior and your _STROKE_ENCODINGS in
ml_disease_models.py already matches it.

The real problem is model instability. Sweeping avg_glucose_level through
the live model (holding everything else fixed for a 55yo hypertensive
patient) gives:

    glucose=170 mg/dL -> P(stroke)=0.4486
    glucose=180 mg/dL -> P(stroke)=0.4332
    glucose=200 mg/dL -> P(stroke)=0.5672
    glucose=220 mg/dL -> P(stroke)=0.0974   <- risk COLLAPSES
    glucose=280 mg/dL -> P(stroke)=0.0091

That kind of non-monotonic swing from small, clinically meaningless input
changes is a classic symptom of overfitting on a small, heavily
class-imbalanced dataset (the standard Kaggle stroke dataset is ~5%
positive class) with too little regularization. It is NOT a units or
encoding bug in the inference code.

WHAT THIS SCRIPT DOES DIFFERENTLY FROM YOUR ORIGINAL TRAINING:
1. scale_pos_weight (or optionally SMOTE) to properly weight the minority
   (stroke=1) class instead of letting the model mostly learn the majority
   class and overfit noisy detail on the few positives it does see.
2. Shallower trees (max_depth) and a stronger min_child_weight / higher
   gamma than an unconstrained grid search would pick on its own, to
   discourage the model from carving out tiny, unstable decision regions.
3. Monotonic constraints on `age` and `avg_glucose_level` (both must be
   directionally sound clinically: risk should never DECREASE as age or
   glucose increases, holding everything else fixed) — XGBoost supports
   this natively via `monotone_constraints` and it directly targets the
   exact instability observed above.

Everything else (target column, other features, SHAP explainer type,
output file naming) matches your existing pipeline so this remains a
drop-in replacement.

USAGE:
    python retrain_stroke_model.py --data path/to/your_stroke_dataset.csv
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

OUTPUT_DIR = os.environ.get("CLINICAL_ML_MODELS_DIR", "ml_models")

# Column order must exactly match ml_disease_models.py's _build_stroke_row /
# 1786196692795_stroke_feature_columns.pkl.
FEATURE_COLUMNS = [
    "sex", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status",
]

# +1 = risk must be non-decreasing as the feature increases; 0 = unconstrained.
# Targets the exact instability found empirically (glucose sweep flipping
# risk down as glucose rises). Age gets the same treatment since stroke risk
# rising with age is one of the most well-established facts in the literature
# and should never be violated by an overfit split.
MONOTONE_CONSTRAINTS = {
    "sex": 0, "age": 1, "hypertension": 1, "heart_disease": 1, "ever_married": 0,
    "work_type": 0, "Residence_type": 0, "avg_glucose_level": 1, "bmi": 0, "smoking_status": 0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to your original stroke training CSV")
    parser.add_argument("--target-col", default="stroke", help="ADAPT THIS if your target column differs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target_col not in df.columns:
        raise SystemExit(f"Target column '{args.target_col}' not found. Pass --target-col explicitly.")

    y = df[args.target_col]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Expected columns missing from --data: {missing}. Your training CSV must already be "
            f"label-encoded to match ml_disease_models.py's _STROKE_ENCODINGS (this script does not "
            f"re-encode raw string categories — ADAPT THIS section if your CSV still has raw strings)."
        )
    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"Train class balance: {pos} positive / {neg} negative -> scale_pos_weight={scale_pos_weight:.2f}")

    monotone_str = "(" + ",".join(str(MONOTONE_CONSTRAINTS[c]) for c in FEATURE_COLUMNS) + ")"

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,                 # shallower than an unconstrained search would pick
        min_child_weight=10,         # discourages tiny, unstable leaf regions
        gamma=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_str,
        random_state=args.random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred)),
        "ROC_AUC": float(roc_auc_score(y_test, y_proba)),
    }
    print("\nRetrained held-out metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Sanity re-check: the exact glucose sweep that exposed the instability,
    # rerun against the new model — should now be monotonically non-decreasing.
    print("\nMonotonicity sanity check (age=55, hypertension=1, everything else fixed):")
    base = pd.DataFrame([{
        "sex": 1, "age": 55, "hypertension": 1, "heart_disease": 0, "ever_married": 1,
        "work_type": 2, "Residence_type": 1, "bmi": 29, "smoking_status": 2,
        "avg_glucose_level": 100,
    }])[FEATURE_COLUMNS]
    prev = -1.0
    for g in [80, 120, 150, 180, 200, 220, 250, 280]:
        row = base.copy()
        row["avg_glucose_level"] = g
        p = model.predict_proba(row)[0][1]
        direction = "OK" if p >= prev - 1e-9 else "STILL NON-MONOTONIC"
        print(f"  glucose={g:5.0f} -> {p:.4f}  [{direction}]")
        prev = p

    explainer = shap.TreeExplainer(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "stroke_model.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(OUTPUT_DIR, "stroke_feature_columns.pkl"))
    joblib.dump(explainer, os.path.join(OUTPUT_DIR, "stroke_shap_explainer.pkl"))
    with open(os.path.join(OUTPUT_DIR, "stroke_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved retrained artifacts to {OUTPUT_DIR}/stroke_*.pkl")
    print(
        "\nNEXT STEP: once the monotonicity check above reads clean, remove "
        "MODEL_RELIABILITY_NOTES['stroke'] in ml_disease_models.py. Consider also adding a "
        "cholesterol feature to a future retrain if your dataset supports it — the current "
        "stroke model has none, so it can never respond to that clinical signal at all."
    )


if __name__ == "__main__":
    main()

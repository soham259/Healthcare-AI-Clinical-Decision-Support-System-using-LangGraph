"""
retrain_diabetes_model.py

WHY THIS SCRIPT EXISTS:
Testing your shipped diabetes_model.pkl against a textbook-healthy 28-year-old
(BMI 22.8, BP 118/70, fasting sugar 5 mmol/L, no history/symptoms) returns
~20.5% predicted risk — above the clinically expected 5-15% range. Isolating
each feature's marginal effect on a baseline profile found the dominant cause:

    CholCheck=0 -> P(diabetes)=0.0753
    CholCheck=1 -> P(diabetes)=0.1457      <- nearly DOUBLES risk, alone

ml_disease_models.py hardcodes CholCheck=1 for every patient ("we have a
cholesterol reading, so a check happened"). That's a reasonable thing to
assert about our own patients, but the underlying issue is on the MODEL
side: in the source BRFSS survey data, CholCheck ("had cholesterol checked
in the past 5 years") is a well-documented confound — people who get
screened skew toward a population that already has more health-system
contact/pre-existing concern, so the model learned it as a strong predictor
for reasons that are about case-ascertainment bias in the SURVEY, not a
causal risk factor for an individual patient. Since our system effectively
sets this to 1 for every single patient it ever sees, this confound gets
baked into EVERY prediction uniformly, which is very likely the single
biggest contributor to the "too high for a healthy patient" pattern
(other features like Age and GenHlth were checked and behave sensibly:
Age has zero isolated effect on a healthy baseline, GenHlth scales smoothly
0.042 -> 0.139 across its 5 tiers, and a genuine worst-case profile
correctly reaches ~0.89).

WHAT THIS SCRIPT DOES:
Retrains the SAME model type (XGBClassifier) on the SAME dataset, with
ONLY the CholCheck column removed — mirroring exactly how
retrain_hypertension_model.py handles BP_History. Everything else (other
features, SHAP explainer type, output file naming) stays a drop-in
replacement.

If you'd rather KEEP CholCheck (e.g. you have reason to believe your
specific training data doesn't have this confound), pass --keep-cholcheck
and the script will instead just retrain as-is so you have a fresh baseline
to compare metrics against; the recommendation above is a default, not a
requirement.

USAGE:
    python retrain_diabetes_model.py --data path/to/your_diabetes_dataset.csv
"""
import argparse
import json
import os

import joblib
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

OUTPUT_DIR = os.environ.get("CLINICAL_ML_MODELS_DIR", "ml_models")

FULL_FEATURE_COLUMNS = [
    "Age", "Sex", "HighChol", "CholCheck", "BMI", "Smoker", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "GenHlth", "MentHlth",
    "PhysHlth", "DiffWalk", "Stroke", "HighBP",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to your original diabetes training CSV (BRFSS-style)")
    parser.add_argument("--target-col", default="Diabetes_binary", help="ADAPT THIS if your target column differs")
    parser.add_argument("--keep-cholcheck", action="store_true",
                         help="Skip removing CholCheck; just retrain as-is for a fresh metrics baseline")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target_col not in df.columns:
        raise SystemExit(f"Target column '{args.target_col}' not found. Pass --target-col explicitly.")

    y = df[args.target_col]
    feature_columns = list(FULL_FEATURE_COLUMNS)
    if not args.keep_cholcheck and "CholCheck" in feature_columns:
        feature_columns.remove("CholCheck")
        print("Removing CholCheck (recommended — see module docstring for why).")
    else:
        print("Keeping CholCheck (--keep-cholcheck passed, or already absent).")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Expected columns missing from --data: {missing}.")
    X = df[feature_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Same model family/complexity as your existing diabetes_best_params.json
    # (subsample 0.8, n_estimators 100, min_child_weight 5, max_depth 3,
    # learning_rate 0.1, gamma 0.2, colsample_bytree 0.7).
    model = XGBClassifier(
        subsample=0.8, n_estimators=100, min_child_weight=5, max_depth=3,
        learning_rate=0.1, gamma=0.2, colsample_bytree=0.7,
        random_state=args.random_state, eval_metric="logloss", use_label_encoder=False,
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

    # Sanity re-check against the exact healthy-patient profile that exposed
    # the bug, using the new feature set.
    healthy = {
        "Age": 28, "Sex": 0, "HighChol": 0, "BMI": 22.8, "Smoker": 0,
        "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
        "HvyAlcoholConsump": 0, "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0,
        "DiffWalk": 0, "Stroke": 0, "HighBP": 0,
    }
    if "CholCheck" in feature_columns:
        healthy["CholCheck"] = 1
    row = pd.DataFrame([healthy], columns=feature_columns)
    healthy_risk = model.predict_proba(row)[0][1]
    print(f"\nSanity check — healthy 28yo profile -> P(diabetes) = {healthy_risk:.4f} "
          f"(target range: 0.05-0.15)")

    explainer = shap.TreeExplainer(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "diabetes_model.pkl"))
    joblib.dump(feature_columns, os.path.join(OUTPUT_DIR, "diabetes_feature_columns.pkl"))
    joblib.dump(explainer, os.path.join(OUTPUT_DIR, "diabetes_shap_explainer.pkl"))
    with open(os.path.join(OUTPUT_DIR, "diabetes_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved retrained artifacts to {OUTPUT_DIR}/diabetes_*.pkl")
    if "CholCheck" not in feature_columns:
        print(
            "\nNEXT STEP: remove the 'CholCheck' line from _build_diabetes_row in "
            "ml_disease_models.py (it's no longer a feature the model expects), and remove "
            "MODEL_RELIABILITY_NOTES['diabetes'] once the sanity check above looks right."
        )


if __name__ == "__main__":
    main()

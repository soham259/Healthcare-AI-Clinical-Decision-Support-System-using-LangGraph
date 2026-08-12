"""
retrain_hypertension_model.py

WHY THIS SCRIPT EXISTS:
Loading your shipped hypertension_model.pkl and inspecting its real
XGBoost feature importances shows:

    BP_History_Prehypertension    0.2209   <- #1 by gain
    BP_History_Normal             0.1502   <- #2 by gain
    Smoking_Status_Smoker         0.1207
    Stress_Score                  0.1068
    ...

BP_History_Normal / BP_History_Prehypertension are the top two features,
together ~37% of total gain. Looking at your own inference code
(_bp_history_category in ml_disease_models.py), BP_History is computed
directly from the patient's CURRENT systolic/diastolic BP reading using
the same AHA-style thresholds that (almost certainly) define whether the
patient counts as "Hypertension" in your training labels too. That makes
this close to literal label leakage, and fully explains the reported
ROC AUC 0.9999 / Accuracy 0.995 — the model is largely reading back the
category it's supposed to predict.

WHAT THIS SCRIPT DOES:
Retrains the SAME model type (XGBClassifier, same best_params you already
tuned via GridSearch/Optuna) on the SAME dataset, with ONLY the two leaking
one-hot columns removed:
    BP_History_Normal
    BP_History_Prehypertension
Everything else — target, split, other features, hyperparameters, SHAP
explainer type, output file naming — is kept identical to your existing
pipeline so this is a drop-in replacement for the three hypertension
artifacts (model / feature_columns / shap_explainer) and requires ONLY
your original training CSV to run.

If your original training script used a different variable name for the
dataframe/target, or a different train/test split strategy, adjust the
two lines marked "ADAPT THIS" below to match — everything else should work
unchanged.

USAGE:
    python retrain_hypertension_model.py --data path/to/your_hypertension_dataset.csv
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

# Your already-tuned hyperparameters, copied verbatim from
# 1786196692791_hypertension_best_params.json so retraining reproduces the
# same model family/complexity — only the leaking features are removed.
BEST_PARAMS = {
    "subsample": 0.8,
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.1,
    "colsample_bytree": 0.9,
}

# The two leaking columns, removed. Every other column your model already
# used is kept exactly as-is.
LEAKING_COLUMNS = ["BP_History_Normal", "BP_History_Prehypertension"]

OUTPUT_DIR = os.environ.get("CLINICAL_ML_MODELS_DIR", "ml_models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to your original hypertension training CSV")
    parser.add_argument("--target-col", default="Has_Hypertension",
                         help="ADAPT THIS if your target column has a different name")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # ADAPT THIS: reproduce your original preprocessing/one-hot-encoding here
    # if it isn't already baked into the CSV (e.g. BP_History/Family_History/
    # Exercise_Level/Smoking_Status one-hot encoding, dropping the first
    # category). If your CSV is already the exact same fully-engineered
    # feature table minus the target, skip straight to the drop below.
    if args.target_col not in df.columns:
        raise SystemExit(
            f"Target column '{args.target_col}' not found in {args.data}. "
            f"Pass --target-col with the correct name for your dataset."
        )

    y = df[args.target_col]
    X = df.drop(columns=[args.target_col])

    dropped = [c for c in LEAKING_COLUMNS if c in X.columns]
    if not dropped:
        print(
            f"WARNING: none of {LEAKING_COLUMNS} were found in the dataframe. "
            "Either they're already removed, or your column names differ from "
            "ml_models/1786196692792_hypertension_feature_columns.pkl — verify before proceeding."
        )
    X = X.drop(columns=[c for c in LEAKING_COLUMNS if c in X.columns])
    feature_columns = list(X.columns)
    print(f"Training on {len(feature_columns)} features (dropped: {dropped}):")
    print(f"  {feature_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = XGBClassifier(
        **BEST_PARAMS,
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
    print("\nRetrained held-out metrics (should be meaningfully below the old "
          "ROC AUC 0.9999 / Accuracy 0.995 — a large drop is EXPECTED and is the "
          "leakage being removed, not a regression):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    explainer = shap.TreeExplainer(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "hypertension_model.pkl"))
    joblib.dump(feature_columns, os.path.join(OUTPUT_DIR, "hypertension_feature_columns.pkl"))
    joblib.dump(explainer, os.path.join(OUTPUT_DIR, "hypertension_shap_explainer.pkl"))
    with open(os.path.join(OUTPUT_DIR, "hypertension_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved retrained artifacts to {OUTPUT_DIR}/hypertension_*.pkl")
    print(
        "\nNEXT STEP: update ml_disease_models.py's _build_hypertension_row so it no longer "
        "emits BP_History_Normal / BP_History_Prehypertension, and remove the corresponding "
        "MODEL_RELIABILITY_NOTES['hypertension'] entry once you've confirmed the new metrics "
        "look clinically reasonable (a meaningful drop from 0.9999 ROC AUC is expected and correct)."
    )


if __name__ == "__main__":
    main()

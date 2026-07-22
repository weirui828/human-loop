#!/usr/bin/env python3
"""
train_twcs.py

Trains a TF-IDF + Logistic Regression classifier directly on the labeled Twitter Customer Support
(`TWCS`) dataset (`data/twcs/labeled_twcs_20k.csv`) using an 80/20 train/test split.

Outputs:
- Serialized model pipeline: `models/tfidf_twcs.pkl`
- Evaluation metrics: `models/twcs_metrics.json`
"""

import os
import json
import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_20k.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "twcs")
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_twcs.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "twcs_metrics.json")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"TWCS dataset not found at {DATA_PATH}. Please run labeling first.")

    print(f"Loading TWCS dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X = df["customer_text"].fillna("").astype(str)
    y = df["escalated"].astype(int)

    # Stratified Train/Test Split (80/20)
    print("\nSplitting TWCS dataset (80% train, 20% test, stratified by `escalated` class)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train):,d} | Test samples: {len(X_test):,d}")

    # Build TF-IDF + Logistic Regression Pipeline
    print("\nBuilding TF-IDF + Logistic Regression classification pipeline on TWCS data...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # Train model
    print("Fitting model on TWCS training set...")
    start_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = time.perf_counter() - start_train
    print(f"Model training completed in {train_duration:.2f} seconds.")

    # Evaluate on test set
    print("\nEvaluating model on held-out TWCS test set (20%)...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n================== TWCS IN-DOMAIN MODEL RESULTS ==================")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision (M): {prec_macro:.4f}")
    print(f"Recall (M):    {rec_macro:.4f}")
    print(f"Macro F1:      {f1_macro:.4f}")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"                     Pred Self-Service (0)    Pred Escalated (1)")
    print(f"True Self-Service (0)         {cm[0][0]:<22} {cm[0][1]}")
    print(f"True Escalated    (1)         {cm[1][0]:<22} {cm[1][1]}\n")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Self-Service (0)", "Escalated (1)"]))
    print("==================================================================")

    # Save model and metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved TWCS-trained model pipeline to: {MODEL_PATH}")

    metrics_dict = {
        "model_name": "TF-IDF + Logistic Regression (TWCS In-Domain)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_duration_sec": round(train_duration, 4),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved evaluation metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()

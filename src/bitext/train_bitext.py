#!/usr/bin/env python3
"""
train_bitext.py

Baseline ML training pipeline for Customer Support Escalation Classification.
Trains a TF-IDF + Logistic Regression classifier on `cleaned_bitext_training_data.csv`.

Outputs:
- Serialized model pipeline: `models/tfidf_bitext.pkl`
- Evaluation metrics and latency benchmarks: `models/bitext_metrics.json`
"""

import os
import json
import time
import pickle
import pandas as pd
import numpy as np
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

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "bitext", "cleaned_bitext_training_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "bitext")
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_bitext.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "bitext_metrics.json")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Please run `src/bitext/preprocess_bitext.py` first."
        )
    print(f"Loading cleaned dataset from: {path}")
    df = pd.read_csv(path)
    # Ensure text column is string and target is integer
    df["instruction"] = df["instruction"].astype(str).fillna("")
    df["escalated"] = df["escalated"].astype(int)
    return df


def benchmark_latency(model: Pipeline, sample_texts: list, n_iterations: int = 500) -> float:
    """Measures average inference latency in milliseconds (ms) per sample."""
    print(f"\nRunning latency benchmark over {n_iterations} single-instance predictions...")
    # Warmup
    for _ in range(10):
        _ = model.predict([sample_texts[0]])

    start_time = time.perf_counter()
    for i in range(n_iterations):
        idx = i % len(sample_texts)
        _ = model.predict([sample_texts[idx]])
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000.0
    avg_latency_ms = total_time_ms / n_iterations
    print(f"Total time for {n_iterations} inferences: {total_time_ms:.2f} ms")
    print(f"Average inference latency: {avg_latency_ms:.4f} ms/query")
    return avg_latency_ms


def main():
    df = load_data(DATA_PATH)

    # Features and Target
    X = df["instruction"]
    y = df["escalated"]

    # Stratified Train/Test Split (80/20)
    print("Splitting dataset (80% train, 20% test, stratified by `escalated` class)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train):,d} | Test samples: {len(X_test):,d}")

    # Build TF-IDF + Logistic Regression Pipeline
    print("\nBuilding TF-IDF + Logistic Regression classification pipeline...")
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
    print("Fitting model on training set...")
    start_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = time.perf_counter() - start_train
    print(f"Model training completed in {train_duration:.2f} seconds.")

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n================== BASELINE EVALUATION RESULTS ==================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Negatives (TN): {cm[0, 0]:,d} | False Positives (FP): {cm[0, 1]:,d}")
    print(f"False Negatives (FN): {cm[1, 0]:,d} | True Positives (TP):  {cm[1, 1]:,d}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Class 0 (Automated)", "Class 1 (Escalated)"]))
    print("=================================================================")

    # Latency Benchmark
    test_texts_sample = X_test.tolist()[:500]
    avg_latency_ms = benchmark_latency(pipeline, test_texts_sample, n_iterations=500)

    # Save model and metrics
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved Bitext model pipeline to: {MODEL_PATH}")

    metrics_dict = {
        "model_name": "TF-IDF + Logistic Regression (Balanced)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_duration_sec": round(train_duration, 4),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "latency_benchmark": {
            "avg_inference_ms_per_query": round(avg_latency_ms, 4),
            "test_iterations": 500,
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved evaluation metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()

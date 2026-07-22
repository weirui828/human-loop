#!/usr/bin/env python3
"""
evaluate.py

Evaluates the Bitext-trained baseline model (`models/tfidf_bitext.pkl`)
against the out-of-domain Twitter Customer Support (`TWCS`) test set.

Computes the exact Cross-Domain Generalization Drop (in-domain Bitext F1 vs. out-of-domain Twitter F1)
and saves benchmark results to `models/cross_domain_metrics.json`.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bitext", "tfidf_bitext.pkl")
BITEXT_METRICS_PATH = os.path.join(BASE_DIR, "models", "bitext", "bitext_metrics.json")
TWCS_TEST_PATH = os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_20k.csv")
OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, "models", "cross_domain_metrics.json")


def run_evaluation(full: bool = False, input_path: str = None, output_path: str = None):
    if full:
        test_file = input_path or os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_full.csv")
        metrics_file = output_path or os.path.join(BASE_DIR, "models", "cross_domain_metrics_full.json")
    else:
        test_file = input_path or TWCS_TEST_PATH
        metrics_file = output_path or OUTPUT_METRICS_PATH

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run src/bitext/train_bitext.py first.")
        return
        
    if not os.path.exists(test_file):
        print(f"Error: TWCS test set not found at {test_file}. Run src/twcs/label_twcs.py first.")
        return

    print("Loading Bitext baseline model and metrics...")
    model = joblib.load(MODEL_PATH)
    
    with open(BITEXT_METRICS_PATH, "r", encoding="utf-8") as f:
        bitext_metrics = json.load(f)["metrics"]

    print(f"Loading labeled Twitter Customer Support (TWCS) dataset from {test_file}...")
    df_twcs = pd.read_csv(test_file)
    
    X_test_customer = df_twcs["customer_text"].fillna("")
    y_test = df_twcs["escalated"].astype(int)
    
    # 1. Evaluate Cross-Domain Performance (Bitext-trained model on Twitter data)
    print("\n--- Running Cross-Domain Evaluation (Bitext Model -> Twitter Data) ---")
    y_pred = model.predict(X_test_customer)
    y_prob = model.predict_proba(X_test_customer)[:, 1]
    
    twcs_f1 = f1_score(y_test, y_pred, average="macro")
    twcs_prec = precision_score(y_test, y_pred, average="macro")
    twcs_rec = recall_score(y_test, y_pred, average="macro")
    twcs_acc = accuracy_score(y_test, y_pred)
    
    try:
        twcs_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        twcs_auc = float("nan")
        
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate exact generalization drop
    f1_drop = bitext_metrics["f1_score"] - twcs_f1
    auc_drop = bitext_metrics["roc_auc"] - twcs_auc
    
    # 2. In-Domain Twitter Cross-Validation Comparison
    # What if we trained TF-IDF directly on Twitter data?
    print("--- Computing In-Domain Twitter Baseline (Cross-Validation on TWCS) ---")
    twcs_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english", sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight="balanced", random_state=42))
    ])
    # Adjust folds for massive dataset speed vs accuracy
    cv_folds = 3 if len(df_twcs) > 200000 else (5 if len(df_twcs) >= 15 else 3)
    cv_scores = cross_val_score(twcs_pipeline, X_test_customer, y_test, cv=cv_folds, scoring="f1_macro", n_jobs=-1)
    in_domain_twcs_f1 = cv_scores.mean()

    # Print Report Table
    print("\n========================================================================")
    print("                 BENCHMARK: CROSS-DOMAIN GENERALIZATION                 ")
    print("========================================================================")
    print(f"{'Metric':<25} | {'Bitext (In-Domain)':<20} | {'TWCS (Cross-Domain)':<20} | {'Generalization Drop':<20}")
    print("-" * 88)
    print(f"{'Macro F1-Score':<25} | {bitext_metrics['f1_score']:<20.4f} | {twcs_f1:<20.4f} | -{f1_drop:<19.4f}")
    print(f"{'ROC-AUC':<25} | {bitext_metrics['roc_auc']:<20.4f} | {twcs_auc:<20.4f} | -{auc_drop:<19.4f}")
    print(f"{'Accuracy':<25} | {bitext_metrics['accuracy']:<20.4f} | {twcs_acc:<20.4f} | -{(bitext_metrics['accuracy']-twcs_acc):<19.4f}")
    print(f"{'Precision (Macro)':<25} | {bitext_metrics['precision']:<20.4f} | {twcs_prec:<20.4f} | -{(bitext_metrics['precision']-twcs_prec):<19.4f}")
    print(f"{'Recall (Macro)':<25} | {bitext_metrics['recall']:<20.4f} | {twcs_rec:<20.4f} | -{(bitext_metrics['recall']-twcs_rec):<19.4f}")
    print("========================================================================\n")
    
    print(f"In-Domain Twitter Baseline (TF-IDF trained & tested inside TWCS via {cv_folds}-Fold CV):")
    print(f"  -> Macro F1: {in_domain_twcs_f1:.4f}\n")
    
    print("Confusion Matrix on TWCS Test Set (Rows: True, Columns: Predicted):")
    print(f"                     Pred Self-Service (0)    Pred Escalated (1)")
    print(f"True Self-Service (0)         {cm[0][0]:<22} {cm[0][1]}")
    print(f"True Escalated    (1)         {cm[1][0]:<22} {cm[1][1]}\n")
    
    # Save structured results
    results_dict = {
        "bitext_in_domain": bitext_metrics,
        "twcs_cross_domain": {
            "f1_macro": twcs_f1,
            "precision_macro": twcs_prec,
            "recall_macro": twcs_rec,
            "accuracy": twcs_acc,
            "roc_auc": twcs_auc,
            "confusion_matrix": cm.tolist()
        },
        "twcs_in_domain_cv_f1": in_domain_twcs_f1,
        "generalization_drop": {
            "f1_drop": f1_drop,
            "auc_drop": auc_drop,
            "accuracy_drop": bitext_metrics["accuracy"] - twcs_acc
        }
    }
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved cross-domain evaluation metrics to: {metrics_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Bitext-trained baseline model on TWCS test dataset.")
    parser.add_argument("--full", action="store_true", help="Evaluate against full dataset (`labeled_twcs_full.csv`).")
    parser.add_argument("--input", type=str, help="Path to custom labeled CSV evaluation dataset.")
    parser.add_argument("--output", type=str, help="Path to save evaluation metrics JSON.")
    args = parser.parse_args()
    
    run_evaluation(full=args.full, input_path=args.input, output_path=args.output)

#!/usr/bin/env python3
"""
preprocess_bitext.py

Rule-based preprocessing, data cleaning, and feature engineering pipeline for the
Bitext Customer Support dataset.

Performs:
1. Data Cleaning: Missing value handling and duplicate instruction removal.
2. Rule-Based Target Mapping: Maps 27 fine-grained customer service intents to a binary
   escalation target (`escalated` = 1 vs. `escalated` = 0).
3. Feature Engineering:
   - Decomposes multi-character linguistic generation tags (`flags`) into binary indicator columns.
   - Extracts character, word, and entity slot (`{{Entity}}`) counts.
4. Export: Saves the ML-ready dataset to `data/bitext/cleaned_bitext_training_data.csv`.
"""

import os
import re
import pandas as pd
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(
    BASE_DIR, "data", "bitext", "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "bitext", "cleaned_bitext_training_data.csv")

# Rule-based escalation target mapping
# Intents that require human intervention, sensitive handling, or indicate high dissatisfaction/friction
ESCALATION_INTENTS = {
    "complaint",
    "contact_human_agent",
    "contact_customer_service",
    "payment_issue",
    "get_refund",
    "check_cancellation_fee",
    "registration_problems",
}

# Linguistic tag taxonomy covered in data_desc.md
FLAG_TAXONOMY = {
    "B": "basic_syntax",
    "I": "interrogative",
    "C": "coordinated_syntax",
    "N": "negation",
    "M": "morphological_var",
    "L": "semantic_var",
    "P": "polite",
    "Q": "colloquial",
    "W": "offensive",
    "K": "keyword_mode",
    "E": "abbreviations",
    "Z": "typos_errors",
}


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads the raw Bitext CSV, removes nulls and duplicates."""
    print(f"Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Initial raw shape: {df.shape}")

    # Check missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
        df = df.dropna(subset=["instruction", "response", "category", "intent"]).copy()
    else:
        print("No missing values found across core columns.")

    # Remove duplicate instruction strings to prevent data leakage between train/test splits
    initial_rows = len(df)
    df = df.drop_duplicates(subset=["instruction"], keep="first").copy()
    print(f"Removed {initial_rows - len(df)} duplicate instructions. Cleaned shape: {df.shape}")

    return df


def map_escalation_target(df: pd.DataFrame) -> pd.DataFrame:
    """Maps the 27 fine-grained intents to binary escalation target (0 or 1)."""
    df["escalated"] = df["intent"].apply(lambda x: 1 if x in ESCALATION_INTENTS else 0)

    counts = df["escalated"].value_counts()
    props = df["escalated"].value_counts(normalize=True) * 100
    print("\n--- Binary Target (`escalated`) Distribution ---")
    print(f"Class 0 (Self-Service / Standard Query): {counts.get(0, 0):,d} ({props.get(0, 0):.2f}%)")
    print(f"Class 1 (Requires Human Agent Escalation): {counts.get(1, 0):,d} ({props.get(1, 0):.2f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts linguistic flag indicators and structural text features."""
    print("\nEngineering linguistic flag features from `flags` column...")
    flags_series = df["flags"].fillna("")

    # Create binary indicator column for each linguistic phenomenon
    for flag_char, flag_name in FLAG_TAXONOMY.items():
        col_name = f"flag_{flag_name}"
        df[col_name] = flags_series.apply(lambda s: int(flag_char in str(s)))

    print("Engineering structural length and entity slot features...")
    # Text lengths
    df["char_len_instruction"] = df["instruction"].astype(str).str.len()
    df["word_len_instruction"] = df["instruction"].astype(str).str.split().str.len()
    df["char_len_response"] = df["response"].astype(str).str.len()
    df["word_len_response"] = df["response"].astype(str).str.split().str.len()

    # Entity slot counts using regex matching {{Slot Name}}
    entity_pattern = re.compile(r"\{\{[^}]+\}\}")
    df["entity_count_instruction"] = df["instruction"].astype(str).apply(lambda s: len(entity_pattern.findall(s)))
    df["entity_count_response"] = df["response"].astype(str).apply(lambda s: len(entity_pattern.findall(s)))

    return df


def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")

    df = load_and_clean_data(RAW_DATA_PATH)
    df = map_escalation_target(df)
    df = engineer_features(df)

    print(f"\nFinal engineered shape: {df.shape}")
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Successfully saved cleaned & engineered dataset to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()

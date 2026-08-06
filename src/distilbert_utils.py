#!/usr/bin/env python3
"""DistilBERT fine-tuning for notebooks 03-05.

Plain PyTorch loop, not `transformers.Trainer`. Loss is class-weighted to match the
`class_weight="balanced"` used by the TF-IDF baselines.
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_CHECKPOINT = "distilbert-base-uncased"
LABEL_NAMES = ["Self-Service (0)", "Escalated (1)"]

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def build_loader(texts, labels, tokenizer, batch_size, max_length, shuffle):
    """DataLoader over (text, label) pairs, tokenizing per-batch to enable dynamic padding.
    A simple list of tuples is used in place of a full Dataset subclass.
    """

    def collate(batch):
        batch_texts, batch_labels = zip(*batch)
        encoded = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        return encoded

    pairs = [(str(t), int(y)) for t, y in zip(texts, labels)]
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def balanced_class_weights(labels, device: torch.device) -> torch.Tensor:
    """Mirrors sklearn's `class_weight="balanced"`: n_samples / (n_classes * n_class_c)."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = len(labels) / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float, device=device)


def fine_tune(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    *,
    checkpoint: str = MODEL_CHECKPOINT,
    device=None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
):
    """Fine-tunes DistilBERT and returns `(model, tokenizer, history)`.
    Returns the model from the epoch with the best validation macro-F1.
    `checkpoint` allows swapping base models.
    """
    device = device or torch.device("mps")  # change this based on your device type
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # pretrained encoder + fresh 2-class head
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    ).to(device)

    # DataLoaders feed data into the network in small batches to avoid crashing GPU memory.
    # train_loader is shuffled so the model doesn't memorize the sequence of the dataset.
    train_loader = build_loader(
        train_texts, train_labels, tokenizer, batch_size, max_length, shuffle=True
    )
    # val_loader isn't shuffled since order doesn't matter for evaluation.
    # Validation has no backward pass (uses less memory), so we double the batch size to speed it up.
    val_loader = build_loader(
        val_texts, val_labels, tokenizer, batch_size * 2, max_length, shuffle=False
    )

    # linear warmup then linear decay over the whole run
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy="linear",
    )
    loss_fn = nn.CrossEntropyLoss(weight=balanced_class_weights(train_labels, device))

    if verbose:
        print(
            f"Device: {device.type.upper()} | epochs: {epochs} | batch: {batch_size} | "
            f"lr: {learning_rate:g} | steps/epoch: {len(train_loader)}"
        )

    history = []
    best_f1, best_state, best_epoch = -1.0, None, -1
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, epoch_start = 0.0, time.perf_counter()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guard against exploding grads
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        torch.mps.synchronize()  # change this based on your device type
        # score the held-out slice; epoch selection reads this, never the test set
        train_loss = running_loss / len(train_loader)
        val_probs = _predict_loader(model, val_loader, device)
        val_pred = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_pred, average="macro")
        elapsed = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_macro_f1": round(float(val_f1), 4),
                "epoch_sec": round(elapsed, 1),
            }
        )
        if verbose:
            print(
                f"  epoch {epoch}/{epochs} | train loss {train_loss:.4f} | "
                f"val macro-F1 {val_f1:.4f} | {elapsed:.0f}s"
            )

        # keep a CPU copy of the best epoch so later epochs can't degrade the result
        if val_f1 > best_f1:
            best_f1, best_epoch = float(val_f1), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    total_train_sec = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    if verbose:
        print(
            f"Done in {total_train_sec:.0f}s. Restored epoch {best_epoch} "
            f"(best val macro-F1 {best_f1:.4f})."
        )

    meta = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_f1, 4),
        "train_duration_sec": round(total_train_sec, 1),
        "hyperparameters": {
            "checkpoint": checkpoint,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "class_weighted_loss": True,
            "seed": seed,
        },
    }
    return model, tokenizer, meta


# --------------------------------------------------------------------------------------
# Inference & evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _predict_loader(model, loader, device) -> np.ndarray:
    """Returns P(escalated) for every row behind `loader`."""
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels", None)
        logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def predict_proba(model, tokenizer, texts, device=None, batch_size: int = 64, max_length: int = 128):
    """Batch inference over raw strings -> P(escalated) array."""
    device = device or torch.device("mps")  # change this based on your device type
    loader = build_loader(
        texts, np.zeros(len(texts), dtype=int), tokenizer, batch_size, max_length, shuffle=False
    )
    return _predict_loader(model, loader, device)


def binary_metrics(y_true, y_pred, y_prob) -> dict:
    """Computes the metric set the TF-IDF baselines report, both macro and positive-class."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro")), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "precision_escalated": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_escalated": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_escalated": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def benchmark_latency(
    model, tokenizer, texts, device=None, n_iterations: int = 500, max_length: int = 128
) -> float:
    """Average single-query inference latency in ms.
    """
    device = device or torch.device("mps")  # change this based on your device type
    model.eval()
    texts = [str(t) for t in texts]

    with torch.no_grad():
        for _ in range(10):  # warmup: first MPS/CUDA kernels are not representative
            enc = tokenizer(
                texts[0], truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            _ = model(**enc)
        torch.mps.synchronize()  # change this based on your device type

        start = time.perf_counter()
        for i in range(n_iterations):
            enc = tokenizer(
                texts[i % len(texts)], truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            _ = model(**enc)
        torch.mps.synchronize()  # change this based on your device type
        total_ms = (time.perf_counter() - start) * 1000.0

    return total_ms / n_iterations


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def save_model(model, tokenizer, output_dir: str) -> str:
    """Writes the fine-tuned weights + tokenizer to `output_dir` (HF format)."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def load_model(model_dir: str, device=None):
    """Reloads a saved fine-tuned model onto the best available device."""
    device = device or torch.device("mps")  # change this based on your device type
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def save_metrics(metrics: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path

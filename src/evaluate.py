"""Evaluate a saved model on the held-out test set.

    python -m src.evaluate --model-path outputs/mobilenet.keras
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from tensorflow import keras

from src.data import CLASS_NAMES, load_splits


def predict(model, ds):
    y_true, y_prob = [], []
    for images, labels in ds:
        y_prob.append(model.predict_on_batch(images).ravel())
        y_true.append(labels.numpy().ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


def evaluate(model, test_ds, out_dir, name, threshold=0.5):
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_prob = predict(model, test_ds)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": name,
        "test_images": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity_recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "missed_infected_cells": int(fn),
        "false_alarms": int(fp),
    }
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                           display_labels=[c.capitalize() for c in CLASS_NAMES]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{name}: test set confusion matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(f"{name}: ROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}_roc.png", dpi=150)
        plt.close(fig)

    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", default="tfds", help='"tfds" or a folder with Parasitized/ and Uninfected/')
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    a = p.parse_args()

    _, _, test, _ = load_splits(a.data, a.img_size, a.batch_size, a.seed)
    model = keras.models.load_model(a.model_path)
    m = evaluate(model, test, a.out_dir, Path(a.model_path).stem, a.threshold)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()

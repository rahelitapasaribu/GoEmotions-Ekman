
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    hamming_loss,
)

from src.config import LABELS


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Mengumpulkan label dan probabilitas dari dataloader."""
    model.eval()

    all_true = []
    all_probs = []

    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )

        probs = torch.sigmoid(outputs.logits)

        all_true.append(
            batch["labels"].cpu().numpy()
        )
        all_probs.append(
            probs.cpu().numpy()
        )

    return (
        np.vstack(all_true).astype(int),
        np.vstack(all_probs),
    )


def evaluate_predictions(y_true, y_probs, threshold=0.5):
    """Menghitung metrik multi-label dan hasil per emosi."""

    y_pred = (y_probs >= threshold).astype(int)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
        )
    )

    per_label = pd.DataFrame({
        "label": LABELS,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    overall = {
        "micro_f1": f1_score(
            y_true, y_pred,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }

    return overall, per_label


def compare_thresholds(y_true, y_probs):
    """Perbandingan threshold pada validation saja."""

    results = []

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        overall, _ = evaluate_predictions(
            y_true,
            y_probs,
            threshold=threshold,
        )

        results.append({
            "threshold": threshold,
            **overall,
        })

    return pd.DataFrame(results)

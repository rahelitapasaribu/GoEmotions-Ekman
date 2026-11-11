
import numpy as np
import pandas as pd

from src.config import LABELS


def analyze_errors(df, y_true, y_probs, threshold):
    """Analisis kesalahan dengan keputusan model final."""

    if len(df) != len(y_true) or len(df) != len(y_probs):
        raise ValueError(
            "Jumlah teks dan prediksi tidak sesuai."
        )

    y_pred = (y_probs >= threshold).astype(int)

    records = []

    for i in range(len(df)):
        true_indices = np.where(y_true[i] == 1)[0]
        pred_indices = np.where(y_pred[i] == 1)[0]

        true_labels = [LABELS[j] for j in true_indices]
        pred_labels = [LABELS[j] for j in pred_indices]

        missed = sorted(set(true_labels) - set(pred_labels))
        extra = sorted(set(pred_labels) - set(true_labels))

        records.append({
            "id": (
                df.iloc[i]["id"]
                if "id" in df.columns else i
            ),
            "text": df.iloc[i]["text"],
            "true_labels": ", ".join(true_labels),
            "predicted_labels": ", ".join(pred_labels),
            "false_negative_labels": ", ".join(missed),
            "false_positive_labels": ", ".join(extra),
            "exact_match": int(
                np.array_equal(y_true[i], y_pred[i])
            ),
        })

    return pd.DataFrame(records)


def summarize_errors(errors):
    """Ringkasan ketepatan seluruh set label per teks."""

    total = len(errors)
    incorrect = int((errors["exact_match"] == 0).sum())

    return {
        "total_samples": total,
        "exact_match_samples": total - incorrect,
        "incorrect_samples": incorrect,
        "exact_match_ratio": (total - incorrect) / total,
    }

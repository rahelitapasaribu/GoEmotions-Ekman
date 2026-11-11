
import torch
from transformers import AutoModelForSequenceClassification

from src.config import (
    MODEL_NAME,
    NUM_LABELS,
    LABEL2ID,
    ID2LABEL,
)


def create_model():
    """Membuat model DistilBERT untuk multi-label emotion classification."""

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    return model


def get_device():
    """Menggunakan GPU jika tersedia."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

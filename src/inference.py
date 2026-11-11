
import json
import torch

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.config import OUTPUT_DIR, LABELS
from src.model import get_device


class EmotionPredictor:
    """Inference model final GoEmotions-Ekman."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = OUTPUT_DIR / "final_config.json"

        config_path = Path(config_path)

        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.device = get_device()
        self.threshold = float(self.config["threshold"])
        self.max_length = int(self.config["max_length"])

        # Menggunakan label yang sudah dibekukan
        if self.config["labels"] != LABELS:
            raise ValueError("Urutan label tidak konsisten.")

        checkpoint = Path(self.config["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = (
                Path(__file__).resolve().parent.parent / checkpoint
            ).resolve()

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: {checkpoint}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(checkpoint)
            .to(self.device)
        )

        self.model.eval()

    @torch.inference_mode()
    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Teks tidak boleh kosong.")

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        probabilities = torch.sigmoid(
            outputs.logits
        )[0].cpu().tolist()

        scores = {
            label: float(prob)
            for label, prob in zip(LABELS, probabilities)
        }

        predicted_labels = [
            label
            for label, prob in scores.items()
            if prob >= self.threshold
        ]

        return {
            "text": text,
            "predicted_labels": predicted_labels,
            "probabilities": scores,
            "threshold": self.threshold,
        }

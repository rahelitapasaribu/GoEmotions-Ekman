
import json
import random

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from torch.optim import AdamW

from src.config import (
    SEED,
    DEFAULT_THRESHOLD,
    OUTPUT_DIR,
    MODEL_NAME,
)
from src.model import create_model, get_device


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = inputs["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Validation"):
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**inputs)

        batch_size = inputs["labels"].size(0)

        total_loss += outputs.loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(outputs.logits)

        all_probs.append(probs.cpu().numpy())
        all_true.append(
            inputs["labels"].cpu().numpy()
        )

    y_true = np.vstack(all_true).astype(int)
    y_probs = np.vstack(all_probs)

    # Fixed threshold for fair checkpoint monitoring.
    # Final threshold tuning is performed later.
    y_pred = (
        y_probs >= DEFAULT_THRESHOLD
    ).astype(int)

    return {
        "val_loss": total_loss / total_samples,
        "val_micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "val_macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def run_training(
    train_loader,
    val_loader,
    tokenizer,
    experiment_name,
    config,
):
    set_seed(SEED)

    device = get_device()

    print("Experiment:", experiment_name)
    print("Device:", device)
    print("Configuration:", config)

    # Fresh pretrained model for every experiment
    model = create_model().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
    )

    checkpoint_dir = (
        OUTPUT_DIR / "checkpoints" / experiment_name
    )

    metrics_dir = OUTPUT_DIR / "metrics"

    # Prevent accidentally overwriting saved runs
    history_path = (
        metrics_dir / f"{experiment_name}_history.csv"
    )

    if checkpoint_dir.exists() or history_path.exists():
        raise FileExistsError(
            f"Hasil {experiment_name} sudah ada. "
            "Gunakan nama eksperimen baru atau "
            "pindahkan hasil lama terlebih dahulu."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Record exact experiment settings
    metadata = {
        "experiment": experiment_name,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "checkpoint_selection": "val_macro_f1",
        "monitor_threshold": DEFAULT_THRESHOLD,
        **config,
    }

    with open(
        checkpoint_dir / "experiment.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    best_macro_f1 = -1.0
    best_epoch = None
    history = []

    for epoch in range(1, config["epochs"] + 1):
        print(
            f"\n=== {experiment_name} "
            f"EPOCH {epoch}/{config['epochs']} ==="
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_result = validate(
            model,
            val_loader,
            device,
        )

        row = {
            "experiment": experiment_name,
            "epoch": epoch,
            "train_loss": train_loss,
            **val_result,
        }

        history.append(row)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_result['val_loss']:.4f}")
        print(
            f"Val Micro-F1: "
            f"{val_result['val_micro_f1']:.4f}"
        )
        print(
            f"Val Macro-F1: "
            f"{val_result['val_macro_f1']:.4f}"
        )

        if val_result["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = val_result["val_macro_f1"]
            best_epoch = epoch

            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            print("Best checkpoint updated.")

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

    summary = {
        **metadata,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "history_path": str(history_path),
        "checkpoint_path": str(checkpoint_dir),
    }

    with open(
        metrics_dir / f"{experiment_name}_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n=== EXPERIMENT COMPLETED ===")
    print("Best epoch:", best_epoch)
    print("Best validation Macro-F1:", best_macro_f1)
    print("Checkpoint:", checkpoint_dir)

    return pd.DataFrame(history)

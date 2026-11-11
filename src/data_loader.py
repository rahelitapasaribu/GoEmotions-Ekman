
import pandas as pd

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    LABELS,
    MIN_POSITIVE_PER_LABEL,
)


def load_split(path):
    """Memuat dan memvalidasi satu split dataset."""

    df = pd.read_csv(path)

    required_columns = ["text"] + LABELS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak ditemukan pada {path}: "
            f"{missing_columns}"
        )

    # Validasi teks
    if df["text"].isna().any():
        raise ValueError(f"Ada missing text: {path}")

    df["text"] = df["text"].astype(str)

    if df["text"].str.strip().eq("").any():
        raise ValueError(f"Ada empty text: {path}")

    # Pastikan tujuh label numerik
    for label in LABELS:
        df[label] = pd.to_numeric(
            df[label],
            errors="raise"
        )

    # Nilai label wajib 0 atau 1
    if not df[LABELS].isin([0, 1]).all().all():
        raise ValueError(
            f"Terdapat nilai label selain 0/1: {path}"
        )

    # Setiap teks minimal memiliki satu label aktif
    active_counts = df[LABELS].sum(axis=1)

    if (active_counts == 0).any():
        raise ValueError(
            f"Terdapat teks tanpa label aktif: {path}"
        )

    return df.reset_index(drop=True)


def normalize_text_key(series):
    """Normalisasi teks hanya untuk pemeriksaan duplikat."""
    return (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )


def validate_dataset(train_df, val_df, test_df):
    """Pemeriksaan dasar dataset final."""

    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    keys = {}

    for name, df in splits.items():
        normalized = normalize_text_key(df["text"])

        if normalized.duplicated().any():
            raise ValueError(
                f"Duplikasi internal ditemukan: {name}"
            )

        keys[name] = set(normalized)

    # Periksa duplikasi lintas split
    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for split_a, split_b in comparisons:
        overlap = keys[split_a] & keys[split_b]

        if overlap:
            raise ValueError(
                f"Duplikasi lintas split: "
                f"{split_a} vs {split_b} = {len(overlap)}"
            )

    # Pemeriksaan ketentuan minimal 500
    all_labels = pd.concat(
        [
            train_df[LABELS],
            val_df[LABELS],
            test_df[LABELS],
        ],
        ignore_index=True,
    )

    counts = all_labels.sum().astype(int)

    if (counts < MIN_POSITIVE_PER_LABEL).any():
        insufficient = counts[
            counts < MIN_POSITIVE_PER_LABEL
        ]
        raise ValueError(
            "Ada label yang belum memenuhi minimum: "
            f"{insufficient.to_dict()}"
        )

    return counts


def load_datasets():
    """Memuat tiga split dan memvalidasi dataset final."""

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    label_counts = validate_dataset(
        train_df, val_df, test_df
    )

    print("=== DATASET FINAL ===")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("\n=== POSITIVE COUNT PER LABEL ===")
    print(label_counts.to_string())

    print("\nValidasi dataset berhasil.")

    return train_df, val_df, test_df

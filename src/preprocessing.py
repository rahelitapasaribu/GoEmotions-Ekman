
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    LABELS,
    SEED,
)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class EmotionDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=MAX_LENGTH,
    ):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe[LABELS].to_numpy(
            dtype="float32"
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding[
                "input_ids"
            ].squeeze(0),

            "attention_mask": encoding[
                "attention_mask"
            ].squeeze(0),

            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.float32,
            ),
        }


def create_dataloader(
    dataframe,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
    shuffle=False,
):
    dataset = EmotionDataset(
        dataframe,
        tokenizer,
        max_length=max_length,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def create_dataloaders(
    train_df,
    val_df,
    test_df=None,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
):
    tokenizer = get_tokenizer()

    train_loader = create_dataloader(
        train_df,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )

    val_loader = create_dataloader(
        val_df,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    test_loader = None

    if test_df is not None:
        test_loader = create_dataloader(
            test_df,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
        )

    return (
        tokenizer,
        train_loader,
        val_loader,
        test_loader,
    )

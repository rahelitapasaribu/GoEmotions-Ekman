
import argparse

from src.config import EXPERIMENTS
from src.data_loader import load_datasets
from src.preprocessing import create_dataloaders
from src.train import run_training


def main():
    parser = argparse.ArgumentParser(
        description="MLFT GoEmotions fine-tuning"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENTS.keys()),
    )

    args = parser.parse_args()

    experiment_name = args.experiment
    experiment_config = EXPERIMENTS[experiment_name]

    # Load validated dataset
    train_df, val_df, _ = load_datasets()

    # Only train and validation are prepared for training.
    tokenizer, train_loader, val_loader, _ = (
        create_dataloaders(
            train_df,
            val_df,
            test_df=None,
            batch_size=experiment_config["batch_size"],
            max_length=experiment_config["max_length"],
        )
    )

    history = run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        experiment_name=experiment_name,
        config=experiment_config,
    )

    print("\n=== TRAINING HISTORY ===")
    print(history.to_string(index=False))


if __name__ == "__main__":
    main()

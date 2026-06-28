"""Dataset loading and prompt formatting for Text2Cypher fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset

from utils import DEFAULT_DATASET_NAME, build_prompt


REQUIRED_SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("question", "schema", "cypher")


def _format_training_example(example: dict) -> dict[str, str]:
    return {
        "text": build_prompt(
            schema=example["schema"],
            question=example["question"],
            cypher=example["cypher"],
        )
    }


def  load_text2cypher_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    cache_dir: str | Path | None = None,
) -> DatasetDict:
    """Load, validate, and format all splits from the curated dataset."""
    dataset = load_dataset(
        dataset_name,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    missing_splits = [
        split for split in REQUIRED_SPLITS 
        if split not in dataset
    ]
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")

    formatted = DatasetDict()
    for split in REQUIRED_SPLITS:
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in dataset[split].column_names
        ]
        if missing_columns:
            raise ValueError(
                f"Dataset split '{split}' is missing columns: {missing_columns}"
            )

        selected = dataset[split].select_columns(list(REQUIRED_COLUMNS))
        formatted[split] = selected.map(_format_training_example)

    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the formatted dataset.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_text2cypher_dataset(args.dataset_name, args.cache_dir)
    print(dataset)
    print("\nFirst formatted training example:\n")
    print(dataset["train"][0]["text"])


if __name__ == "__main__":
    main()

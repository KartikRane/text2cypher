"""Generate test predictions and evaluate a fine-tuned Text2Cypher model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import load_text2cypher_dataset
from utils import (
    DEFAULT_DATASET_NAME,
    batched,
    build_prompt,
    load_model_for_evaluation,
    load_tokenizer,
    token_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("final_model"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-file", type=Path, default=Path("results/predictions.json"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def structural_validity(predicted: str) -> int:
    """Return 1 when a prediction has the minimum Cypher query structure."""
    normalized = predicted.strip().upper()
    return int(normalized.startswith("MATCH") and "RETURN" in normalized)


def generate_predictions(
    model,
    tokenizer,
    examples,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> list[dict]:
    """Generate one Cypher continuation and metrics for every example."""
    results: list[dict] = []
    tokenizer.padding_side = "left"
    model.eval()

    for batch in batched(examples, batch_size):
        prompts = [
            build_prompt(schema=item["schema"], question=item["question"])
            for item in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_width = encoded["input_ids"].shape[1]
        continuations = generated[:, prompt_width:]
        predictions = tokenizer.batch_decode(
            continuations,
            skip_special_tokens=True,
        )

        for item, prediction in zip(batch, predictions, strict=True):
            predicted = prediction.strip()
            ground_truth = item["cypher"].strip()
            results.append(
                {
                    "question": item["question"],
                    "schema": item["schema"],
                    "ground_truth": item["cypher"],
                    "predicted": predicted,
                    "exact_match": int(predicted == ground_truth),
                    "token_f1": token_f1(predicted, ground_truth),
                    "structural_validity": structural_validity(predicted),
                }
            )

    return results


def main() -> None:
    args = parse_args()
    dataset = load_text2cypher_dataset(args.dataset_name, args.cache_dir)
    tokenizer = load_tokenizer(args.model_dir)
    model = load_model_for_evaluation(args.model_dir)

    results = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        examples=dataset["test"],
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2, ensure_ascii=False)

    exact_match_accuracy = sum(item["exact_match"] for item in results) / len(results)
    average_token_f1 = sum(item["token_f1"] for item in results) / len(results)
    average_structural_validity = (
        sum(item["structural_validity"] for item in results) / len(results)
    )
    print(f"Evaluated {len(results)} test examples")
    print(f"Overall Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"Average Token F1: {average_token_f1:.4f}")
    print(f"Structural Validity: {average_structural_validity:.4f}")
    print(f"Predictions saved to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()

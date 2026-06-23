"""Shared prompt, model-loading, and evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_DATASET_NAME = "RomanTeucher/text2cypher-curated"


def build_prompt(schema: str, question: str, cypher: str | None = None) -> str:
    """Build the canonical prompt used for both training and inference."""
    prompt = f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### Cypher:\n"
    if cypher is not None:
        prompt += cypher
    return prompt


def load_tokenizer(model_name_or_path: str | Path) -> PreTrainedTokenizerBase:
    """Load a tokenizer and configure padding for causal language modeling."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_for_evaluation(model_name_or_path: str | Path):
    """Load either a saved LoRA adapter or a base causal language model on CPU."""
    model_name_or_path = str(model_name_or_path)
    local_path = Path(model_name_or_path)

    if local_path.exists() and (local_path / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map=None,
        )
        return PeftModel.from_pretrained(base_model, model_name_or_path)

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=None,
    )


def token_f1(predicted: str, ground_truth: str) -> float:
    """Compute set-based F1 over whitespace-delimited query tokens."""
    predicted_tokens = set(predicted.split())
    ground_truth_tokens = set(ground_truth.split())

    if not predicted_tokens and not ground_truth_tokens:
        return 1.0
    if not predicted_tokens or not ground_truth_tokens:
        return 0.0

    overlap = len(predicted_tokens & ground_truth_tokens)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def batched(items: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    """Yield dictionaries in fixed-size batches without loading extra copies."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    batch: list[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

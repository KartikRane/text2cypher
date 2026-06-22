"""Fine-tune SmolLM2 for Text2Cypher generation with LoRA on CPU."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

# Keep CPU usage predictable on shared machines, as required by the task.
torch.set_num_threads(4)

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, set_seed  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from dataset import load_text2cypher_dataset  # noqa: E402
from utils import DEFAULT_DATASET_NAME, DEFAULT_MODEL_NAME, load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("final_model"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Use only the first N training examples (default: use all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="Hub repository ID. Required when --push-to-hub is set.",
    )
    return parser.parse_args()


def create_sft_config(args: argparse.Namespace) -> SFTConfig:
    """Create an SFTConfig while tolerating renamed TRL config fields."""
    supported = inspect.signature(SFTConfig.__init__).parameters
    config = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 2e-4,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": False,
        "fp16": False,
        "bf16": False,
        "packing": False,
        "dataset_text_field": "text",
        "use_cpu": True,
        "hub_model_id": args.hub_model_id,
    }

    # Transformers renamed evaluation_strategy to eval_strategy, while TRL
    # renamed max_seq_length to max_length. Supporting both keeps the script
    # usable across the compatible releases allowed in requirements.txt.
    if "eval_strategy" in supported:
        config["eval_strategy"] = "epoch"
    else:
        config["evaluation_strategy"] = "epoch"

    if "max_length" in supported:
        config["max_length"] = args.max_seq_length
    else:
        config["max_seq_length"] = args.max_seq_length

    return SFTConfig(**{key: value for key, value in config.items() if key in supported})


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_config: SFTConfig,
) -> SFTTrainer:
    """Construct SFTTrainer across TRL's tokenizer API rename."""
    supported = inspect.signature(SFTTrainer.__init__).parameters
    trainer_args = {
        "model": model,
        "args": training_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if "processing_class" in supported:
        trainer_args["processing_class"] = tokenizer
    else:
        trainer_args["tokenizer"] = tokenizer
    return SFTTrainer(**trainer_args)


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub-model-id is required with --push-to-hub")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        raise ValueError("--max-train-samples must be at least 1")

    set_seed(args.seed)
    dataset = load_text2cypher_dataset(args.dataset_name, args.cache_dir)
    if args.max_train_samples is not None:
        sample_count = min(args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(sample_count))
    tokenizer = load_tokenizer(args.model_name)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=None)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_config = create_sft_config(args)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        training_config=training_config,
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub()
        tokenizer.push_to_hub(args.hub_model_id)

    print(f"Final LoRA adapter and tokenizer saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

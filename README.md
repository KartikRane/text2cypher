# Text2Cypher Fine-tuning

Fine-tunes `HuggingFaceTB/SmolLM2-135M-Instruct` with LoRA to translate a
natural-language question and graph schema into a Cypher query. Training and
inference are configured for CPU-only execution.

## Model

Fine-tuned adapter available on HuggingFace Hub:
https://huggingface.co/kv-rane/text2cypher-smollm2

## Setup

Python 3.11 is recommended (3.14 is not yet supported by the ML dependencies).

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The model and dataset are downloaded from Hugging Face on first use.

## Reproduce

Run both commands from the repository root:

```bash
python src/train.py
python src/evaluate.py
```

Training uses three epochs, a batch size of four, a learning rate of `2e-4`,
and validation at the end of each epoch. The final LoRA adapter and tokenizer
are saved under `./final_model/`. Evaluation runs deterministic greedy decoding
over all 50 test examples and overwrites `results/predictions.json`.

For a quick test run on reduced data (~30 minutes on CPU):

```bash
python src/train.py --max-train-samples 200
```

Paths and generation limits can be changed through command-line arguments:

```bash
python src/train.py --output-dir final_model --cache-dir .cache/huggingface
python src/evaluate.py --model-dir final_model --output-file results/predictions.json
```

To publish the trained adapter after training:

```bash
huggingface-cli login
python src/train.py --push-to-hub --hub-model-id kv-rane/text2cypher-smollm2
```

## Results

Evaluated on 50 test examples:

| Metric | Score |
|--------|-------|
| Exact Match | 0.00% |
| Token F1 | 27.4% |
| Structural Validity | 74.0% |

Token F1 improved 64% when scaling from 200 to 1,000 training examples
(16.7% → 27.4%), confirming genuine learning from more data.

## Design Decisions & Limitations

### Why I chose LoRA

LoRA freezes the 135M-parameter base model and adds trainable low-rank updates
only to its query and value projection layers. With rank 16, this trains roughly
1% of the parameters, reducing optimizer memory and backward-pass cost on CPU.
It also produces a small adapter artifact while preserving the reusable base
model. Full fine-tuning could offer more capacity, but its CPU cost is not
justified by a 1,000-example training set and would increase overfitting risk.

I set the model to process at most 512 tokens per training example as a
CPU cost/coverage tradeoff. Raising the cap would
preserve more large schemas but substantially increase attention cost;
schema-aware truncation would be a better follow-up for this task.

### Why these metrics

- **Exact Match** — strict comparison after stripping whitespace. Simple and
  interpretable, but penalizes semantically equivalent queries that differ in
  aliases, clause ordering, or formatting.
- **Token F1** — set-based overlap over whitespace-delimited tokens. Gives
  partial credit when the model recovers relevant labels, relationships,
  properties, and clauses, but ignores token order and can reward a query
  that is not executable or semantically correct.
- **Structural Validity** — checks whether the predicted query starts with
  MATCH and contains RETURN. A lightweight proxy for syntactic correctness
  that requires no external infrastructure.

I intentionally excluded execution accuracy because it requires a populated,
versioned Neo4j test environment. Reporting it without that infrastructure
would produce irreproducible results.


### What the model learned

I first trained on 200 samples for quick pipeline validation (~30 minutes on
CPU), then on the full 1,000 samples (~2.5 hours on CPU).

Training on 1,000 examples reduced loss from 2.54 to 0.76 across 3 epochs.
Validation loss tracked closely at 0.87, indicating no significant overfitting.
Token F1 improved from 16.7% (200 samples) to 27.4% (1,000 samples) — a 64%
relative improvement — confirming the model benefits from more data.

Inspection of predictions.json reveals the model learned:
- Basic MATCH clause structure with correct node labels from the schema
- Simple WHERE filters for equality and comparison conditions
- RETURN clauses with correct property names on simple schemas

### Where it fails

Three failure modes appear consistently in predictions.json:

1. **Repetition loops** — the model generates a valid opening clause then
   repeats the same token indefinitely until max_new_tokens is reached.
   This suggests the model has not learned a reliable end-of-query signal.

2. **Schema copying on complex inputs** — when schemas exceed the 512-token
   context limit, the model copies schema text instead of generating a query.
   This is a direct consequence of the sequence length cap.

3. **Property hallucination** — the model invents property names not present
   in the schema (e.g. `question_score` instead of `score`). It learns the
   `node.property` pattern but not the exact field names from the schema.

Exact Match remains 0% across all 50 test examples, reflecting that even
partially correct queries fail strict string comparison.

### What would I do to improve it

With more data, I would balance examples by Cypher operation and schema
pattern, add hard negatives for relationship direction and near-matching
properties, and evaluate on held-out schemas to measure compositional
generalization.

With more compute, I would compare LoRA ranks and target modules, increase
max_seq_length to 1024 to handle complex schemas without truncation, use
completion-only loss so the model only learns from the Cypher output rather
than the full prompt, and benchmark a larger base model.

For a stronger evaluation, I would add syntax validation and execution-based
comparison against isolated read-only Neo4j fixtures. A production system
would also validate generated identifiers against the supplied schema, use
read-only credentials, block mutating clauses, cap query cost and runtime,
and route low-confidence outputs to repair or human review.
# Text2Cypher Fine-tuning

Fine-tunes `HuggingFaceTB/SmolLM2-135M-Instruct` with LoRA to translate a
natural-language question and graph schema into a Cypher query. Training and
inference are configured for CPU-only execution.

## Setup

Python 3.11 is recommended (3.14 is not yet supported by the ML dependencies I used)

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

Paths and generation limits can be changed through command-line arguments:

```bash
python src/train.py --output-dir final_model --cache-dir .cache/huggingface
python src/evaluate.py --model-dir final_model --output-file results/predictions.json
```

To publish the trained adapter after training, authenticate with Hugging Face
and provide a repository ID:

```bash
huggingface-cli login
python src/train.py --push-to-hub --hub-model-id USER/text2cypher-smollm2-lora
```

## Design Decisions & Limitations

### Why I chose LoRA

LoRA freezes the 135M-parameter base model and adds trainable low-rank updates
only to its query and value projection layers. With rank 16, this trains roughly
1% of the parameters, reducing optimizer memory and backward-pass cost on CPU.
It also produces a small adapter artifact while preserving the reusable base
model. Full fine-tuning could offer more capacity, but its CPU cost is not
justified by a 1,000-example training set and would increase overfitting risk.

I set the model to process at most 512 tokens per training example keeping the  CPU cost/coverage tradeoff. A length audit found that 156 of the 1,000 training examples exceed this cap, so those examples are truncated. Raising the cap would preserve more large schemas but substantially increase attention cost; schema-aware truncation would be a better follow-up than silently increasing it.

### Why these metrics

  - **Exact Match** I included it as a standard measure even after knowing it wont  add to the value much as it measures whether the generated query is textually identical to the reference after trimming leading and trailing whitespace. It is strict and easy to interpret, but penalizes semantically equivalent queries that use different aliases, clause ordering, or formatting.
- **Token F1** reports set-based overlap over whitespace-delimited tokens. It
  gives partial credit when the model recovers relevant labels, relationships,
  properties, and clauses, but it ignores token order and can reward a query
  that is not executable or semantically correct.

I intentionally excluded execution accuracy because it requires a populated,
versioned Neo4j test environment and explicit rules for side effects, timeouts,
and result-set equivalence. Reporting it without that infrastructure would produce irreproducible results.

### What the model learned

I first trained it on 200 samples for quick testing which took ~15 mins on CPU and then complete 1000 samples.

Training on 1,000 examples reduced loss from 2.54 to 0.76 across 3 epochs.
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
   in the schema (e.g. question_score instead of score). It learns the
   node.property pattern but not the exact field names.

Exact Match remains 0% across all 50 test examples, reflecting that even
partially correct queries fail strict string comparison.


### What would I do to improve it

With more data, I would balance examples by Cypher operation and schema pattern,
add hard negatives for relationship direction and near-matching properties,
and evaluate on held-out schemas to measure compositional generalization. With
more compute, I would compare LoRA ranks and target modules, tune generation
length, use completion-only loss, and benchmark a larger base model.

For a stronger evaluation, I would add syntax validation and execution-based
comparison against isolated read-only Neo4j fixtures, including result-set
equivalence and timeout/error rates. A production system would also validate
generated identifiers against the supplied schema, use read-only credentials,
block mutating clauses, cap query cost and runtime, log model/data versions, and
route low-confidence or invalid outputs to repair or human review.



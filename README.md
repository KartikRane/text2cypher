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


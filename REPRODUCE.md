# Offline CPU Reproduction Guide

## 1. Environment Setup (Python 3.11 / 3.12)

Install the official benchmark dependencies and locked model runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r submission/requirements.txt
```

## 2. Provision the Declared Public Model (Offline Local Store)

Download the unchanged public model `cycloneboy/SLM-SQL-0.6B` at immutable commit revision `88f5b943278482ac9bc7a1ac631c926266f541f` (596,049,920 deduplicated parameters) into a local directory before offline evaluation:

```bash
huggingface-cli download cycloneboy/SLM-SQL-0.6B \
  --revision 88f5b943278482ac9bc7a1ac631c926266f541f \
  --local-dir ./models/SLM-SQL-0.6B
```

Expected key artifact SHA-256 checksums:
- `model.safetensors`: `666cda41ecf77abd5c24e4a5446d9f5ecc9377e8f59263d90b871bf30907aeea`
- `config.json`: `a1a4b0e765cc593df73d7d72e6128170894fadd9d9107afdd795efecc57b69f9`

## 3. Run Official Prediction & Scoring on CPU (4 Threads, Offline)

Run from the root of the unmodified official `mp3_sql` benchmark package:

```bash
export PYTHONPATH="$PWD/submission"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MP3_BASE_MODEL="$PWD/models/SLM-SQL-0.6B"
export MP3_DEVICE=cpu
export MP3_GENERATION_LOG="$PWD/outputs/generations.jsonl"

mkdir -p outputs
python -B -m mp3_sql.predict \
  --split test \
  --method submission/submission.py \
  --checkpoint submission/config.json \
  --workers 1 \
  --output outputs/test.json

python -B -m mp3_sql.score \
  --split test \
  --predictions outputs/test.json \
  --workers 4 \
  --output outputs/test-score.json
```

All model weights are loaded strictly offline (`local_files_only=True, trust_remote_code=False, use_safetensors=True`) in native `torch.bfloat16` precision without network calls or code execution from remote repositories.

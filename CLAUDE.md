# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Encoder-Decoder Transformer that generates polyphonic musical accompaniment conditioned on a solo melody, genre (Rock, Pop, Funk, Jazz, Latin, Classical, Electronic), and mood (Happy, Sad, Dark, Relaxed, Tense). Input/output format is MIDI; final export is MusicXML.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

All scripts require `PYTHONPATH=src` from the project root.

## Pipeline Commands

```bash
# 1. Label and filter MIDI files
PYTHONPATH=src python src/data/02_filter_and_label.py

# 2. Tokenize MIDIs → JSON token sequences
PYTHONPATH=src python src/data/03_midi_to_tokens.py

# 3. Build HDF5 train/val/test splits
PYTHONPATH=src python src/data/04_build_dataset.py

# 4. Train
PYTHONPATH=src python src/model/train.py

# 5. Generate accompaniment (end-to-end)
PYTHONPATH=src python src/utils/generate_score.py \
  --input_midi input.mid --genre FUNK --mood DARK \
  --instrument BASS --output partitura.xml \
  --temperature 0.9 --top_p 0.92 --top_k 50 --repetition_penalty 1.3
```

```bash
# Evaluate generated MIDIs (with LMD reference)
PYTHONPATH=src python src/utils/evaluate_midi.py \
  --generated results/batch/ \
  --reference data/raw/lmd_matched/ \
  --max_ref 100 \
  --output results/metrics.csv
```

There are no automated tests in this repository.

## Architecture

### Dual Tokenization Scheme

The model uses **two different token representations** — one per side of the encoder-decoder:

- **Encoder (melody)**: Positional tokens — `BAR`, `BEAT`, `POS`, `PITCH`, `DUR`, `VEL`. Structured, bar-aligned.
- **Decoder (accompaniment)**: Event tokens — `NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`, `VELOCITY`. Continuous event stream.

Both share the same vocabulary (`data/tokens/vocabulary.json`, ~525 tokens) and the same embedding matrix (weight tying also applies to the output projection).

### Model (`src/model/transformer.py`)

`MusicTransformer`: standard PyTorch `nn.Transformer` with:

- Pre-LayerNorm (`norm_first=True`) for training stability
- Sinusoidal positional encoding
- Weight tying between token embedding and output linear projection
- 6 encoder + 6 decoder layers, 8 heads, d_model=512, FFN=2048 (~44M params)

### Training (`src/model/train.py`)

- Teacher forcing with causal mask on the decoder
- `TIME_SHIFT` tokens get loss weight 0.5 (`config.time_shift_weight`) to discourage generated silence
- Mixed precision (AMP fp16) + gradient accumulation: `batch_size=4`, `grad_accum=8` → effective batch 32
- OneCycleLR scheduler
- Checkpoints: `checkpoints/best_model_cic.pt`, logs: `checkpoints/train_log.json`

### Inference (`src/model/inference.py`)

Autoregressive decoding with top-k + nucleus (top-p) sampling. Silence suppression logic:

- Tracks consecutive `TIME_SHIFT` tokens and total accumulated silence ticks
- Penalizes `TIME_SHIFT` if consecutive count exceeds `MAX_CONSEC_TIME_SHIFTS=8`
- Applies `NOTE_ON` bonus after `SILENCE_BONUS_THRESHOLD=16` ticks
- Repetition penalty over a rolling window of 32 tokens for pitch variety

### Dataset (`src/model/dataset.py`)

`MusicDataset` reads HDF5 files. Windowing at build time: `max_seq_len=1536`, stride=768 (50% overlap). Genre-balanced `WeightedRandomSampler` for training.

### Configuration (`src/model/config.py`)

`ModelConfig` dataclass is the single source of truth for all hyperparameters. Key values:

| Parameter           | Default | Notes                                    |
| ------------------- | ------- | ---------------------------------------- |
| `vocab_size`        | 525     | Must match `data/tokens/vocabulary.json` |
| `d_model`           | 512     | Embedding dimension                      |
| `max_seq_len`       | 1536    | Max tokens per window                    |
| `time_shift_weight` | 0.5     | Loss weight for TIME_SHIFT tokens        |
| `batch_size`        | 4       | Per-GPU batch                            |
| `grad_accum`        | 8       | Effective batch = 32                     |
| `learning_rate`     | 5e-5    | AdamW                                    |

### Export (`src/utils/tokens_to_musicxml.py`)

Converts both encoder (positional) and decoder (event-based) token sequences into a two-staff MusicXML file via `music21`. `generate_score.py` is the CLI entry point that chains tokenization → inference → export.

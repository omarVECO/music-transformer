# 🎵 Music Accompaniment Transformer

Generative model that creates musical accompaniment from a guitar, bass, or piano audio input. The model takes an audio recording, converts it to a symbolic token representation, and generates an accompanying instrument part as MusicXML.

---

## Overview

```
Audio Input (guitar / bass / piano)
        ↓
  Basic Pitch (Spotify) — audio → MIDI
        ↓
  Post-processor — quantization + chord detection
        ↓
  Tokenizer — MIDI → token sequences
        ↓
  Transformer (Encoder-Decoder) — generates accompaniment tokens
        ↓
  De-tokenizer — tokens → music21 Stream
        ↓
  MusicXML Output
```

The model is conditioned on **genre** (Rock, Pop, Funk, Jazz, Latin, Classical, Electronic) and **mood** (Happy, Sad, Dark, Relaxed, Tense), allowing users to guide the style of the generated accompaniment.

---

## Project Structure

```
music-transformer/
├── data/
│   ├── raw/                  # Lakh MIDI Dataset + MSD HDF5 files
│   ├── processed/            # labeled_midi.csv — filtered & labeled MIDIs
│   └── tokens/               # Tokenized sequences + HDF5 train/val/test splits
│       ├── vocabulary.json
│       ├── index.csv
│       ├── train.h5
│       ├── val.h5
│       └── test.h5
├── src/
│   ├── data/
│   │   ├── 01_explore_dataset.py     # Dataset exploration & statistics
│   │   ├── 02_filter_and_label.py    # Filter MIDIs + assign genre/mood labels
│   │   ├── 03_midi_to_tokens.py      # MIDI → token sequences
│   │   └── 04_build_dataset.py       # Windowing + HDF5 packing
│   ├── model/
│   │   ├── config.py                 # Hyperparameters
│   │   ├── dataset.py                # PyTorch Dataset + DataLoader
│   │   ├── transformer.py            # Encoder-Decoder Transformer
│   │   └── train.py                  # Training loop
│   └── utils/                        # (planned) inference + MusicXML export
├── notebooks/                        # Exploration and analysis
├── checkpoints/                      # Saved model weights
├── requirements.txt
└── README.md
```

---

## Token Vocabulary (~355 tokens)

Each musical event is represented as a flat sequence of atomic tokens:

```
<BAR_3> <CHORD_A_MIN> <BEAT_2> <POS_2> <INST_BASS> <PITCH_45> <DUR_4> <VEL_64>
```

| Category | Examples | Count |
|---|---|---|
| Control | `<SOS>` `<EOS>` `<PAD>` | 6 |
| Conditioning | `<GENRE_FUNK>` `<MOOD_DARK>` `<ENERGY_HIGH>` | 17 |
| Global context | `<TIMESIG_4_4>` `<KEY_A_MIN>` `<TEMPO_120>` | 36 |
| Structural | `<BAR_1>` `<BEAT_2>` `<POS_8>` | 52 |
| Instrument | `<INST_PIANO>` `<INST_BASS>` `<INST_GUITAR>` | 3 |
| Pitch + REST | `<PITCH_28>` … `<PITCH_108>` `<REST>` | 82 |
| Duration | `<DUR_1>` … `<DUR_16>` `<DUR_T2>` `<DUR_T4>` | 10 |
| Velocity | `<VEL_16>` … `<VEL_127>` | 8 |
| Chords | `<CHORD_C_MAJ>` … `<CHORD_B_DIM7>` | 96 |
| **Total** | | **~355** |

---

## Model Architecture

Encoder-Decoder Transformer (~44M parameters):

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `n_heads` | 8 |
| `n_enc_layers` | 6 |
| `n_dec_layers` | 6 |
| `d_ff` | 2048 |
| `dropout` | 0.1 |
| `max_seq_len` | 1536 |
| `vocab_size` | 355 |

Training uses mixed precision (fp16), gradient accumulation (effective batch = 32), and OneCycleLR scheduling.

---

## Dataset

Built from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (LMD-matched) aligned with the Million Song Dataset for genre/mood labels via Last.fm tags.

| Split | Samples |
|---|---|
| Train | 6,098 |
| Validation | 718 |
| Test | 341 |
| **Total** | **7,157** |

Genre distribution: FUNK (1023) · LATIN (996) · POP (993) · ELECTRONIC (937) · JAZZ (924) · CLASSICAL (725) · ROCK (500)

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- ~5GB free disk space for dataset

### Installation

```bash
git clone https://github.com/your-org/music-transformer.git
cd music-transformer

python -m venv venv
source venv/bin/activate

# PyTorch with CUDA (adjust cu version to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Project dependencies
pip install -r requirements.txt
```

> **RTX 4090 / 5000-series users:** These cards require PyTorch nightly. Use the `--pre` flag and `cu128` index URL as shown above.

### Download Dataset

```bash
cd data/raw

# Lakh MIDI Dataset - matched subset
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz
wget http://hog.ee.columbia.edu/craffel/lmd/match_scores.json

tar -xzf lmd_matched.tar.gz
tar -xzf lmd_matched_h5.tar.gz
```

---

## Data Pipeline

Run scripts in order from the project root:

```bash
# 1. Explore dataset structure and tag distribution
PYTHONPATH=src python src/data/01_explore_dataset.py

# 2. Filter MIDIs and assign genre/mood labels
PYTHONPATH=src python src/data/02_filter_and_label.py
# Output: data/processed/labeled_midi.csv (~9,495 labeled MIDIs)

# 3. Convert MIDI to token sequences
PYTHONPATH=src python src/data/03_midi_to_tokens.py
# Output: data/tokens/*.json + index.csv (~8,803 token pairs)

# 4. Apply windowing and build train/val/test HDF5 splits
PYTHONPATH=src python src/data/04_build_dataset.py
# Output: data/tokens/train.h5, val.h5, test.h5
```

---

## Training

```bash
PYTHONPATH=src python src/model/train.py
```

Checkpoints are saved to `checkpoints/best_model.pt` when validation loss improves. Training log is written to `checkpoints/train_log.json`.

**Memory note:** With `batch_size=2` and `grad_accum=16` the effective batch size is 32. If you have more than 16GB VRAM, increase `batch_size` in `src/model/config.py`.

---

## Roadmap

- [x] Token vocabulary design
- [x] Data pipeline (Lakh MIDI + Last.fm labels)
- [x] MIDI tokenizer
- [x] Dataset builder with windowing
- [x] Encoder-Decoder Transformer
- [x] Training loop with mixed precision
- [ ] Inference + greedy/beam search decoding
- [ ] Token → MusicXML export module (via music21)
- [ ] Basic Pitch audio → MIDI integration
- [ ] Web UI / demo interface
- [ ] Phase 2: text-prompt conditioning (replace fixed tokens with text embeddings)

---

## Citation / References

- Raffel, C. (2016). [Learning-Based Methods for Comparing Sequences](https://colinraffel.com/projects/lmd/) — Lakh MIDI Dataset
- Bertin-Mahieux et al. (2011). The Million Song Dataset — MSD / Last.fm tags
- Bitteur et al. — [Basic Pitch](https://basicpitch.spotify.com/) by Spotify
- Vaswani et al. (2017). Attention Is All You Need

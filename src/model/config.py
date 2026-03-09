# src/model/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Vocabulario y secuencia
    vocab_size:     int   = 355
    max_seq_len:    int   = 1536
    pad_id:         int   = 0

    # Arquitectura
    d_model:        int   = 512
    n_heads:        int   = 8
    n_enc_layers:   int   = 6
    n_dec_layers:   int   = 6
    d_ff:           int   = 2048
    dropout:        float = 0.1

    # Entrenamiento — AJUSTADO para 8GB VRAM
    batch_size:     int   = 2      # bajado de 8 → 2
    grad_accum:     int   = 16     # effective batch = 32 (igual que antes)
    learning_rate:  float = 1e-4
    warmup_steps:   int   = 1000
    max_epochs:     int   = 50
    clip_grad:      float = 1.0

    # Paths
    train_h5:   str = "data/tokens/train.h5"
    val_h5:     str = "data/tokens/val.h5"
    test_h5:    str = "data/tokens/test.h5"
    vocab_json: str = "data/tokens/vocabulary.json"
    ckpt_dir:   str = "checkpoints"
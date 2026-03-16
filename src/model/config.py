# src/model/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size:     int   = 355
    max_seq_len:    int   = 1536
    pad_id:         int   = 0

    d_model:        int   = 512
    n_heads:        int   = 8
    n_enc_layers:   int   = 6
    n_dec_layers:   int   = 6
    d_ff:           int   = 2048
    dropout:        float = 0.15    # subir ligeramente — más datos, más capacidad de overfitting

    batch_size:     int   = 4
    grad_accum:     int   = 8       # batch efectivo = 32
    learning_rate:  float = 5e-5    # más bajo — partimos de pesos entrenados
    warmup_steps:   int   = 500     # menos warmup — ya conoce la tarea
    max_epochs:     int   = 30      # menos epochs — 21× más datos por epoch
    clip_grad:      float = 1.0

    train_h5:   str = "data/tokens/train.h5"
    val_h5:     str = "data/tokens/val.h5"
    test_h5:    str = "data/tokens/test.h5"
    vocab_json: str = "data/tokens/vocabulary.json"
    ckpt_dir:   str = "checkpoints"
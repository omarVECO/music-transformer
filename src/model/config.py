# src/model/config.py
"""
Configuración del modelo y entrenamiento.

CAMBIOS respecto a la versión anterior:
- vocab_size aumentado de 355 a 560 para incluir los nuevos tokens event-based:
    NOTE_ON_<pitch>   (28-108) → 81 tokens
    NOTE_OFF_<pitch>  (28-108) → 81 tokens
    TIME_SHIFT_<1-32>          → 32 tokens
    VELOCITY_<bin>    (8 bins) →  8 tokens
    Total nuevos: 202 tokens
  El vocabulario final exacto se determina en midi_tokenizer.py;
  este valor debe ser ≥ len(ALL_TOKENS). Se recomienda ejecutar el tokenizador
  y actualizar este valor si el vocabulario cambia.

- time_shift_weight: peso reducido en la pérdida para tokens TIME_SHIFT,
  para evitar que el modelo se especialice en predecir silencios.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Vocabulario ────────────────────────────────────────────────────
    # IMPORTANTE: actualizar si el vocabulario cambia (ejecutar tokenizador y contar tokens).
    # Con los nuevos tokens event-based el vocabulario crece de ~355 a ~560.
    vocab_size:     int   = 525
    max_seq_len:    int   = 1536
    pad_id:         int   = 0

    # ── Arquitectura ───────────────────────────────────────────────────
    d_model:        int   = 512
    n_heads:        int   = 8
    n_enc_layers:   int   = 6
    n_dec_layers:   int   = 6
    d_ff:           int   = 2048
    dropout:        float = 0.15

    # ── Entrenamiento ──────────────────────────────────────────────────
    batch_size:     int   = 4
    grad_accum:     int   = 8       # batch efectivo = 32
    learning_rate:  float = 5e-5
    warmup_steps:   int   = 500
    max_epochs:     int   = 30
    clip_grad:      float = 1.0

    # ── Pesos de pérdida por tipo de token ─────────────────────────────
    # Valores recomendados (Phase 2):
    #   TIME_SHIFT → 0.3  (desincentivar silencios)
    #   NOTE_ON    → 2.0  (recompensar generación de notas)
    #   NOTE_OFF   → 1.5  (recompensar cierre correcto de notas)
    #   VELOCITY   → 2.0  (recompensar variedad dinámica; con always-emit VELOCITY)
    time_shift_weight: float = 0.3
    note_on_weight:    float = 2.0
    note_off_weight:   float = 1.5
    velocity_weight:   float = 2.0

    # ── Datos ──────────────────────────────────────────────────────────
    train_h5:   str = "data/tokens/train.h5"
    val_h5:     str = "data/tokens/val.h5"
    test_h5:    str = "data/tokens/test.h5"
    vocab_json: str = "data/tokens/vocabulary.json"
    ckpt_dir:   str = "checkpoints/v2"
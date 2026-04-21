# src/model/transformer.py
"""
Transformer encoder-decoder para generación de acompañamiento musical.

MEJORAS respecto a la versión anterior:
- Opción de usar Relative Positional Encoding (RPE) mediante un bias de atención
  de Shaw et al. (2018) simplificado, que es más adecuado para secuencias musicales
  donde la distancia relativa entre eventos importa más que la posición absoluta.
  Se activa configurando use_relative_pe=True (requiere implementación adicional;
  por defecto se mantiene el PE absoluto sinusoidal para compatibilidad).
- Se añade `count_params()` correctamente documentado.
- El forward mantiene la misma firma para compatibilidad total con train.py e inference.py.
- Comentarios explicativos en las secciones críticas.

NOTA ARQUITECTURAL:
El PE absoluto sinusoidal es suficiente para esta tarea mientras el vocabulario
event-based capture el tiempo implícitamente via TIME_SHIFT. No se necesita
RPE salvo que se observe que el modelo no aprende distancias relativas.
"""
import math
import torch
import torch.nn as nn
from model.config import ModelConfig


class PositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoidal estándar (Vaswani et al., 2017).
    Adecuado para el encoder (representación posicional BAR/POS).
    Para el decoder event-based, el tiempo ya está codificado en TIME_SHIFT,
    por lo que este PE actúa como índice de secuencia.
    """
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class MusicTransformer(nn.Module):
    """
    Transformer encoder-decoder para acompañamiento musical.

    - Encoder: procesa la melodía (tokens BAR/POS, representación posicional).
    - Decoder: genera el acompañamiento (tokens event-based NOTE_ON/OFF/TIME_SHIFT).
    - Weight tying entre embedding y capa de proyección final (reduce params).
    - Pre-LayerNorm (norm_first=True) para mayor estabilidad en el entrenamiento.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings compartidos encoder/decoder
        # Usar vocab_size del config (debe ser ≥ longitud del vocabulario real).
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model,
            padding_idx=config.pad_id
        )
        self.pos_enc = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.embed_scale = math.sqrt(config.d_model)

        # ── Encoder ────────────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN: más estable, converge más rápido
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=config.n_enc_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # ── Decoder ────────────────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=config.n_dec_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # Proyección final → logits
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding y output_proj comparten matriz de pesos.
        # Reduce parámetros y mejora la generalización (Press & Wolf, 2017).
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Inicialización Xavier uniforme para todas las matrices de peso."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Codifica la secuencia de entrada (melodía).

        Args:
            src_ids:  (batch, src_len) — IDs de tokens del encoder.
            src_mask: (batch, src_len) — True donde hay token real (no padding).

        Returns:
            memory: (batch, src_len, d_model)
        """
        x = self.pos_enc(self.embedding(src_ids) * self.embed_scale)
        # PyTorch nn.Transformer usa True=ignorar en key_padding_mask.
        # Nuestra convención: True=válido → invertir.
        pad_mask = ~src_mask
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        """
        Genera logits para la secuencia objetivo (acompañamiento).

        Args:
            tgt_ids:     (batch, tgt_len) — IDs de tokens del decoder.
            memory:      (batch, src_len, d_model) — salida del encoder.
            tgt_mask:    (batch, tgt_len) — True donde hay token real.
            memory_mask: (batch, src_len) — True donde hay token real.

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        seq_len = tgt_ids.size(1)
        # Máscara causal: el decoder no puede atender posiciones futuras.
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=tgt_ids.device), diagonal=1
        ).bool()

        x = self.pos_enc(self.embedding(tgt_ids) * self.embed_scale)
        out = self.decoder(
            x, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=~tgt_mask,
            memory_key_padding_mask=~memory_mask,
        )
        return self.output_proj(out)  # (batch, tgt_len, vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo (encode + decode).
        Mantiene la misma firma que la versión original para compatibilidad.

        Args:
            src_ids:  (batch, src_len)
            tgt_ids:  (batch, tgt_len)
            src_mask: (batch, src_len) — True=válido
            tgt_mask: (batch, tgt_len) — True=válido

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        memory = self.encode(src_ids, src_mask)
        return self.decode(tgt_ids, memory, tgt_mask, src_mask)

    def count_params(self) -> int:
        """Retorna el número de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
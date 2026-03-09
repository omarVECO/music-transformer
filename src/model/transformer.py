# src/model/transformer.py
import math
import torch
import torch.nn as nn
from model.config import ModelConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])

class MusicTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings compartidos encoder/decoder
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model,
            padding_idx=config.pad_id
        )
        self.pos_enc = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.embed_scale = math.sqrt(config.d_model)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,     # Pre-LN — más estable en entrenamiento
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=config.n_enc_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # Decoder
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

        # Proyección final → logits sobre vocabulario
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — embedding y output comparten pesos (reduce params, mejora generalización)
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids, src_mask):
        """
        src_ids:  (batch, src_len)
        src_mask: (batch, src_len) — True donde hay token real
        """
        x = self.pos_enc(self.embedding(src_ids) * self.embed_scale)
        # PyTorch usa True=ignorar en key_padding_mask, nosotros tenemos True=válido
        pad_mask = ~src_mask  # invertir
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def decode(self, tgt_ids, memory, tgt_mask, memory_mask):
        """
        tgt_ids:     (batch, tgt_len)
        memory:      (batch, src_len, d_model) — salida del encoder
        tgt_mask:    (batch, tgt_len) — True donde hay token real
        memory_mask: (batch, src_len) — True donde hay token real
        """
        seq_len    = tgt_ids.size(1)
        causal     = torch.triu(
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

    def forward(self, src_ids, tgt_ids, src_mask, tgt_mask):
        memory = self.encode(src_ids, src_mask)
        return self.decode(tgt_ids, memory, tgt_mask, src_mask)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
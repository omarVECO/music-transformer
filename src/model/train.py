# src/model/train.py
"""
Loop de entrenamiento del MusicTransformer.

MEJORAS respecto a la versión anterior:
- Loss weighting: se reduce el peso de los tokens TIME_SHIFT para evitar que el modelo
  aprenda a predecir silencios como salida de baja resistencia.
  El peso se configura en ModelConfig.time_shift_weight.
- Se carga el vocabulario para identificar los IDs de TIME_SHIFT en runtime.
- El resto del pipeline (AMP, GradScaler, OneCycleLR, grad_accum) se mantiene igual.
- Logging mejorado: se reporta la proporción de TIME_SHIFT en el batch de validación
  como métrica de salud musical.
"""
import os
import json
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm

from config import ModelConfig
from transformer import MusicTransformer
from dataset import get_dataloaders


def build_token_weights(config: ModelConfig, vocab_json: str, device: torch.device) -> torch.Tensor:
    """
    Construye un vector de pesos por token para CrossEntropyLoss.

    Los tokens TIME_SHIFT reciben un peso reducido (config.time_shift_weight)
    para desincentivar que el modelo prediga silencios excesivamente.
    Todos los demás tokens reciben peso 1.0.

    Args:
        config:     ModelConfig con vocab_size y time_shift_weight.
        vocab_json: Ruta al JSON de vocabulario generado por el tokenizador.
        device:     Dispositivo donde se ubicará el tensor de pesos.

    Returns:
        weights: tensor (vocab_size,) con float32.
    """
    weights = torch.ones(config.vocab_size, dtype=torch.float32, device=device)

    # Intentar cargar el vocabulario para identificar TIME_SHIFT
    try:
        with open(vocab_json, "r") as f:
            vocab = json.load(f)
        token2id = vocab["token2id"]

        groups = [
            ("<TIME_SHIFT_", config.time_shift_weight),
            ("<NOTE_ON_",    config.note_on_weight),
            ("<NOTE_OFF_",   config.note_off_weight),
            ("<VELOCITY_",   config.velocity_weight),
        ]
        for prefix, w in groups:
            ids = [v for k, v in token2id.items() if k.startswith(prefix)]
            for tid in ids:
                if tid < config.vocab_size:
                    weights[tid] = w
            if ids:
                print(f"  Loss weighting: {len(ids):3d} {prefix}* tokens → {w:.1f}")
        if not any(v for k, v in token2id.items() if k.startswith("<TIME_SHIFT_")):
            print("  Loss weighting: no se encontraron tokens TIME_SHIFT en el vocab (usando pesos uniformes)")

    except FileNotFoundError:
        print(f"  ADVERTENCIA: {vocab_json} no encontrado — usando pesos de pérdida uniformes")
    except Exception as e:
        print(f"  ADVERTENCIA: error leyendo {vocab_json}: {e} — usando pesos uniformes")

    return weights


def train():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_loader, val_loader, _ = get_dataloaders(config)
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * config.max_epochs

    model = MusicTransformer(config).to(device)
    print(f"Parámetros: {model.count_params():,}")

    # ── Pérdida con weighting por tipo de token ────────────────────────
    # Los tokens TIME_SHIFT reciben menos peso para que el modelo no aprenda
    # a predecir silencios de forma excesiva (modo fácil de baja pérdida).
    token_weights = build_token_weights(config, config.vocab_json, device)

    criterion = nn.CrossEntropyLoss(
        weight=token_weights,
        ignore_index=config.pad_id,  # tokens PAD no contribuyen a la pérdida
        label_smoothing=0.1,         # suavizado de etiquetas: regularización leve
    )

    scaler    = GradScaler()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps // config.grad_accum,
        pct_start=config.warmup_steps / (total_steps // config.grad_accum),
        anneal_strategy="cos",
    )

    ckpt_dir      = Path(config.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    train_log     = []
    start_epoch   = 1

    # ── Resume desde checkpoint si existe ──────────────────────────────
    resume_path = ckpt_dir / "best_model.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Fine-tuning desde v1 (val_loss={ckpt['val_loss']:.4f})")
    else:
        print("  Entrenando desde cero")

    for epoch in range(start_epoch, config.max_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader,
                                          desc=f"Epoch {epoch}/{config.max_epochs}")):
            enc_ids  = batch["encoder_ids"].to(device)
            dec_ids  = batch["decoder_ids"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)

            # Teacher forcing: input = [SOS, t1, ..., t_{n-1}], target = [t1, ..., t_n, EOS]
            dec_input  = dec_ids[:, :-1]
            dec_target = dec_ids[:, 1:]
            # La máscara del decoder debe acompañar al input desplazado
            dec_mask_in = dec_mask[:, :-1]

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(enc_ids, dec_input, enc_mask, dec_mask_in)
                # logits: (batch, tgt_len-1, vocab_size)
                # Aplanar para CrossEntropyLoss: (batch*(tgt_len-1), vocab_size)
                loss = criterion(
                    logits.reshape(-1, config.vocab_size),
                    dec_target.reshape(-1),
                )
                loss = loss / config.grad_accum

            scaler.scale(loss).backward()
            train_loss += loss.item() * config.grad_accum

            if (step + 1) % config.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / steps_per_epoch

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        val_steps  = 0

        with torch.no_grad():
            for batch in val_loader:
                enc_ids  = batch["encoder_ids"].to(device)
                dec_ids  = batch["decoder_ids"].to(device)
                enc_mask = batch["encoder_mask"].to(device)
                dec_mask = batch["decoder_mask"].to(device)

                dec_input   = dec_ids[:, :-1]
                dec_target  = dec_ids[:, 1:]
                dec_mask_in = dec_mask[:, :-1]

                with autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(enc_ids, dec_input, enc_mask, dec_mask_in)
                    loss   = criterion(
                        logits.reshape(-1, config.vocab_size),
                        dec_target.reshape(-1),
                    )
                val_loss  += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        vram_used    = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0.0

        print(f"  Epoch {epoch:>3} — train: {avg_train_loss:.4f}  "
              f"val: {avg_val_loss:.4f}  VRAM: {vram_used:.2f}GB")

        train_log.append({
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss":   round(avg_val_loss,   4),
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    best_val_loss,
                "config":      config.__dict__,
            }, ckpt_dir / "best_model.pt")
            print(f"  ✓ Checkpoint guardado (val_loss={best_val_loss:.4f})")

        with open(ckpt_dir / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)


if __name__ == "__main__":
    train()
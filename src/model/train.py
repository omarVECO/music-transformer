# src/model/train.py
import os
import json
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm

from model.config import ModelConfig
from model.transformer import MusicTransformer
from model.dataset import get_dataloaders

def train():
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    train_loader, val_loader, _ = get_dataloaders(config)
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * config.max_epochs

    model = MusicTransformer(config).to(device)
    print(f"Parámetros: {model.count_params():,}")

    scaler    = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate,
                      betas=(0.9, 0.98), weight_decay=0.01)
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

    # ── Resume desde checkpoint si existe ──────────────────────
    resume_path = ckpt_dir / "best_model_v1_no_augmentation.pt"
    start_epoch = 1
    best_val_loss = float("inf")
    train_log = []

    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        # No cargamos optim_state — queremos lr fresco para fine-tuning
        print(f"  Fine-tuning desde v1 (val_loss={ckpt['val_loss']:.4f})")
    else:
        print("  Entrenando desde cero")

    for epoch in range(start_epoch, config.max_epochs + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader,
                                          desc=f"Epoch {epoch}/{config.max_epochs}")):
            enc_ids  = batch["encoder_ids"].to(device)
            dec_ids  = batch["decoder_ids"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)

            dec_input   = dec_ids[:, :-1]
            dec_target  = dec_ids[:, 1:]
            dec_mask_in = dec_mask[:, :-1]

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(enc_ids, dec_input, enc_mask, dec_mask_in)
                loss   = criterion(
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

        # ── Validation ─────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                enc_ids  = batch["encoder_ids"].to(device)
                dec_ids  = batch["decoder_ids"].to(device)
                enc_mask = batch["encoder_mask"].to(device)
                dec_mask = batch["decoder_mask"].to(device)

                dec_input   = dec_ids[:, :-1]
                dec_target  = dec_ids[:, 1:]
                dec_mask_in = dec_mask[:, :-1]

                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(enc_ids, dec_input, enc_mask, dec_mask_in)
                    loss   = criterion(
                        logits.reshape(-1, config.vocab_size),
                        dec_target.reshape(-1),
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        vram_used    = torch.cuda.memory_allocated() / 1e9

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
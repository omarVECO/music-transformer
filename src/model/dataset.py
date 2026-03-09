# src/model/dataset.py
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
import numpy as np

GENRE_LIST = ["CLASSICAL","ELECTRONIC","FUNK","JAZZ","LATIN","POP","ROCK"]
MOOD_LIST  = ["DARK","HAPPY","RELAXED","SAD","TENSE"]

class MusicDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Cargar todo en memoria — 9.2MB cabe perfectamente
        with h5py.File(h5_path, "r") as f:
            self.enc_ids   = torch.tensor(f["encoder_ids"][:],  dtype=torch.long)
            self.dec_ids   = torch.tensor(f["decoder_ids"][:],  dtype=torch.long)
            self.enc_masks = torch.tensor(f["encoder_mask"][:], dtype=torch.bool)
            self.dec_masks = torch.tensor(f["decoder_mask"][:], dtype=torch.bool)
            self.genres    = [g.decode() if isinstance(g, bytes) else g
                              for g in f["genres"][:]]
            self.moods     = [m.decode() if isinstance(m, bytes) else m
                              for m in f["moods"][:]]

        print(f"  Dataset cargado: {len(self)} ejemplos desde {h5_path}")

    def __len__(self):
        return len(self.enc_ids)

    def __getitem__(self, idx):
        return {
            "encoder_ids":  self.enc_ids[idx],
            "decoder_ids":  self.dec_ids[idx],
            "encoder_mask": self.enc_masks[idx],
            "decoder_mask": self.dec_masks[idx],
            "genre":        self.genres[idx],
            "mood":         self.moods[idx],
        }

def make_weighted_sampler(dataset):
    """
    Weighted sampler para compensar desbalance de géneros.
    Cada género tiene probabilidad uniforme; dentro de cada género
    cada ejemplo tiene la misma probabilidad.
    """
    genre_counts = defaultdict(int)
    for g in dataset.genres:
        genre_counts[g] += 1

    weights = []
    for g in dataset.genres:
        weights.append(1.0 / genre_counts[g])

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

def make_causal_mask(seq_len, device):
    """Máscara causal para el decoder (no ve tokens futuros)."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()  # True = ignorar

def get_dataloaders(config):
    train_ds = MusicDataset(config.train_h5)
    val_ds   = MusicDataset(config.val_h5)
    test_ds  = MusicDataset(config.test_h5)

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
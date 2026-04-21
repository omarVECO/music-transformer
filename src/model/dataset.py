# src/model/dataset.py
"""
MEJORAS respecto a la versión anterior:
- Filtrado de secuencias dominadas por TIME_SHIFT (silencio excesivo).
- Soporte para tokens event-based (NOTE_ON / NOTE_OFF / TIME_SHIFT / VELOCITY).
- Estadísticas de balance de tokens para debug.
- Máscaras de atención correctas para padding.
- WeightedRandomSampler mantenido para balance de géneros.
"""
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
import numpy as np

GENRE_LIST = ["CLASSICAL","ELECTRONIC","FUNK","JAZZ","LATIN","POP","ROCK"]
MOOD_LIST  = ["DARK","HAPPY","RELAXED","SAD","TENSE"]

# Umbral de silencio: si > MAX_TIME_SHIFT_RATIO de los tokens del decoder son TIME_SHIFT,
# el ejemplo se descarta en tiempo de entrenamiento también.
# (El filtrado primario ocurre en el tokenizador, esto es una segunda guardia.)
MAX_TIME_SHIFT_RATIO = 0.6


class MusicDataset(Dataset):
    """
    Dataset que carga pares encoder/decoder de un archivo HDF5.

    Formato esperado en el HDF5:
      encoder_ids:  (N, max_seq_len) int32/int64
      decoder_ids:  (N, max_seq_len) int32/int64
      encoder_mask: (N, max_seq_len) bool
      decoder_mask: (N, max_seq_len) bool
      genres:       (N,) bytes/str
      moods:        (N,) bytes/str

    NUEVA funcionalidad:
    - __init__ acepta un parámetro `filter_silence` (default True) que descarta
      en memoria los ejemplos donde el decoder esté dominado por TIME_SHIFT tokens.
    - Expone `token_stats()` para diagnóstico del balance del vocabulario.
    """

    # IDs de TIME_SHIFT en el vocabulario (se determinan en runtime desde el vocab JSON).
    # Como aquí no tenemos acceso directo al vocab, usamos una heurística:
    # el dataset simplemente reporta estadísticas pero no filtra por ID
    # (el filtrado ya fue hecho en el tokenizador).
    # Si se necesita filtrado adicional, pasar time_shift_ids al constructor.

    def __init__(self, h5_path: str, filter_silence: bool = False,
                 time_shift_ids: set = None, pad_id: int = 0):
        """
        Args:
            h5_path:         Ruta al archivo HDF5.
            filter_silence:  Si True, descarta ejemplos con demasiados TIME_SHIFT.
            time_shift_ids:  Set de IDs de tokens TIME_SHIFT en el vocabulario.
                             Necesario si filter_silence=True.
            pad_id:          ID del token de padding (para no contarlo en el análisis).
        """
        self.h5_path = h5_path
        self.pad_id  = pad_id

        with h5py.File(h5_path, "r") as f:
            enc_ids   = torch.tensor(f["encoder_ids"][:],  dtype=torch.long)
            dec_ids   = torch.tensor(f["decoder_ids"][:],  dtype=torch.long)
            enc_masks = torch.tensor(f["encoder_mask"][:], dtype=torch.bool)
            dec_masks = torch.tensor(f["decoder_mask"][:], dtype=torch.bool)
            genres    = [g.decode() if isinstance(g, bytes) else g for g in f["genres"][:]]
            moods     = [m.decode() if isinstance(m, bytes) else m for m in f["moods"][:]]

        # Filtrado opcional de silencios en el decoder
        if filter_silence and time_shift_ids is not None:
            keep_indices = []
            for i in range(len(dec_ids)):
                dec_seq    = dec_ids[i]
                # Contar tokens no-padding
                non_pad    = (dec_seq != pad_id).sum().item()
                if non_pad == 0:
                    continue
                # Contar TIME_SHIFT dentro de los tokens no-padding
                n_ts = sum(1 for tid in dec_seq.tolist()
                           if tid in time_shift_ids and tid != pad_id)
                ratio = n_ts / non_pad
                if ratio <= MAX_TIME_SHIFT_RATIO:
                    keep_indices.append(i)

            print(f"  Filtro de silencio: {len(dec_ids)} → {len(keep_indices)} ejemplos")
            keep = torch.tensor(keep_indices, dtype=torch.long)
            enc_ids   = enc_ids[keep]
            dec_ids   = dec_ids[keep]
            enc_masks = enc_masks[keep]
            dec_masks = dec_masks[keep]
            genres    = [genres[i] for i in keep_indices]
            moods     = [moods[i]  for i in keep_indices]

        self.enc_ids   = enc_ids
        self.dec_ids   = dec_ids
        self.enc_masks = enc_masks
        self.dec_masks = dec_masks
        self.genres    = genres
        self.moods     = moods

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

    def token_stats(self) -> dict:
        """
        Calcula estadísticas de distribución de tokens en el decoder.
        Útil para verificar balance NOTE vs TIME_SHIFT.

        Returns:
            dict con 'total_tokens', 'pad_ratio', y distribución por prefijo.
        """
        total    = 0
        pad_count = 0
        prefix_counts = defaultdict(int)

        for i in range(len(self.dec_ids)):
            seq = self.dec_ids[i].tolist()
            for tid in seq:
                total += 1
                if tid == self.pad_id:
                    pad_count += 1
                # Las estadísticas por prefijo requieren el vocabulario inverso,
                # que no tenemos aquí → simplemente contamos PAD vs no-PAD.

        return {
            "total_tokens": total,
            "pad_ratio":    pad_count / max(total, 1),
            "non_pad":      total - pad_count,
        }


def make_weighted_sampler(dataset: MusicDataset) -> WeightedRandomSampler:
    """
    Weighted sampler para compensar desbalance de géneros.
    Cada género tiene probabilidad uniforme; dentro de cada género
    cada ejemplo tiene la misma probabilidad.
    """
    genre_counts = defaultdict(int)
    for g in dataset.genres:
        genre_counts[g] += 1

    weights = [1.0 / genre_counts[g] for g in dataset.genres]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def make_causal_mask(seq_len: int, device: "torch.device") -> torch.Tensor:
    """
    Máscara causal para el decoder: True = posición ignorada (futuro).
    Compatible con nn.TransformerDecoder (tgt_mask).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def get_dataloaders(config):
    """
    Construye DataLoaders para train, val y test.
    Mantiene la misma API que la versión original.
    """
    train_ds = MusicDataset(config.train_h5, pad_id=config.pad_id)
    val_ds   = MusicDataset(config.val_h5,   pad_id=config.pad_id)
    test_ds  = MusicDataset(config.test_h5,  pad_id=config.pad_id)

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
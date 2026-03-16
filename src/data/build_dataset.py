# src/data/04_build_dataset.py
"""
Toma los tokens generados por 03_midi_to_tokens.py,
aplica windowing y empaca todo en archivos .h5 para el DataLoader.

Salida:
  data/tokens/train.h5
  data/tokens/val.h5
  data/tokens/test.h5
"""
import csv
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

TOKENS_DIR  = Path("data/tokens")
INDEX_PATH  = TOKENS_DIR / "index.csv"
VOCAB_PATH  = TOKENS_DIR / "vocabulary.json"

MAX_SEQ_LEN = 1536
STRIDE      = MAX_SEQ_LEN // 2      # 768 — overlap del 50%
MIN_WIN_LEN = 32                    # ventanas más cortas se descartan

# Split ratios
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
TEST_RATIO  = 0.05

# ─────────────────────────────────────────────────────────────
# Cargar vocabulario
# ─────────────────────────────────────────────────────────────
with open(VOCAB_PATH) as f:
    vocab = json.load(f)

TOKEN2ID = vocab["token2id"]
PAD_ID   = TOKEN2ID["<PAD>"]
SOS_ID   = TOKEN2ID["<SOS>"]
EOS_ID   = TOKEN2ID["<EOS>"]
VOCAB_SIZE = len(TOKEN2ID)
print(f"Vocabulario cargado: {VOCAB_SIZE} tokens")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def tokens_to_ids(tokens):
    return [TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in tokens]

def find_bar_boundaries(token_ids):
    """
    Retorna lista de índices donde comienza cada <BAR_N>.
    Usados para cortar ventanas en límites de compás.
    """
    bar_ids = {TOKEN2ID[f"<BAR_{i}>"] for i in range(1, 33)}
    return [i for i, tid in enumerate(token_ids) if tid in bar_ids]

def make_windows(enc_ids, dec_ids):
    """
    Genera pares (encoder_window, decoder_window) con stride.
    Corta siempre en límites de compás para mantener coherencia musical.
    """
    enc_bars = find_bar_boundaries(enc_ids)
    dec_bars = find_bar_boundaries(dec_ids)

    # Si no hay suficientes compases, una sola ventana
    if len(enc_bars) < 2:
        enc_win = enc_ids[:MAX_SEQ_LEN]
        dec_win = dec_ids[:MAX_SEQ_LEN]
        if len(enc_win) >= MIN_WIN_LEN and len(dec_win) >= MIN_WIN_LEN:
            return [(enc_win, dec_win)]
        return []

    windows = []
    start_bar = 0

    while start_bar < len(enc_bars):
        enc_start = enc_bars[start_bar]

        # Encontrar el bar de fin que no supere MAX_SEQ_LEN
        enc_end_bar = start_bar + 1
        while (enc_end_bar < len(enc_bars) and
               enc_bars[enc_end_bar] - enc_start <= MAX_SEQ_LEN):
            enc_end_bar += 1

        enc_end = enc_bars[enc_end_bar - 1] if enc_end_bar > start_bar + 1 else len(enc_ids)
        enc_win = enc_ids[enc_start:min(enc_start + MAX_SEQ_LEN, len(enc_ids))]

        # Alinear decoder con el mismo rango de compases
        if start_bar < len(dec_bars):
            dec_start = dec_bars[min(start_bar, len(dec_bars) - 1)]
            dec_win   = dec_ids[dec_start:min(dec_start + MAX_SEQ_LEN, len(dec_ids))]
        else:
            dec_win = dec_ids[:MAX_SEQ_LEN]

        if len(enc_win) >= MIN_WIN_LEN and len(dec_win) >= MIN_WIN_LEN:
            windows.append((enc_win, dec_win))

        # Avanzar stride en términos de compases
        bars_per_stride = max(1, int(STRIDE / (MAX_SEQ_LEN / max(len(enc_bars), 1))))
        start_bar += bars_per_stride

        # Si ya cubrimos toda la secuencia, salir
        if enc_start + MAX_SEQ_LEN >= len(enc_ids):
            break

    return windows

def pad_sequence(seq, length):
    """Pad o trunca a length exacto."""
    if len(seq) >= length:
        return seq[:length]
    return seq + [PAD_ID] * (length - len(seq))

def make_attention_mask(seq, length):
    """1 donde hay token real, 0 donde hay PAD."""
    mask = [1] * min(len(seq), length) + [0] * max(0, length - len(seq))
    return mask[:length]

# ─────────────────────────────────────────────────────────────
# Cargar índice y hacer split por canción (no por ventana)
# ─────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(INDEX_PATH)))
print(f"Archivos de tokens: {len(rows)}")

# Shuffle determinista por género para split estratificado
rng = np.random.default_rng(42)

by_genre = defaultdict(list)
for r in rows:
    by_genre[r["genre"]].append(r)

train_rows, val_rows, test_rows = [], [], []

for genre, items in by_genre.items():
    items = list(items)
    rng.shuffle(items)
    n      = len(items)
    n_val  = max(1, int(n * VAL_RATIO))
    n_test = max(1, int(n * TEST_RATIO))
    test_rows  += items[:n_test]
    val_rows   += items[n_test:n_test + n_val]
    train_rows += items[n_test + n_val:]

print(f"Split — train:{len(train_rows)}  val:{len(val_rows)}  test:{len(test_rows)}")

# ─────────────────────────────────────────────────────────────
# Función para construir un split y guardarlo en HDF5
# ─────────────────────────────────────────────────────────────
def build_hdf5(split_rows, out_path, split_name):
    all_enc, all_dec, all_enc_mask, all_dec_mask = [], [], [], []
    all_genres, all_moods = [], []

    genre_counter = defaultdict(int)
    mood_counter  = defaultdict(int)
    window_count  = 0
    skipped       = 0

    for row in tqdm(split_rows, desc=f"Building {split_name}"):
        try:
            with open(row["token_path"]) as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        enc_ids = tokens_to_ids(data["encoder_tokens"])
        dec_ids = tokens_to_ids(data["decoder_tokens"])

        windows = make_windows(enc_ids, dec_ids)
        if not windows:
            skipped += 1
            continue

        for enc_win, dec_win in windows:
            enc_pad  = pad_sequence(enc_win, MAX_SEQ_LEN)
            dec_pad  = pad_sequence(dec_win, MAX_SEQ_LEN)
            enc_mask = make_attention_mask(enc_win, MAX_SEQ_LEN)
            dec_mask = make_attention_mask(dec_win, MAX_SEQ_LEN)

            all_enc.append(enc_pad)
            all_dec.append(dec_pad)
            all_enc_mask.append(enc_mask)
            all_dec_mask.append(dec_mask)
            all_genres.append(data["genre"])
            all_moods.append(data["mood"])

            genre_counter[data["genre"]] += 1
            mood_counter[data["mood"]]   += 1
            window_count += 1

    # Convertir a numpy
    enc_arr      = np.array(all_enc,      dtype=np.int32)
    dec_arr      = np.array(all_dec,      dtype=np.int32)
    enc_mask_arr = np.array(all_enc_mask, dtype=np.int8)
    dec_mask_arr = np.array(all_dec_mask, dtype=np.int8)

    # Guardar HDF5
    with h5py.File(out_path, "w") as f:
        f.create_dataset("encoder_ids",   data=enc_arr,      compression="gzip", chunks=True)
        f.create_dataset("decoder_ids",   data=dec_arr,      compression="gzip", chunks=True)
        f.create_dataset("encoder_mask",  data=enc_mask_arr, compression="gzip", chunks=True)
        f.create_dataset("decoder_mask",  data=dec_mask_arr, compression="gzip", chunks=True)

        # Metadatos como strings
        dt = h5py.special_dtype(vlen=str)
        genre_ds = f.create_dataset("genres", (window_count,), dtype=dt)
        mood_ds  = f.create_dataset("moods",  (window_count,), dtype=dt)
        genre_ds[:] = all_genres
        mood_ds[:]  = all_moods

        # Atributos globales
        f.attrs["split"]      = split_name
        f.attrs["n_samples"]  = window_count
        f.attrs["max_seq_len"]= MAX_SEQ_LEN
        f.attrs["vocab_size"] = VOCAB_SIZE
        f.attrs["pad_id"]     = PAD_ID

    print(f"\n  [{split_name}] {window_count} ventanas — guardado en {out_path}")
    print(f"  Skipped: {skipped}")
    print(f"  Géneros: { {g: c for g, c in sorted(genre_counter.items())} }")
    print(f"  Moods:   { {m: c for m, c in sorted(mood_counter.items())} }")
    print(f"  Tamaño en disco: {out_path.stat().st_size / 1e6:.1f} MB")

    return window_count

# ─────────────────────────────────────────────────────────────
# Construir los tres splits
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
n_train = build_hdf5(train_rows, TOKENS_DIR / "train.h5", "train")
n_val   = build_hdf5(val_rows,   TOKENS_DIR / "val.h5",   "val")
n_test  = build_hdf5(test_rows,  TOKENS_DIR / "test.h5",  "test")

print("\n" + "=" * 55)
print("  DATASET FINAL")
print("=" * 55)
print(f"  Train: {n_train} ventanas")
print(f"  Val:   {n_val} ventanas")
print(f"  Test:  {n_test} ventanas")
print(f"  Total: {n_train + n_val + n_test} ventanas")
print(f"  max_seq_len: {MAX_SEQ_LEN}")
print(f"  vocab_size:  {VOCAB_SIZE}")
print("=" * 55)
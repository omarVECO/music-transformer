# src/data/05_augment_and_rebuild.py
"""
Data augmentation por transposición y variación de tempo.
Regenera los splits HDF5 desde cero incluyendo los ejemplos aumentados.

Transformaciones por cada MIDI original:
  - Transposiciones: +1, +2, +3, +5, +7, +10 semitonos
  - Tempos: ×0.85 (lento) y ×1.15 (rápido)
  - Combinadas: 6 transp × 2 tempos = 12 variantes + 2 tempos del original
  - Total por MIDI: 1 original + 14 variantes = 15×
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
STRIDE      = MAX_SEQ_LEN // 2
MIN_WIN_LEN = 32
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
TEST_RATIO  = 0.05

TRANSPOSITIONS = [0, 1, 2, 3, 5, 7, 10]   # 0 = original
TEMPO_FACTORS  = [1.0, 0.85, 1.15]          # 1.0 = original

# ─────────────────────────────────────────────────────────────
# Cargar vocabulario
# ─────────────────────────────────────────────────────────────
with open(VOCAB_PATH) as f:
    vocab = json.load(f)

TOKEN2ID   = vocab["token2id"]
ID2TOKEN   = {int(k): v for k, v in vocab["id2token"].items()}
PAD_ID     = TOKEN2ID["<PAD>"]
VOCAB_SIZE = len(TOKEN2ID)

# Pre-computar mapas de transposición para pitches
MIN_PITCH = 28
MAX_PITCH = 108

def build_pitch_transpose_map(semitones):
    """
    Retorna dict {token_id_original: token_id_transpuesto}
    para todos los tokens de pitch.
    Notas que salen del rango se mapean a la octava más cercana dentro del rango.
    """
    mapping = {}
    for p in range(MIN_PITCH, MAX_PITCH + 1):
        src_tok = f"<PITCH_{p}>"
        if src_tok not in TOKEN2ID:
            continue
        new_p = p + semitones
        # Wrap a rango válido por octavas
        while new_p > MAX_PITCH:
            new_p -= 12
        while new_p < MIN_PITCH:
            new_p += 12
        dst_tok = f"<PITCH_{new_p}>"
        if dst_tok in TOKEN2ID:
            mapping[TOKEN2ID[src_tok]] = TOKEN2ID[dst_tok]
    return mapping

def build_key_transpose_map(semitones):
    """Transpone tokens de KEY y CHORD."""
    note_names = ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
    mapping = {}

    # Tonalidades
    for i, n in enumerate(note_names):
        for mode in ["MAJ", "MIN"]:
            src = f"<KEY_{n}_{mode}>"
            if src not in TOKEN2ID:
                continue
            new_n = note_names[(i + semitones) % 12]
            dst = f"<KEY_{new_n}_{mode}>"
            if dst in TOKEN2ID:
                mapping[TOKEN2ID[src]] = TOKEN2ID[dst]

    # Acordes
    for i, n in enumerate(note_names):
        for q in ["MAJ","MIN","DIM","AUG","MAJ7","MIN7","DOM7","DIM7"]:
            src = f"<CHORD_{n}_{q}>"
            if src not in TOKEN2ID:
                continue
            new_n = note_names[(i + semitones) % 12]
            dst = f"<CHORD_{new_n}_{q}>"
            if dst in TOKEN2ID:
                mapping[TOKEN2ID[src]] = TOKEN2ID[dst]

    return mapping

def build_tempo_remap(factor):
    """
    Remapea tokens de TEMPO según el factor.
    factor < 1.0 = más lento → tempo token más bajo
    factor > 1.0 = más rápido → tempo token más alto
    """
    TEMPO_VALS   = [60, 80, 100, 120, 140, 160, 180, 200]
    TEMPO_TOKENS = [f"<TEMPO_{v}>" for v in TEMPO_VALS]
    mapping = {}
    for i, tok in enumerate(TEMPO_TOKENS):
        if tok not in TOKEN2ID:
            continue
        new_bpm = TEMPO_VALS[i] * factor
        # Encontrar bin más cercano
        closest = min(TEMPO_VALS, key=lambda x: abs(x - new_bpm))
        dst = f"<TEMPO_{closest}>"
        if dst in TOKEN2ID:
            mapping[TOKEN2ID[tok]] = TOKEN2ID[dst]
    return mapping

# Pre-computar todos los mapas
print("Pre-computando mapas de transformación...")
transpose_maps = {}
key_maps       = {}
for s in TRANSPOSITIONS:
    transpose_maps[s] = build_pitch_transpose_map(s)
    key_maps[s]       = build_key_transpose_map(s)

tempo_maps = {}
for f in TEMPO_FACTORS:
    tempo_maps[f] = build_tempo_remap(f)

print(f"  Transposiciones: {TRANSPOSITIONS}")
print(f"  Factores de tempo: {TEMPO_FACTORS}")

# ─────────────────────────────────────────────────────────────
# Aplicar transformación a una secuencia de token IDs
# ─────────────────────────────────────────────────────────────
def transform_sequence(token_ids, semitones, tempo_factor):
    """Aplica transposición + cambio de tempo a una secuencia."""
    if semitones == 0 and tempo_factor == 1.0:
        return token_ids  # sin cambios

    p_map = transpose_maps[semitones]
    k_map = key_maps[semitones]
    t_map = tempo_maps[tempo_factor]

    result = []
    for tid in token_ids:
        if tid in p_map:
            result.append(p_map[tid])
        elif tid in k_map:
            result.append(k_map[tid])
        elif tid in t_map:
            result.append(t_map[tid])
        else:
            result.append(tid)
    return result

# ─────────────────────────────────────────────────────────────
# Windowing (igual que build_dataset.py)
# ─────────────────────────────────────────────────────────────
def find_bar_boundaries(token_ids):
    bar_ids = {TOKEN2ID[f"<BAR_{i}>"] for i in range(1, 33)}
    return [i for i, tid in enumerate(token_ids) if tid in bar_ids]

def make_windows(enc_ids, dec_ids):
    enc_bars = find_bar_boundaries(enc_ids)
    dec_bars = find_bar_boundaries(dec_ids)

    if len(enc_bars) < 2:
        enc_win = enc_ids[:MAX_SEQ_LEN]
        dec_win = dec_ids[:MAX_SEQ_LEN]
        if len(enc_win) >= MIN_WIN_LEN and len(dec_win) >= MIN_WIN_LEN:
            return [(enc_win, dec_win)]
        return []

    windows    = []
    start_bar  = 0

    while start_bar < len(enc_bars):
        enc_start   = enc_bars[start_bar]
        enc_end_bar = start_bar + 1
        while (enc_end_bar < len(enc_bars) and
               enc_bars[enc_end_bar] - enc_start <= MAX_SEQ_LEN):
            enc_end_bar += 1

        enc_win = enc_ids[enc_start:min(enc_start + MAX_SEQ_LEN, len(enc_ids))]

        if start_bar < len(dec_bars):
            dec_start = dec_bars[min(start_bar, len(dec_bars) - 1)]
            dec_win   = dec_ids[dec_start:min(dec_start + MAX_SEQ_LEN, len(dec_ids))]
        else:
            dec_win = dec_ids[:MAX_SEQ_LEN]

        if len(enc_win) >= MIN_WIN_LEN and len(dec_win) >= MIN_WIN_LEN:
            windows.append((enc_win, dec_win))

        bars_per_stride = max(1, int(STRIDE / (MAX_SEQ_LEN / max(len(enc_bars), 1))))
        start_bar += bars_per_stride

        if enc_start + MAX_SEQ_LEN >= len(enc_ids):
            break

    return windows

def pad_sequence(seq, length):
    if len(seq) >= length:
        return seq[:length]
    return seq + [PAD_ID] * (length - len(seq))

def make_attention_mask(seq, length):
    mask = [1] * min(len(seq), length) + [0] * max(0, length - len(seq))
    return mask[:length]

# ─────────────────────────────────────────────────────────────
# Split estratificado por canción
# ─────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(INDEX_PATH)))
print(f"\nArchivos originales: {len(rows)}")

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

print(f"Split base — train:{len(train_rows)}  val:{len(val_rows)}  test:{len(test_rows)}")

# ─────────────────────────────────────────────────────────────
# Construir HDF5 con augmentación
# ─────────────────────────────────────────────────────────────
def build_hdf5_augmented(split_rows, out_path, split_name, augment=True):
    all_enc, all_dec       = [], []
    all_enc_mask, all_dec_mask = [], []
    all_genres, all_moods  = [], []

    genre_counter = defaultdict(int)
    mood_counter  = defaultdict(int)
    window_count  = 0
    skipped       = 0

    # Definir qué transformaciones aplicar según el split
    if augment:
        transforms = [
            (s, f)
            for s in TRANSPOSITIONS
            for f in TEMPO_FACTORS
            if not (s == 0 and f == 1.0)  # excluir original (se añade siempre)
        ]
    else:
        transforms = []  # val y test sin augmentación para evaluación limpia

    for row in tqdm(split_rows, desc=f"Building {split_name}"):
        try:
            with open(row["token_path"]) as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        enc_ids_orig = [TOKEN2ID.get(t, TOKEN2ID["<UNK>"])
                        for t in data["encoder_tokens"]]
        dec_ids_orig = [TOKEN2ID.get(t, TOKEN2ID["<UNK>"])
                        for t in data["decoder_tokens"]]

        # Siempre incluir el original
        all_variants = [(enc_ids_orig, dec_ids_orig)]

        # Añadir variantes aumentadas
        for semitones, tempo_f in transforms:
            enc_aug = transform_sequence(enc_ids_orig, semitones, tempo_f)
            dec_aug = transform_sequence(dec_ids_orig, semitones, tempo_f)
            all_variants.append((enc_aug, dec_aug))

        for enc_ids, dec_ids in all_variants:
            windows = make_windows(enc_ids, dec_ids)
            if not windows:
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

    enc_arr      = np.array(all_enc,      dtype=np.int32)
    dec_arr      = np.array(all_dec,      dtype=np.int32)
    enc_mask_arr = np.array(all_enc_mask, dtype=np.int8)
    dec_mask_arr = np.array(all_dec_mask, dtype=np.int8)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("encoder_ids",  data=enc_arr,      compression="gzip", chunks=True)
        f.create_dataset("decoder_ids",  data=dec_arr,      compression="gzip", chunks=True)
        f.create_dataset("encoder_mask", data=enc_mask_arr, compression="gzip", chunks=True)
        f.create_dataset("decoder_mask", data=dec_mask_arr, compression="gzip", chunks=True)

        dt = h5py.special_dtype(vlen=str)
        genre_ds = f.create_dataset("genres", (window_count,), dtype=dt)
        mood_ds  = f.create_dataset("moods",  (window_count,), dtype=dt)
        genre_ds[:] = all_genres
        mood_ds[:]  = all_moods

        f.attrs["split"]       = split_name
        f.attrs["n_samples"]   = window_count
        f.attrs["max_seq_len"] = MAX_SEQ_LEN
        f.attrs["vocab_size"]  = VOCAB_SIZE
        f.attrs["pad_id"]      = PAD_ID
        f.attrs["augmented"]   = augment

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  [{split_name}] {window_count:,} ventanas — {size_mb:.1f} MB")
    print(f"  Skipped: {skipped}")
    print(f"  Géneros: { {g: c for g, c in sorted(genre_counter.items())} }")
    print(f"  Moods:   { {m: c for m, c in sorted(mood_counter.items())} }")

    return window_count

# ─────────────────────────────────────────────────────────────
# Construir los tres splits
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)

# Respaldar splits anteriores
for name in ["train", "val", "test"]:
    src = TOKENS_DIR / f"{name}.h5"
    dst = TOKENS_DIR / f"{name}_v1.h5"
    if src.exists() and not dst.exists():
        import shutil
        shutil.copy(src, dst)
        print(f"  Respaldado: {dst}")

n_train = build_hdf5_augmented(train_rows, TOKENS_DIR / "train.h5", "train", augment=True)
n_val   = build_hdf5_augmented(val_rows,   TOKENS_DIR / "val.h5",   "val",   augment=False)
n_test  = build_hdf5_augmented(test_rows,  TOKENS_DIR / "test.h5",  "test",  augment=False)

print("\n" + "=" * 55)
print("  DATASET AUMENTADO")
print("=" * 55)
print(f"  Train: {n_train:,} ventanas  (augmentado)")
print(f"  Val:   {n_val:,} ventanas   (sin augmentar)")
print(f"  Test:  {n_test:,} ventanas  (sin augmentar)")
print(f"  Total: {n_train + n_val + n_test:,} ventanas")
print(f"  Factor de aumento aprox: {n_train // 6098:.1f}×")
print("=" * 55)
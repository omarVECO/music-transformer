# src/data/03_midi_to_tokens.py
"""
Convierte cada MIDI etiquetado en pares de secuencias de tokens:
  - encoder_tokens: pista melódica (lo que tocó el usuario)
  - decoder_tokens: pista de acompañamiento (lo que debe generar el modelo)

Salida: data/tokens/<msd_id>.json por cada MIDI procesado
        data/tokens/index.csv con rutas + etiquetas
"""
import csv
import json
import pretty_midi
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

CSV_PATH   = Path("data/processed/labeled_midi.csv")
TOKENS_DIR = Path("data/tokens")
TOKENS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
PPQ             = 16        # subdivisiones por negra (semicorchea = 1 tick)
TICKS_PER_BAR   = 64        # 4/4: 4 negras × 16 subdivisiones
MAX_BARS        = 32        # máximo de compases por segmento
MIN_BARS        = 4         # mínimo para que valga la pena
MAX_PITCH       = 108
MIN_PITCH       = 28
VELOCITY_BINS   = [16, 32, 48, 64, 80, 96, 112, 127]

# Clases de instrumento → rol
MELODY_CLASSES  = {"Guitar", "Piano", "Reed", "Synth Lead", "Chromatic Percussion"}
ACCOMP_CLASSES  = {"Bass", "Piano", "Guitar", "Strings", "Ensemble"}

# ─────────────────────────────────────────────────────────────
# Vocabulario
# ─────────────────────────────────────────────────────────────
SPECIAL   = ["<PAD>", "<SOS>", "<EOS>", "<SEP>", "<UNK>", "<MASK>"]
GENRES    = ["<GENRE_ROCK>", "<GENRE_POP>", "<GENRE_FUNK>", "<GENRE_JAZZ>",
             "<GENRE_LATIN>", "<GENRE_CLASSICAL>", "<GENRE_ELECTRONIC>"]
MOODS     = ["<MOOD_HAPPY>", "<MOOD_SAD>", "<MOOD_DARK>", "<MOOD_RELAXED>", "<MOOD_TENSE>"]
ENERGIES  = ["<ENERGY_LOW>", "<ENERGY_MED>", "<ENERGY_HIGH>"]
TIMESIGS  = ["<TIMESIG_3_4>", "<TIMESIG_4_4>", "<TIMESIG_6_8>"]
KEYS      = [f"<KEY_{n}_{m}>" for n in
             ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
             for m in ["MAJ","MIN"]]
TEMPOS    = ["<TEMPO_60>","<TEMPO_80>","<TEMPO_100>","<TEMPO_120>",
             "<TEMPO_140>","<TEMPO_160>","<TEMPO_180>","<TEMPO_200>"]
BARS      = [f"<BAR_{i}>" for i in range(1, MAX_BARS + 1)]
BEATS     = [f"<BEAT_{i}>" for i in range(1, 5)]
POSITIONS = [f"<POS_{i}>" for i in range(TICKS_PER_BAR)]
INSTS     = ["<INST_PIANO>", "<INST_BASS>", "<INST_GUITAR>"]
PITCHES   = [f"<PITCH_{i}>" for i in range(MIN_PITCH, MAX_PITCH + 1)] + ["<REST>"]
DURATIONS = ["<DUR_1>","<DUR_2>","<DUR_3>","<DUR_4>","<DUR_6>",
             "<DUR_8>","<DUR_12>","<DUR_16>","<DUR_T2>","<DUR_T4>"]
VELOCITIES= [f"<VEL_{v}>" for v in VELOCITY_BINS]
CHORDS    = [f"<CHORD_{n}_{q}>"
             for n in ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
             for q in ["MAJ","MIN","DIM","AUG","MAJ7","MIN7","DOM7","DIM7"]]

ALL_TOKENS = (SPECIAL + GENRES + MOODS + ENERGIES + TIMESIGS + KEYS +
              TEMPOS + BARS + BEATS + POSITIONS + INSTS + PITCHES +
              DURATIONS + VELOCITIES + CHORDS)

TOKEN2ID = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}

# Guardar vocabulario
with open(TOKENS_DIR / "vocabulary.json", "w") as f:
    json.dump({"token2id": TOKEN2ID, "id2token": {str(k): v for k,v in ID2TOKEN.items()}}, f)
print(f"Vocabulario: {len(TOKEN2ID)} tokens")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def quantize_velocity(vel):
    for bin_val in VELOCITY_BINS:
        if vel <= bin_val:
            return f"<VEL_{bin_val}>"
    return f"<VEL_127>"

def quantize_duration(ticks):
    # Mapear ticks a token de duración más cercano
    DUR_MAP = {1:"<DUR_1>", 2:"<DUR_2>", 3:"<DUR_3>", 4:"<DUR_4>",
               6:"<DUR_6>", 8:"<DUR_8>", 12:"<DUR_12>", 16:"<DUR_16>"}
    if ticks <= 0:
        return "<DUR_1>"
    closest = min(DUR_MAP.keys(), key=lambda x: abs(x - ticks))
    return DUR_MAP[closest]

def quantize_tempo(bpm):
    thresholds = [70, 90, 110, 130, 150, 170, 190]
    labels     = ["<TEMPO_60>","<TEMPO_80>","<TEMPO_100>","<TEMPO_120>",
                  "<TEMPO_140>","<TEMPO_160>","<TEMPO_180>","<TEMPO_200>"]
    for t, label in zip(thresholds, labels):
        if bpm < t:
            return label
    return "<TEMPO_200>"

def energy_to_token(energy):
    # energy en MSD es 0-1
    if energy < 0.33:   return "<ENERGY_LOW>"
    if energy < 0.66:   return "<ENERGY_MED>"
    return "<ENERGY_HIGH>"

def detect_key(pm):
    """Estima la tonalidad usando el perfil de Krumhansl."""
    pitch_classes = np.zeros(12)
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                pitch_classes[note.pitch % 12] += note.end - note.start

    # Perfiles de Krumhansl-Schmuckler
    major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    best_key, best_mode, best_score = 0, "MAJ", -np.inf
    note_names = ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]

    for root in range(12):
        rotated = np.roll(pitch_classes, -root)
        score_maj = np.corrcoef(rotated, major)[0,1]
        score_min = np.corrcoef(rotated, minor)[0,1]
        if score_maj > best_score:
            best_score, best_key, best_mode = score_maj, root, "MAJ"
        if score_min > best_score:
            best_score, best_key, best_mode = score_min, root, "MIN"

    return f"<KEY_{note_names[best_key]}_{best_mode}>"

def detect_chord(notes_in_beat):
    """Detecta el acorde más probable dado un conjunto de pitches."""
    if not notes_in_beat:
        return None
    pcs = list({n % 12 for n in notes_in_beat})
    note_names = ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]

    # Intervalos característicos por calidad
    CHORD_TEMPLATES = {
        "MAJ":  [0, 4, 7],
        "MIN":  [0, 3, 7],
        "DIM":  [0, 3, 6],
        "AUG":  [0, 4, 8],
        "MAJ7": [0, 4, 7, 11],
        "MIN7": [0, 3, 7, 10],
        "DOM7": [0, 4, 7, 10],
        "DIM7": [0, 3, 6, 9],
    }

    best_root, best_quality, best_score = 0, "MAJ", -1
    for root in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            chord_pcs = {(root + i) % 12 for i in intervals}
            overlap   = len(chord_pcs & set(pcs))
            score     = overlap / max(len(chord_pcs), len(pcs))
            if score > best_score:
                best_score, best_root, best_quality = score, root, quality

    if best_score < 0.4:   # no hay acorde claro
        return None
    return f"<CHORD_{note_names[best_root]}_{best_quality}>"

def select_tracks(pm):
    """
    Selecciona la pista melódica y la pista de acompañamiento.
    Retorna (melody_inst, accomp_inst) o (None, None) si no es viable.
    """
    non_drum = [i for i in pm.instruments if not i.is_drum and len(i.notes) > 0]
    if len(non_drum) < 2:
        return None, None

    melody_candidates = []
    accomp_candidates = []

    for inst in non_drum:
        cls = pretty_midi.program_to_instrument_class(inst.program)

        # Calcular nota más alta promedio (proxy de voz melódica)
        avg_pitch = np.mean([n.pitch for n in inst.notes])
        note_count = len(inst.notes)

        if cls in MELODY_CLASSES:
            melody_candidates.append((inst, avg_pitch, note_count))
        if cls in ACCOMP_CLASSES:
            accomp_candidates.append((inst, avg_pitch, note_count))

    if not melody_candidates or not accomp_candidates:
        return None, None

    # Melodía: mayor pitch promedio (voz más aguda)
    melody = max(melody_candidates, key=lambda x: x[1])[0]

    # Acompañamiento: diferente a la melodía, preferir bajo o piano
    accomp_candidates = [(i, p, c) for i, p, c in accomp_candidates if i != melody]
    if not accomp_candidates:
        return None, None

    # Priorizar bajo (pitch más bajo) para acompañamiento
    accomp = min(accomp_candidates, key=lambda x: x[1])[0]

    return melody, accomp

def inst_to_token(inst):
    cls = pretty_midi.program_to_instrument_class(inst.program)
    if cls == "Bass":     return "<INST_BASS>"
    if cls == "Guitar":   return "<INST_GUITAR>"
    return "<INST_PIANO>"

def notes_to_token_sequence(inst, pm, tempo_bpm, key_token,
                             genre, mood, energy, inst_token,
                             is_encoder=True):
    """
    Convierte las notas de un instrumento en secuencia de tokens.
    """
    if tempo_bpm < 10.0:
        tempo_bpm = 120.0
        
    seconds_per_tick = 60.0 / (tempo_bpm * PPQ)
    end_time         = pm.get_end_time()
    total_bars       = min(int(end_time / (seconds_per_tick * TICKS_PER_BAR)), MAX_BARS)

    if total_bars < MIN_BARS:
        return None

    # Indexar notas por tick de inicio
    notes_by_tick = defaultdict(list)
    for note in inst.notes:
        tick  = int(round(note.start / seconds_per_tick))
        dur_t = int(round((note.end - note.start) / seconds_per_tick))
        dur_t = max(1, dur_t)
        notes_by_tick[tick].append((note.pitch, dur_t, note.velocity))

    # Construir secuencia
    tokens = []

    # Header — contexto global
    timesig = "<TIMESIG_4_4>"   # asumimos 4/4, se puede detectar
    tempo_t = quantize_tempo(tempo_bpm)

    if is_encoder:
        tokens += ["<SOS>", timesig, key_token, tempo_t]
    else:
        tokens += ["<SOS>", f"<GENRE_{genre}>", f"<MOOD_{mood}>",
                   energy_to_token(energy), inst_token]

    prev_bar  = -1
    prev_beat = -1

    for bar_idx in range(total_bars):
        bar_token  = f"<BAR_{bar_idx + 1}>"

        # Detectar acorde del compás (primeras notas del bar)
        bar_start_tick = bar_idx * TICKS_PER_BAR
        bar_notes = []
        for tick_offset in range(TICKS_PER_BAR):
            tick = bar_start_tick + tick_offset
            bar_notes += [p for p, d, v in notes_by_tick.get(tick, [])]
        chord_token = detect_chord(bar_notes)

        bar_emitted   = False
        chord_emitted = False

        for pos in range(TICKS_PER_BAR):
            tick        = bar_idx * TICKS_PER_BAR + pos
            beat_idx    = pos // PPQ          # 0-3
            beat_token  = f"<BEAT_{beat_idx + 1}>"
            pos_token   = f"<POS_{pos}>"

            if tick not in notes_by_tick:
                continue

            # Emitir BAR y CHORD una sola vez por compás
            if not bar_emitted:
                tokens.append(bar_token)
                bar_emitted = True
            if not chord_emitted and chord_token:
                tokens.append(chord_token)
                chord_emitted = True

            # Emitir BEAT solo cuando cambia
            if beat_idx != prev_beat:
                tokens.append(beat_token)
                prev_beat = beat_idx

            tokens.append(pos_token)

            # Emitir notas en este tick
            for pitch, dur_ticks, velocity in notes_by_tick[tick]:
                if pitch < MIN_PITCH or pitch > MAX_PITCH:
                    continue
                tokens.append(inst_token)
                tokens.append(f"<PITCH_{pitch}>")
                tokens.append(quantize_duration(dur_ticks))
                tokens.append(quantize_velocity(velocity))

    tokens.append("<EOS>")
    return tokens

# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"\nProcesando {len(rows)} MIDIs...")

index_rows = []
stats = defaultdict(int)

for row in tqdm(rows, desc="Tokenizando"):
    msd_id    = row["msd_id"]
    midi_path = row["midi_path"]
    genre     = row["genre"]
    mood      = row["mood"]
    tempo_bpm = float(row["tempo"]) if row["tempo"] else 120.0
    energy    = float(row["energy"]) if row["energy"] else 0.5

    # Cargar MIDI
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        stats["midi_load_error"] += 1
        continue

    if tempo_bpm < 10.0:
        estimated = pm.estimate_tempo()
        tempo_bpm = estimated if 30 < estimated < 300 else 120.0

    # Seleccionar tracks
    melody_inst, accomp_inst = select_tracks(pm)
    if melody_inst is None:
        stats["no_valid_tracks"] += 1
        continue

    # Detectar tonalidad
    key_token   = detect_key(pm)
    inst_token  = inst_to_token(accomp_inst)

    # Tokenizar encoder (melodía)
    enc_tokens = notes_to_token_sequence(
        melody_inst, pm, tempo_bpm, key_token,
        genre, mood, energy, inst_to_token(melody_inst),
        is_encoder=True
    )
    if enc_tokens is None:
        stats["too_short"] += 1
        continue

    # Tokenizar decoder (acompañamiento)
    dec_tokens = notes_to_token_sequence(
        accomp_inst, pm, tempo_bpm, key_token,
        genre, mood, energy, inst_token,
        is_encoder=False
    )
    if dec_tokens is None:
        stats["too_short"] += 1
        continue

    # Validar que los tokens estén en el vocabulario
    enc_valid = all(t in TOKEN2ID for t in enc_tokens)
    dec_valid = all(t in TOKEN2ID for t in dec_tokens)
    if not enc_valid or not dec_valid:
        stats["oov_tokens"] += 1
        # Log cuáles tokens no están
        for t in enc_tokens + dec_tokens:
            if t not in TOKEN2ID:
                print(f"  OOV: {t}")
        continue

    # Guardar
    out_path = TOKENS_DIR / f"{msd_id}.json"
    with open(out_path, "w") as f:
        json.dump({
            "msd_id":         msd_id,
            "genre":          genre,
            "mood":           mood,
            "encoder_tokens": enc_tokens,
            "decoder_tokens": dec_tokens,
        }, f)

    index_rows.append({
        "msd_id":       msd_id,
        "token_path":   str(out_path),
        "genre":        genre,
        "mood":         mood,
        "enc_len":      len(enc_tokens),
        "dec_len":      len(dec_tokens),
    })
    stats["accepted"] += 1

# Guardar índice
index_path = TOKENS_DIR / "index.csv"
with open(index_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["msd_id","token_path","genre","mood","enc_len","dec_len"])
    writer.writeheader()
    writer.writerows(index_rows)

# ─────────────────────────────────────────────────────────────
# Reporte
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  REPORTE TOKENIZACIÓN")
print("=" * 55)
print(f"  MIDIs procesados:      {len(rows)}")
print(f"  Error carga MIDI:      {stats['midi_load_error']}")
print(f"  Sin tracks válidos:    {stats['no_valid_tracks']}")
print(f"  Demasiado cortos:      {stats['too_short']}")
print(f"  Tokens fuera de vocab: {stats['oov_tokens']}")
print(f"  ── ACEPTADOS:          {stats['accepted']}")

if index_rows:
    enc_lens = [r["enc_len"] for r in index_rows]
    dec_lens = [r["dec_len"] for r in index_rows]
    print(f"\n  Longitud encoder — min:{min(enc_lens)}  max:{max(enc_lens)}  avg:{int(np.mean(enc_lens))}")
    print(f"  Longitud decoder — min:{min(dec_lens)}  max:{max(dec_lens)}  avg:{int(np.mean(dec_lens))}")

    print(f"\n  Distribución por género:")
    genre_c = defaultdict(int)
    for r in index_rows: genre_c[r["genre"]] += 1
    for g, c in sorted(genre_c.items(), key=lambda x: -x[1]):
        print(f"    {g:<15} {c}")

print(f"\n  Índice guardado en: {index_path}")
print("=" * 55)
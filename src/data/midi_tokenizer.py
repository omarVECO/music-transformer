# src/data/midi_tokenizer.py
"""
Convierte cada MIDI etiquetado en pares de secuencias de tokens.

MEJORAS respecto a la versión anterior:
- Representación event-based con TIME_SHIFT explícitos para modelar el tiempo de forma
  continua y evitar silencios excesivos.
- Cuantización temporal en bins de 1/32 de nota (= 1 tick con PPQ=8).
- Cap de TIME_SHIFT a MAX_TIME_SHIFT unidades; silencios largos se dividen en múltiples tokens.
- Normalización de tempo a 120 BPM antes de tokenizar para consistencia entre archivos.
- Detección y eliminación de regiones silenciosas > MAX_SILENCE_TICKS durante la preparación.
- Vocabulario ampliado con NOTE_ON, NOTE_OFF, TIME_SHIFT y VELOCITY explícitos.
- Codificación y decodificación simétricas: lo que se codifica se puede decodificar exactamente.
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
# Normalizamos todo a 120 BPM antes de tokenizar.
# Esto garantiza que TIME_SHIFT tenga semántica musical consistente entre archivos.
TARGET_TEMPO     = 120.0          # BPM canónico
PPQ              = 8              # subdivisiones por negra → 1 tick = 1/32 nota
TICKS_PER_BEAT   = PPQ
TICKS_PER_BAR    = PPQ * 4       # 4/4
SECONDS_PER_TICK = 60.0 / (TARGET_TEMPO * PPQ)  # = 0.03125 s/tick

MAX_BARS         = 32
MIN_BARS         = 4
MAX_PITCH        = 108
MIN_PITCH        = 28
VELOCITY_BINS    = [16, 32, 48, 64, 80, 96, 112, 127]

# TIME_SHIFT máximo por token (evita silencio infinito)
# 32 ticks = 1 compás a 120 BPM — silencios más largos se parten.
MAX_TIME_SHIFT   = 32

# Silencio máximo permitido entre notas (en ticks) antes de truncarlo.
# 4 compases a 120 BPM = 128 ticks. Los gaps mayores se recortan a este valor.
MAX_SILENCE_TICKS = 128

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
INSTS     = ["<INST_PIANO>", "<INST_BASS>", "<INST_GUITAR>"]
CHORDS    = [f"<CHORD_{n}_{q}>"
             for n in ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
             for q in ["MAJ","MIN","DIM","AUG","MAJ7","MIN7","DOM7","DIM7"]]

# ── Tokens event-based (NUEVOS / MODIFICADOS) ─────────────────
# NOTE_ON y NOTE_OFF para cada pitch válido
NOTE_ON_TOKENS  = [f"<NOTE_ON_{p}>"  for p in range(MIN_PITCH, MAX_PITCH + 1)]
NOTE_OFF_TOKENS = [f"<NOTE_OFF_{p}>" for p in range(MIN_PITCH, MAX_PITCH + 1)]

# TIME_SHIFT de 1 a MAX_TIME_SHIFT ticks
# 1 tick = 1/32 nota a 120 BPM = 31.25 ms
TIME_SHIFT_TOKENS = [f"<TIME_SHIFT_{i}>" for i in range(1, MAX_TIME_SHIFT + 1)]

# VELOCITY discretizado
VELOCITY_TOKENS = [f"<VELOCITY_{v}>" for v in VELOCITY_BINS]

# ── Tokens de estructura (mantenidos para compatibilidad con el encoder) ──
# Se usan solo en el encoder (contexto de la melodía).
BARS      = [f"<BAR_{i}>" for i in range(1, MAX_BARS + 1)]
BEATS     = [f"<BEAT_{i}>" for i in range(1, 5)]
POSITIONS = [f"<POS_{i}>" for i in range(TICKS_PER_BAR)]
# Tokens de duración y pitch clásicos — mantenidos para retrocompatibilidad con el decoder
PITCHES   = [f"<PITCH_{i}>" for i in range(MIN_PITCH, MAX_PITCH + 1)] + ["<REST>"]
DURATIONS = ["<DUR_1>","<DUR_2>","<DUR_3>","<DUR_4>","<DUR_6>",
             "<DUR_8>","<DUR_12>","<DUR_16>","<DUR_T2>","<DUR_T4>"]
VELOCITIES_OLD = [f"<VEL_{v}>" for v in VELOCITY_BINS]

ALL_TOKENS = (
    SPECIAL + GENRES + MOODS + ENERGIES + TIMESIGS + KEYS + TEMPOS +
    INSTS + CHORDS + BARS + BEATS + POSITIONS +
    PITCHES + DURATIONS + VELOCITIES_OLD +          # tokens clásicos (encoder)
    NOTE_ON_TOKENS + NOTE_OFF_TOKENS +              # tokens event-based (decoder)
    TIME_SHIFT_TOKENS + VELOCITY_TOKENS             # tokens de tiempo y velocidad
)

# Eliminar duplicados preservando orden
seen = set()
ALL_TOKENS_DEDUP = []
for t in ALL_TOKENS:
    if t not in seen:
        seen.add(t)
        ALL_TOKENS_DEDUP.append(t)
ALL_TOKENS = ALL_TOKENS_DEDUP

TOKEN2ID = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}

# Guardar vocabulario
with open(TOKENS_DIR / "vocabulary.json", "w") as f:
    json.dump({"token2id": TOKEN2ID, "id2token": {str(k): v for k,v in ID2TOKEN.items()}}, f)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def quantize_velocity(vel: int) -> str:
    """Devuelve el token VELOCITY_<bin> más cercano (event-based)."""
    for bin_val in VELOCITY_BINS:
        if vel <= bin_val:
            return f"<VELOCITY_{bin_val}>"
    return "<VELOCITY_127>"

def quantize_velocity_old(vel: int) -> str:
    """Token VEL_<bin> clásico (para compatibilidad con el encoder)."""
    for bin_val in VELOCITY_BINS:
        if vel <= bin_val:
            return f"<VEL_{bin_val}>"
    return "<VEL_127>"

def quantize_duration(ticks: int) -> str:
    """Mapea ticks a token de duración clásico (encoder)."""
    DUR_MAP = {1:"<DUR_1>", 2:"<DUR_2>", 3:"<DUR_3>", 4:"<DUR_4>",
               6:"<DUR_6>", 8:"<DUR_8>", 12:"<DUR_12>", 16:"<DUR_16>"}
    if ticks <= 0:
        return "<DUR_1>"
    closest = min(DUR_MAP.keys(), key=lambda x: abs(x - ticks))
    return DUR_MAP[closest]

def quantize_tempo(bpm: float) -> str:
    thresholds = [70, 90, 110, 130, 150, 170, 190]
    labels = ["<TEMPO_60>","<TEMPO_80>","<TEMPO_100>","<TEMPO_120>",
              "<TEMPO_140>","<TEMPO_160>","<TEMPO_180>","<TEMPO_200>"]
    for t, label in zip(thresholds, labels):
        if bpm < t:
            return label
    return "<TEMPO_200>"

def energy_to_token(energy: float) -> str:
    if energy < 0.33:   return "<ENERGY_LOW>"
    if energy < 0.66:   return "<ENERGY_MED>"
    return "<ENERGY_HIGH>"

def seconds_to_ticks_normalized(t: float, original_tempo: float) -> int:
    """
    Convierte segundos (a tempo original) a ticks normalizados a TARGET_TEMPO.
    La normalización garantiza que 1 tick = 1/32 nota independientemente del tempo original.
    """
    # Primero convertimos a beats en el tempo original, luego a ticks en TARGET_TEMPO.
    beats = t * (original_tempo / 60.0)
    ticks = int(round(beats * PPQ))
    return max(0, ticks)

def emit_time_shift(delta_ticks: int, tokens: list) -> None:
    """
    Emite uno o varios TIME_SHIFT para cubrir delta_ticks.
    Si delta_ticks > MAX_TIME_SHIFT, lo divide en múltiples tokens.
    Si delta_ticks > MAX_SILENCE_TICKS, lo recorta para evitar silencios masivos.

    CRÍTICO: esto evita que la generación quede atrapada emitiendo
    TIME_SHIFT infinitos durante silencios en el entrenamiento.
    """
    # Recortar silencio excesivo
    delta_ticks = min(delta_ticks, MAX_SILENCE_TICKS)
    if delta_ticks <= 0:
        return
    while delta_ticks > 0:
        shift = min(delta_ticks, MAX_TIME_SHIFT)
        tokens.append(f"<TIME_SHIFT_{shift}>")
        delta_ticks -= shift

def detect_key(pm) -> str:
    """Estima la tonalidad usando el perfil de Krumhansl-Schmuckler."""
    pitch_classes = np.zeros(12)
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                pitch_classes[note.pitch % 12] += note.end - note.start

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

def detect_chord(notes_in_beat: list):
    if not notes_in_beat:
        return None
    pcs = list({n % 12 for n in notes_in_beat})
    note_names = ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
    CHORD_TEMPLATES = {
        "MAJ": [0,4,7], "MIN": [0,3,7], "DIM": [0,3,6], "AUG": [0,4,8],
        "MAJ7": [0,4,7,11], "MIN7": [0,3,7,10], "DOM7": [0,4,7,10], "DIM7": [0,3,6,9],
    }
    best_root, best_quality, best_score = 0, "MAJ", -1
    for root in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            chord_pcs = {(root + i) % 12 for i in intervals}
            overlap   = len(chord_pcs & set(pcs))
            score     = overlap / max(len(chord_pcs), len(pcs))
            if score > best_score:
                best_score, best_root, best_quality = score, root, quality
    if best_score < 0.4:
        return None
    return f"<CHORD_{note_names[best_root]}_{best_quality}>"

def select_tracks(pm):
    """Selecciona la pista melódica y la pista de acompañamiento."""
    non_drum = [i for i in pm.instruments if not i.is_drum and len(i.notes) > 0]
    if len(non_drum) < 2:
        return None, None

    melody_candidates = []
    accomp_candidates = []

    for inst in non_drum:
        cls = pretty_midi.program_to_instrument_class(inst.program)
        avg_pitch  = np.mean([n.pitch for n in inst.notes])
        note_count = len(inst.notes)
        if cls in MELODY_CLASSES:
            melody_candidates.append((inst, avg_pitch, note_count))
        if cls in ACCOMP_CLASSES:
            accomp_candidates.append((inst, avg_pitch, note_count))

    if not melody_candidates or not accomp_candidates:
        return None, None

    melody = max(melody_candidates, key=lambda x: x[1])[0]
    accomp_candidates = [(i, p, c) for i, p, c in accomp_candidates if i != melody]
    if not accomp_candidates:
        return None, None
    accomp = min(accomp_candidates, key=lambda x: x[1])[0]
    return melody, accomp

def inst_to_token(inst) -> str:
    cls = pretty_midi.program_to_instrument_class(inst.program)
    if cls == "Bass":   return "<INST_BASS>"
    if cls == "Guitar": return "<INST_GUITAR>"
    return "<INST_PIANO>"

def filter_silent_notes(notes: list, tempo_bpm: float) -> list:
    """
    Elimina notas con duración ≤ 0 y reordena por tiempo de inicio.
    También filtra pitches fuera de rango.
    """
    valid = [n for n in notes
             if n.end > n.start
             and MIN_PITCH <= n.pitch <= MAX_PITCH]
    return sorted(valid, key=lambda n: n.start)

# ─────────────────────────────────────────────────────────────
# Tokenización del encoder (melodía): representación posicional BAR/POS
# Se mantiene igual que la versión original para compatibilidad.
# ─────────────────────────────────────────────────────────────

def notes_to_token_sequence_encoder(inst, pm, tempo_bpm: float, key_token: str,
                                     genre: str, mood: str, energy: float,
                                     inst_token: str) -> list:
    """
    Encoder: representación posicional (BAR / BEAT / POS / PITCH / DUR / VEL).
    Normaliza internamente a TARGET_TEMPO.
    """
    if tempo_bpm < 10.0:
        tempo_bpm = 120.0

    end_time   = pm.get_end_time()
    total_bars = min(
        int(end_time * (TARGET_TEMPO / 60.0) / TICKS_PER_BAR * PPQ),
        MAX_BARS
    )
    if total_bars < MIN_BARS:
        return None

    notes = filter_silent_notes(inst.notes, tempo_bpm)
    if not notes:
        return None

    # Indexar notas por tick normalizado
    notes_by_tick = defaultdict(list)
    for note in notes:
        tick  = seconds_to_ticks_normalized(note.start, tempo_bpm)
        dur_t = seconds_to_ticks_normalized(note.end - note.start, tempo_bpm)
        dur_t = max(1, dur_t)
        notes_by_tick[tick].append((note.pitch, dur_t, note.velocity))

    # Re-anchor notes that start beyond bar 0 (e.g. late-starting tracks in long files)
    if notes_by_tick:
        min_tick    = min(notes_by_tick.keys())
        tick_offset = (min_tick // TICKS_PER_BAR) * TICKS_PER_BAR
        if tick_offset > 0:
            notes_by_tick = defaultdict(list, {
                t - tick_offset: v for t, v in notes_by_tick.items()
            })
            # Recompute total_bars against the shifted range
            max_shifted = max(notes_by_tick.keys())
            total_bars  = min(
                max(int(np.ceil((max_shifted + 1) / TICKS_PER_BAR)), MIN_BARS),
                MAX_BARS
            )

    tokens = ["<SOS>", "<TIMESIG_4_4>", key_token, quantize_tempo(tempo_bpm)]

    prev_beat = -1
    for bar_idx in range(total_bars):
        bar_token      = f"<BAR_{bar_idx + 1}>"
        bar_start_tick = bar_idx * TICKS_PER_BAR
        bar_notes = []
        for tick_offset in range(TICKS_PER_BAR):
            tick = bar_start_tick + tick_offset
            bar_notes += [p for p, d, v in notes_by_tick.get(tick, [])]
        chord_token = detect_chord(bar_notes)

        bar_emitted   = False
        chord_emitted = False

        for pos in range(TICKS_PER_BAR):
            tick      = bar_start_tick + pos
            beat_idx  = pos // PPQ
            beat_token = f"<BEAT_{beat_idx + 1}>"
            pos_token  = f"<POS_{pos}>"

            if tick not in notes_by_tick:
                continue

            if not bar_emitted:
                tokens.append(bar_token)
                bar_emitted = True
            if not chord_emitted and chord_token:
                tokens.append(chord_token)
                chord_emitted = True
            if beat_idx != prev_beat:
                tokens.append(beat_token)
                prev_beat = beat_idx

            tokens.append(pos_token)

            for pitch, dur_ticks, velocity in notes_by_tick[tick]:
                tokens.append(inst_token)
                tokens.append(f"<PITCH_{pitch}>")
                tokens.append(quantize_duration(dur_ticks))
                tokens.append(quantize_velocity_old(velocity))

    tokens.append("<EOS>")
    return tokens

# ─────────────────────────────────────────────────────────────
# Tokenización del decoder (acompañamiento): representación event-based
# NUEVO: NOTE_ON / NOTE_OFF / TIME_SHIFT / VELOCITY
# Esto modela el tiempo de forma continua y evita silencios excesivos.
# ─────────────────────────────────────────────────────────────

def notes_to_token_sequence_decoder(inst, pm, tempo_bpm: float, key_token: str,
                                     genre: str, mood: str, energy: float,
                                     inst_token: str) -> list:
    """
    Decoder: representación event-based con BEAT tokens.
    Secuencia: BEAT_X → VELOCITY → NOTE_ON → TIME_SHIFT → NOTE_OFF → ...

    Phase-2 changes vs previous version:
    - BEAT tokens (<BEAT_1>..<BEAT_4>) emitted at every beat boundary (every
      TICKS_PER_BEAT ticks) even in silence, giving the model explicit rhythmic
      position conditioning.
    - VELOCITY emitted before EVERY NOTE_ON (not only on change), so the model
      must learn to express dynamics on each note.
    - 4-bar silent-window rejection: returns None if any fixed 4-bar window
      contains no NOTE_ON events.
    """
    if tempo_bpm < 10.0:
        tempo_bpm = 120.0

    end_time   = pm.get_end_time()
    total_bars = min(
        int(end_time * (TARGET_TEMPO / 60.0) / TICKS_PER_BAR * PPQ),
        MAX_BARS
    )
    if total_bars < MIN_BARS:
        return None

    notes = filter_silent_notes(inst.notes, tempo_bpm)
    if not notes:
        return None

    # Limitar notas al segmento de MAX_BARS
    max_seconds = total_bars * TICKS_PER_BAR * SECONDS_PER_TICK * (tempo_bpm / TARGET_TEMPO)
    notes = [n for n in notes if n.start < max_seconds]
    if not notes:
        return None

    # Construir lista de eventos: (tick, tipo, pitch, velocity)
    events = []
    for note in notes:
        tick_on  = seconds_to_ticks_normalized(note.start, tempo_bpm)
        tick_off = seconds_to_ticks_normalized(note.end,   tempo_bpm)
        tick_off = max(tick_off, tick_on + 1)
        events.append((tick_on,  "on",  note.pitch, note.velocity))
        events.append((tick_off, "off", note.pitch, 0))

    # Ordenar por tick; empates: NOTE_OFF antes que NOTE_ON
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    # Reject sequences with any fully-silent 4-bar window
    for w in range(0, total_bars, 4):
        w_start = w * TICKS_PER_BAR
        w_end   = min((w + 4) * TICKS_PER_BAR, total_bars * TICKS_PER_BAR)
        if not any(e[1] == "on" and w_start <= e[0] < w_end for e in events):
            return None

    # Header de contexto
    tokens = [
        "<SOS>",
        f"<GENRE_{genre}>",
        f"<MOOD_{mood}>",
        energy_to_token(energy),
        inst_token,
    ]

    current_tick      = 0
    last_beat_global  = -1  # global beat index last emitted

    def maybe_emit_beat(tick: int) -> None:
        nonlocal last_beat_global
        if tick % TICKS_PER_BEAT == 0:
            bg = tick // TICKS_PER_BEAT
            if bg > last_beat_global:
                tokens.append(f"<BEAT_{(bg % 4) + 1}>")
                last_beat_global = bg

    # Emit BEAT_1 at tick 0
    maybe_emit_beat(0)

    for tick, event_type, pitch, velocity in events:
        if tick > current_tick:
            # Walk from current_tick to min(tick, current_tick+MAX_SILENCE_TICKS),
            # emitting BEAT tokens at every beat boundary and TIME_SHIFT chunks.
            cap = min(tick, current_tick + MAX_SILENCE_TICKS)
            pos = current_tick
            while pos < cap:
                maybe_emit_beat(pos)
                next_beat = (pos // TICKS_PER_BEAT + 1) * TICKS_PER_BEAT
                step_end  = min(next_beat, cap)
                shift     = step_end - pos
                while shift > 0:
                    s = min(shift, MAX_TIME_SHIFT)
                    tokens.append(f"<TIME_SHIFT_{s}>")
                    shift -= s
                pos = step_end
            current_tick = pos
            maybe_emit_beat(current_tick)

        else:
            maybe_emit_beat(tick)

        if event_type == "on":
            tokens.append(quantize_velocity(velocity))  # Always emit VELOCITY
            tokens.append(f"<NOTE_ON_{pitch}>")
        else:
            tokens.append(f"<NOTE_OFF_{pitch}>")

    tokens.append("<EOS>")
    return tokens


def notes_to_token_sequence(inst, pm, tempo_bpm: float, key_token: str,
                             genre: str, mood: str, energy: float,
                             inst_token: str, is_encoder: bool = True) -> list:
    """
    Punto de entrada unificado, compatible con la API original.
    - is_encoder=True  → representación posicional (BAR/POS) para el encoder.
    - is_encoder=False → representación event-based (NOTE_ON/OFF/TIME_SHIFT) para el decoder.
    """
    if is_encoder:
        return notes_to_token_sequence_encoder(
            inst, pm, tempo_bpm, key_token, genre, mood, energy, inst_token
        )
    else:
        return notes_to_token_sequence_decoder(
            inst, pm, tempo_bpm, key_token, genre, mood, energy, inst_token
        )


# ─────────────────────────────────────────────────────────────
# Decodificación: tokens event-based → MIDI
# ─────────────────────────────────────────────────────────────

def decode_event_tokens_to_midi(token_ids: list, instrument_program: int = 33) -> "pretty_midi.PrettyMIDI":
    """
    Convierte una lista de IDs de tokens (decoder, event-based) en un objeto PrettyMIDI.

    IMPORTANTE: consistencia con la codificación.
    - VELOCITY_<v>  → establece velocidad activa.
    - NOTE_ON_<p>   → inicia nota con velocidad activa y tick actual.
    - NOTE_OFF_<p>  → cierra nota (registra en el MIDI).
    - TIME_SHIFT_<n>→ avanza el tiempo n ticks.
    - Todo lo demás se ignora.
    """
    pm   = pretty_midi.PrettyMIDI(initial_tempo=TARGET_TEMPO)
    inst = pretty_midi.Instrument(program=instrument_program)

    current_tick = 0
    active_vel   = 64      # velocidad por defecto
    open_notes   = {}      # pitch → tick_on

    skip_prefixes = (
        "<GENRE_", "<MOOD_", "<ENERGY_", "<INST_",
        "<TIMESIG_", "<KEY_", "<TEMPO_", "<CHORD_", "<BEAT_",
        "<BAR_", "<POS_", "<PITCH_", "<DUR_", "<VEL_", "<REST>",
    )

    for tid in token_ids:
        tok = ID2TOKEN.get(tid, "<UNK>")

        if tok in ("<SOS>", "<EOS>", "<PAD>", "<UNK>", "<SEP>", "<MASK>"):
            if tok == "<EOS>":
                break
            continue

        if any(tok.startswith(p) for p in skip_prefixes):
            continue

        if tok.startswith("<VELOCITY_"):
            try:
                active_vel = int(tok[len("<VELOCITY_"):-1])
            except ValueError:
                pass

        elif tok.startswith("<TIME_SHIFT_"):
            try:
                shift = int(tok[len("<TIME_SHIFT_"):-1])
                current_tick += max(0, shift)
            except ValueError:
                pass

        elif tok.startswith("<NOTE_ON_"):
            try:
                pitch = int(tok[len("<NOTE_ON_"):-1])
                if MIN_PITCH <= pitch <= MAX_PITCH:
                    open_notes[pitch] = (current_tick, active_vel)
            except ValueError:
                pass

        elif tok.startswith("<NOTE_OFF_"):
            try:
                pitch = int(tok[len("<NOTE_OFF_"):-1])
                if pitch in open_notes:
                    tick_on, vel = open_notes.pop(pitch)
                    t_start = tick_on  * SECONDS_PER_TICK
                    t_end   = max(current_tick, tick_on + 1) * SECONDS_PER_TICK
                    inst.notes.append(pretty_midi.Note(
                        velocity=min(127, max(1, vel)),
                        pitch=pitch,
                        start=t_start,
                        end=t_end,
                    ))
            except ValueError:
                pass

    # Cerrar notas que quedaron abiertas (seguridad)
    for pitch, (tick_on, vel) in open_notes.items():
        t_start = tick_on * SECONDS_PER_TICK
        t_end   = (current_tick + 1) * SECONDS_PER_TICK
        inst.notes.append(pretty_midi.Note(
            velocity=min(127, max(1, vel)),
            pitch=pitch,
            start=t_start,
            end=t_end,
        ))

    pm.instruments.append(inst)
    return pm


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

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

        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            stats["midi_load_error"] += 1
            continue

        if tempo_bpm < 10.0:
            estimated = pm.estimate_tempo()
            tempo_bpm = estimated if 30 < estimated < 300 else 120.0

        melody_inst, accomp_inst = select_tracks(pm)
        if melody_inst is None:
            stats["no_valid_tracks"] += 1
            continue

        key_token  = detect_key(pm)
        inst_token = inst_to_token(accomp_inst)

        # Encoder: representación posicional
        enc_tokens = notes_to_token_sequence(
            melody_inst, pm, tempo_bpm, key_token,
            genre, mood, energy, inst_to_token(melody_inst),
            is_encoder=True
        )
        if enc_tokens is None:
            stats["too_short"] += 1
            continue

        # Decoder: representación event-based (NUEVO)
        dec_tokens = notes_to_token_sequence(
            accomp_inst, pm, tempo_bpm, key_token,
            genre, mood, energy, inst_token,
            is_encoder=False
        )
        if dec_tokens is None:
            stats["too_short"] += 1
            continue

        # Verificar tokens en vocabulario
        enc_valid = all(t in TOKEN2ID for t in enc_tokens)
        dec_valid = all(t in TOKEN2ID for t in dec_tokens)
        if not enc_valid or not dec_valid:
            stats["oov_tokens"] += 1
            for t in enc_tokens + dec_tokens:
                if t not in TOKEN2ID:
                    print(f"  OOV: {t}")
            continue

        # Estadística de balance de tokens en el decoder
        n_time_shift = sum(1 for t in dec_tokens if t.startswith("<TIME_SHIFT_"))
        n_note_on    = sum(1 for t in dec_tokens if t.startswith("<NOTE_ON_"))
        total_dec    = len(dec_tokens)

        # Rechazar secuencias dominadas por TIME_SHIFT (> 40%)
        if total_dec > 0 and n_time_shift / total_dec > 0.4:
            stats["too_much_silence"] += 1
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
            "msd_id":     msd_id,
            "token_path": str(out_path),
            "genre":      genre,
            "mood":       mood,
            "enc_len":    len(enc_tokens),
            "dec_len":    len(dec_tokens),
        })
        stats["accepted"] += 1

    # Guardar índice
    index_path = TOKENS_DIR / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["msd_id","token_path","genre","mood","enc_len","dec_len"])
        writer.writeheader()
        writer.writerows(index_rows)

    print("\n" + "=" * 55)
    print("  REPORTE TOKENIZACIÓN")
    print("=" * 55)
    print(f"  MIDIs procesados:      {len(rows)}")
    print(f"  Error carga MIDI:      {stats['midi_load_error']}")
    print(f"  Sin tracks válidos:    {stats['no_valid_tracks']}")
    print(f"  Demasiado cortos:      {stats['too_short']}")
    print(f"  Demasiado silencio:    {stats['too_much_silence']}")
    print(f"  Tokens fuera de vocab: {stats['oov_tokens']}")
    print(f"  ── ACEPTADOS:          {stats['accepted']}")

    if index_rows:
        enc_lens = [r["enc_len"] for r in index_rows]
        dec_lens = [r["dec_len"] for r in index_rows]
        print(f"\n  Longitud encoder — min:{min(enc_lens)}  max:{max(enc_lens)}  avg:{int(np.mean(enc_lens))}")
        print(f"  Longitud decoder — min:{min(dec_lens)}  max:{max(dec_lens)}  avg:{int(np.mean(dec_lens))}")
        print(f"\n  Vocabulario total: {len(TOKEN2ID)} tokens")

        genre_c = defaultdict(int)
        for r in index_rows: genre_c[r["genre"]] += 1
        print(f"\n  Distribución por género:")
        for g, c in sorted(genre_c.items(), key=lambda x: -x[1]):
            print(f"    {g:<15} {c}")

    print(f"\n  Índice guardado en: {index_path}")
    print("=" * 55)
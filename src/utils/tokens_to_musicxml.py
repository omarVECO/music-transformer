# src/utils/tokens_to_musicxml.py
"""
Conversión de tokens → MusicXML de dos pentagramas (melodía + acompañamiento).

CAMBIOS respecto a la versión anterior:
- El decoder ahora usa representación event-based (NOTE_ON/NOTE_OFF/TIME_SHIFT/VELOCITY)
  en lugar de la representación posicional (BAR/POS/PITCH/DUR/VEL).
  Se añade parse_event_tokens() para manejar este formato.
- parse_tokens() se mantiene intacto para el encoder (representación posicional).
- tokens_to_musicxml() ahora acepta dec_token_ids (lista de ints) en lugar de
  dec_tokens (lista de strings), y convierte internamente usando ID2TOKEN.
  Esto es coherente con lo que devuelve generate() en inference.py.
- La decodificación event-based construye eventos con posición absoluta en ticks
  normalizados, igual que hace decode_event_tokens_to_midi() en el tokenizador.
- Se mantiene toda la lógica de construcción de pentagramas music21 sin cambios.
"""
import music21
from music21 import stream, note, chord, meter, tempo, key, dynamics
from music21 import harmony, instrument, clef
from collections import defaultdict

from data.midi_tokenizer import (
    ID2TOKEN, PPQ as TOKENIZER_PPQ,
    SECONDS_PER_TICK, TARGET_TEMPO,
    MIN_PITCH, MAX_PITCH,
)

# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────

# PPQ del encoder (representación posicional): 8 subdivisiones/negra = 1/32 nota.
# Coincide con PPQ definido en midi_tokenizer.py.
PPQ_ENC       = 8
TICKS_PER_BAR = 32   # 4/4 con PPQ_ENC=8

# PPQ del decoder (event-based): 8 subdivisiones/negra = 1/32 nota
# Coincide con TOKENIZER_PPQ definido en midi_tokenizer.py
PPQ_DEC = TOKENIZER_PPQ  # = 8

VELOCITY_DYNAMICS = {
    16:"ppp", 32:"pp", 48:"p",  64:"mp",
    80:"mf",  96:"f", 112:"ff", 127:"fff",
}

DUR_TOKEN_TICKS = {
    "<DUR_1>":1,  "<DUR_2>":2,  "<DUR_3>":3,  "<DUR_4>":4,
    "<DUR_6>":6,  "<DUR_8>":8,  "<DUR_12>":12, "<DUR_16>":16,
    "<DUR_T2>":2, "<DUR_T4>":4,
}

TEMPO_TOKEN_BPM = {
    "<TEMPO_60>":60,   "<TEMPO_80>":80,   "<TEMPO_100>":100,
    "<TEMPO_120>":120, "<TEMPO_140>":140, "<TEMPO_160>":160,
    "<TEMPO_180>":180, "<TEMPO_200>":200,
}


# ─────────────────────────────────────────────────────────────
# Parser del ENCODER (representación posicional — sin cambios)
# ─────────────────────────────────────────────────────────────

def ticks_to_quarterLength_enc(ticks: int) -> float:
    """Encoder: convierte ticks (PPQ=8, 1/32 nota=1) a quarter notes."""
    return ticks / PPQ_ENC


def parse_tokens(tokens: list) -> tuple:
    """
    Convierte tokens del ENCODER (BAR/POS/PITCH/DUR/VEL) a eventos con
    posición absoluta en ticks.

    Returns:
        (events, meta, bar_chords)
        events:     lista de dicts {bar, pos_ticks, pitch, dur_ticks, velocity}
        meta:       dict con tempo, key, mode, timesig, genre, mood
        bar_chords: dict {bar_idx: (root_str, quality_str)}
    """
    meta = {"tempo": 120, "key": "C", "mode": "major",
            "timesig": "4/4", "genre": None, "mood": None}

    events     = []
    bar_chords = {}

    current_bar   = 1
    current_pos   = 0
    current_pitch = None
    current_dur   = None
    current_vel   = 80

    def flush():
        nonlocal current_pitch, current_dur
        if current_pitch is None or current_dur is None:
            return
        events.append({
            "bar":       current_bar,
            "pos_ticks": current_pos,
            "pitch":     current_pitch,
            "dur_ticks": current_dur,
            "velocity":  current_vel,
        })
        current_pitch = None
        current_dur   = None

    for tok in tokens:
        if tok.startswith("<TEMPO_"):
            meta["tempo"] = TEMPO_TOKEN_BPM.get(tok, 120)
        elif tok.startswith("<TIMESIG_"):
            p = tok[9:-1].split("_")
            meta["timesig"] = f"{p[0]}/{p[1]}"
        elif tok.startswith("<KEY_"):
            p = tok[5:-1].split("_")
            meta["key"]  = p[0].replace("s", "#")
            meta["mode"] = "major" if p[1] == "MAJ" else "minor"
        elif tok.startswith("<GENRE_"):
            meta["genre"] = tok[7:-1].capitalize()
        elif tok.startswith("<MOOD_"):
            meta["mood"] = tok[6:-1].capitalize()
        elif tok.startswith("<BAR_"):
            flush()
            current_bar = int(tok[5:-1])
            current_pos = 0
        elif tok.startswith("<POS_"):
            flush()
            current_pos = int(tok[5:-1])
        elif tok.startswith("<INST_"):
            if not meta.get("inst"):
                meta["inst"] = tok[6:-1].upper()   # "PIANO", "BASS", "GUITAR"
        elif tok.startswith("<CHORD_"):
            inner = tok[7:-1].split("_")
            root  = inner[0].replace("s", "#")
            qual  = inner[1] if len(inner) > 1 else "MAJ"
            bar_chords[current_bar] = (root, qual)
        elif tok.startswith("<PITCH_"):
            flush()
            current_pitch = int(tok[7:-1])
            current_dur   = None
            current_vel   = 80
        elif tok == "<REST>":
            flush()
            current_pitch = None
        elif tok in DUR_TOKEN_TICKS:
            current_dur = DUR_TOKEN_TICKS[tok]
        elif tok.startswith("<VEL_"):
            try:
                current_vel = int(tok[5:-1])
            except ValueError:
                pass
            flush()
        elif tok in ("<SOS>", "<EOS>", "<PAD>", "<UNK>"):
            pass

    flush()
    return events, meta, bar_chords


# ─────────────────────────────────────────────────────────────
# Parser del DECODER (representación event-based — NUEVO)
# ─────────────────────────────────────────────────────────────

def ticks_to_quarterLength_dec(ticks: int) -> float:
    """Decoder: convierte ticks (PPQ=8, 1/32 nota=1) a quarter notes."""
    return ticks / PPQ_DEC


def parse_event_tokens(token_ids: list) -> tuple:
    """
    Convierte token IDs del DECODER (NOTE_ON/NOTE_OFF/TIME_SHIFT/VELOCITY) a
    eventos con posición absoluta en ticks.

    La secuencia event-based tiene la forma:
        VELOCITY_<v> NOTE_ON_<p> ... TIME_SHIFT_<n> ... NOTE_OFF_<p> ...

    NOTA: 1 tick = 1/32 nota a 120 BPM con PPQ_DEC=8.

    Returns:
        (events, meta, bar_chords)
        events:     lista de dicts {bar, pos_ticks, pitch, dur_ticks, velocity}
                    con posición en TICKS (PPQ_DEC), que luego se convierten a
                    quarter notes con ticks_to_quarterLength_dec().
        meta:       dict con info de contexto del header (genre, mood, inst)
        bar_chords: dict vacío (el decoder event-based no emite acordes por compás)
    """
    meta = {"tempo": 120, "key": "C", "mode": "major",
            "timesig": "4/4", "genre": None, "mood": None}

    events     = []
    bar_chords = {}

    current_tick = 0
    active_vel   = 64      # velocidad por defecto si no hay VELOCITY previo
    open_notes   = {}      # pitch → (tick_on, velocity)

    # Prefijos que se ignoran (tokens de contexto del header)
    skip_prefixes = (
        "<GENRE_", "<MOOD_", "<ENERGY_", "<INST_",
        "<TIMESIG_", "<KEY_", "<TEMPO_", "<CHORD_", "<BEAT_",
        "<BAR_", "<POS_",
        # Tokens del encoder que podrían colarse
        "<PITCH_", "<DUR_", "<VEL_", "<REST>",
    )

    for tid in token_ids:
        tok = ID2TOKEN.get(tid, "<UNK>")

        if tok in ("<SOS>", "<PAD>", "<UNK>", "<SEP>", "<MASK>"):
            continue
        if tok == "<EOS>":
            break

        # Leer meta del header (primeros tokens del decoder)
        if tok.startswith("<GENRE_"):
            meta["genre"] = tok[7:-1].capitalize()
            continue
        if tok.startswith("<MOOD_"):
            meta["mood"] = tok[6:-1].capitalize()
            continue
        if tok.startswith("<INST_"):
            meta["inst"] = tok[6:-1].capitalize()
            continue

        # Ignorar el resto de tokens de contexto
        if any(tok.startswith(p) for p in skip_prefixes):
            continue

        # ── Tokens event-based ────────────────────────────────────────
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
                    dur_ticks = max(current_tick - tick_on, 1)
                    events.append({
                        "abs_tick": tick_on,       # posición absoluta en ticks PPQ_DEC
                        "dur_ticks": dur_ticks,
                        "pitch":     pitch,
                        "velocity":  min(127, max(1, vel)),
                    })
            except ValueError:
                pass

    # Cerrar notas que quedaron abiertas al final de la secuencia
    for pitch, (tick_on, vel) in open_notes.items():
        dur_ticks = max(current_tick - tick_on, 1)
        events.append({
            "abs_tick":  tick_on,
            "dur_ticks": dur_ticks,
            "pitch":     pitch,
            "velocity":  min(127, max(1, vel)),
        })

    # Convertir posición absoluta en ticks a (bar, pos_within_bar)
    # usando TICKS_PER_BAR adaptado al PPQ del decoder.
    # Con PPQ_DEC=8 y compás 4/4: ticks_per_bar = 8*4 = 32 ticks.
    ticks_per_bar_dec = PPQ_DEC * 4  # = 32 ticks por compás en 4/4

    structured_events = []
    for ev in events:
        bar_idx  = ev["abs_tick"] // ticks_per_bar_dec       # 0-indexed
        pos_tick = ev["abs_tick"] %  ticks_per_bar_dec
        structured_events.append({
            "bar":       bar_idx + 1,   # 1-indexed para compatibilidad con el encoder
            "pos_ticks": pos_tick,
            "pitch":     ev["pitch"],
            "dur_ticks": ev["dur_ticks"],
            "velocity":  ev["velocity"],
        })

    return structured_events, meta, bar_chords


# ─────────────────────────────────────────────────────────────
# Construcción de pentagramas music21
# ─────────────────────────────────────────────────────────────

def _chord_symbol_str(root: str, qual: str) -> str:
    """Convierte (root, quality) a string de ChordSymbol para music21."""
    return (root + qual
        .replace("MAJ7", "maj7").replace("MIN7", "m7")
        .replace("DOM7", "7").replace("DIM7", "dim7")
        .replace("MAJ", "").replace("MIN", "m")
        .replace("DIM", "dim").replace("AUG", "aug"))


def _snap_ql(ql: float, minimum: float = 0.0) -> float:
    """
    Rounds a quarterLength to the nearest 1/16-note grid point that music21
    can represent unambiguously.  `minimum` lets callers enforce a floor:
      - positions use minimum=0  (beat 1 onset must be exactly 0)
      - durations use minimum=0.0625  (no zero-length notes)
    """
    grid = 0.0625
    snapped = round(round(ql / grid) * grid, 6)
    return max(snapped, minimum)


def _snap_to_valid_ql(ql: float) -> float:
    """Snap a duration to the 1/16-note grid; never returns 0."""
    return _snap_ql(ql, minimum=0.0625)


def events_to_part(events: list, meta: dict, bar_chords: dict,
                   part_name: str, inst_obj,
                   ppq: int = PPQ_ENC) -> stream.Part:
    """
    Construye un music21 Part a partir de una lista de eventos.

    Args:
        events:     Lista de dicts {bar, pos_ticks, pitch, dur_ticks, velocity}.
        meta:       Dict con tempo, key, mode, timesig.
        bar_chords: Dict {bar: (root, qual)}.
        part_name:  Nombre de la parte.
        inst_obj:   Objeto music21 Instrument.
        ppq:        Resolución: PPQ_ENC=8 (encoder) o PPQ_DEC=8 (decoder).
    """
    def to_ql(ticks: int) -> float:
        return ticks / ppq

    p = stream.Part()
    p.id       = part_name
    p.partName = part_name

    ts_num     = int(meta["timesig"].split("/")[0])
    bar_dur_qn = float(ts_num)   # 4.0 para 4/4

    inst_class   = type(inst_obj).__name__ if inst_obj else ""
    bass_classes = ("ElectricBass", "Bass", "BassGuitar", "Contrabass", "Tuba")
    use_bass_clef = inst_class in bass_classes

    # Insert header elements at absolute position 0
    if inst_obj:
        p.insert(0, inst_obj)
    p.insert(0, meter.TimeSignature(meta["timesig"]))
    p.insert(0, key.Key(meta["key"], meta["mode"]))
    p.insert(0, clef.BassClef() if use_bass_clef else clef.TrebleClef())

    if not events:
        p.makeMeasures(inPlace=True)
        return p

    prev_dynamic = None

    for ev in events:
        # Absolute quarter-note position from bar + position-within-bar
        abs_ql = (ev["bar"] - 1) * bar_dur_qn + to_ql(ev["pos_ticks"])
        abs_ql = max(0.0, abs_ql)

        dur_ql = max(to_ql(ev["dur_ticks"]), 0.0625)

        try:
            d = music21.duration.Duration(quarterLength=dur_ql)
        except Exception:
            d = music21.duration.Duration(quarterLength=0.25)

        n = note.Note(ev["pitch"])
        n.duration        = d
        n.volume.velocity = max(1, min(127, ev["velocity"]))

        # Dynamics (only when velocity bucket changes)
        vel_bin = min(VELOCITY_DYNAMICS.keys(),
                      key=lambda x: abs(x - ev["velocity"]))
        dyn_str = VELOCITY_DYNAMICS[vel_bin]
        if dyn_str != prev_dynamic:
            p.insert(abs_ql, dynamics.Dynamic(dyn_str))
            prev_dynamic = dyn_str

        p.insert(abs_ql, n)

    # Chord symbols at bar offsets
    for bar_idx, (root, qual) in bar_chords.items():
        bar_ql = (bar_idx - 1) * bar_dur_qn
        try:
            p.insert(bar_ql, harmony.ChordSymbol(_chord_symbol_str(root, qual)))
        except Exception:
            pass

    # Convert flat stream → measures with automatic ties for cross-bar notes
    p.makeMeasures(inPlace=True)
    try:
        p.makeNotation(inPlace=True)
    except Exception:
        pass

    return p


# ─────────────────────────────────────────────────────────────
# Función principal (API pública)
# ─────────────────────────────────────────────────────────────

def tokens_to_musicxml(
    enc_tokens: list,
    dec_token_ids: list,
    output_path: str,
    tempo_bpm: float = 120.0,
    melody_name: str = "Melodía",
    accomp_name: str = "Acompañamiento",
) -> "music21.stream.Score":
    """
    Convierte tokens del encoder y IDs del decoder en un MusicXML de dos pentagramas.

    Args:
        enc_tokens:    Lista de strings de tokens del encoder (BAR/POS/PITCH/DUR/VEL).
        dec_token_ids: Lista de ints — salida directa de generate() en inference.py.
                       Contiene NOTE_ON/NOTE_OFF/TIME_SHIFT/VELOCITY en formato event-based.
        output_path:   Ruta de salida (.xml).
        tempo_bpm:     Tempo real del MIDI original (para ajustar el marcador de tempo).
        melody_name:   Nombre de la parte melódica en la partitura.
        accomp_name:   Nombre de la parte de acompañamiento.

    Returns:
        Objeto music21.stream.Score (también guardado en disco).
    """
    # ── Parsear encoder (representación posicional) ────────────────────
    enc_events, enc_meta, enc_chords = parse_tokens(enc_tokens)

    # ── Parsear decoder — detección automática de formato ─────────────
    # El modelo puede haber sido entrenado con el tokenizador antiguo (BAR/POS/PITCH)
    # o con el nuevo (NOTE_ON/NOTE_OFF/TIME_SHIFT). Detectamos el formato contando
    # qué tipo de tokens aparece en la secuencia generada.
    dec_tokens_str = [ID2TOKEN.get(tid, "<UNK>") for tid in dec_token_ids]

    n_note_on    = sum(1 for t in dec_tokens_str if t.startswith("<NOTE_ON_"))
    n_time_shift = sum(1 for t in dec_tokens_str if t.startswith("<TIME_SHIFT_"))
    n_pitch      = sum(1 for t in dec_tokens_str if t.startswith("<PITCH_"))
    n_bar        = sum(1 for t in dec_tokens_str if t.startswith("<BAR_"))

    event_based = (n_note_on + n_time_shift) > (n_pitch + n_bar)
    print(f"  Decoder — NOTE_ON:{n_note_on} TIME_SHIFT:{n_time_shift} "
          f"PITCH:{n_pitch} BAR:{n_bar} → "
          f"formato: {'event-based' if event_based else 'posicional (legado)'}")

    if event_based:
        dec_events, dec_meta, dec_chords = parse_event_tokens(dec_token_ids)
    else:
        # Modelo antiguo: genera tokens BAR/POS/PITCH/DUR/VEL
        dec_events, dec_meta, dec_chords = parse_tokens(dec_tokens_str)

    # ── Meta compartido (prioridad: encoder para tonalidad/compás) ─────
    shared_meta = enc_meta.copy()
    # Usar el tempo real del MIDI original, no el token (que está cuantizado)
    shared_meta["tempo"] = max(40, min(int(tempo_bpm), 220))
    # Completar genre/mood del decoder si el encoder no los tiene
    if not shared_meta.get("genre") and dec_meta.get("genre"):
        shared_meta["genre"] = dec_meta["genre"]
    if not shared_meta.get("mood") and dec_meta.get("mood"):
        shared_meta["mood"] = dec_meta["mood"]

    # ── Construir Score ────────────────────────────────────────────────
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata()

    title = "Acompañamiento Generado"
    if shared_meta.get("genre"):
        title += f" — {shared_meta['genre']}"
    if shared_meta.get("mood"):
        title += f" ({shared_meta['mood']})"
    score.metadata.title = title

    # ── Pentagrama melodía — instrumento detectado de los tokens del encoder ──
    _mel_inst_map = {
        "PIANO":  instrument.Piano(),
        "BASS":   instrument.ElectricBass(),
        "GUITAR": instrument.Guitar(),
    }
    mel_inst = _mel_inst_map.get(enc_meta.get("inst", "GUITAR"), instrument.Guitar())
    mel_inst.partName = melody_name
    mel_part = events_to_part(
        enc_events, shared_meta, enc_chords,
        melody_name, mel_inst, ppq=PPQ_ENC
    )

    # Marcador de tempo en el primer compás de la melodía
    mel_measures = mel_part.getElementsByClass(stream.Measure)
    if mel_measures:
        mel_measures[0].insert(0, tempo.MetronomeMark(number=shared_meta["tempo"]))

    # ── Pentagrama acompañamiento (decoder, PPQ=8) ─────────────────────
    # Determinar instrumento a partir de los tokens del decoder
    acc_inst = instrument.ElectricBass()  # por defecto
    for tid in dec_token_ids:
        tok = ID2TOKEN.get(tid, "")
        if tok == "<INST_PIANO>":
            acc_inst = instrument.Piano()
            break
        elif tok == "<INST_GUITAR>":
            acc_inst = instrument.Guitar()
            break
        elif tok == "<INST_BASS>":
            acc_inst = instrument.ElectricBass()
            break

    acc_inst.partName = accomp_name
    # ppq depende del formato detectado:
    # - event-based (nuevo tokenizador): PPQ_DEC=8 (1/32 nota = 1 tick)
    # - posicional  (modelo antiguo):    PPQ_ENC=16 (semicorchea = 1 tick)
    dec_ppq = PPQ_DEC if event_based else PPQ_ENC
    acc_part = events_to_part(
        dec_events, shared_meta, dec_chords,
        accomp_name, acc_inst, ppq=dec_ppq
    )

    score.append(mel_part)
    score.append(acc_part)

    # ── Exportar a MusicXML ────────────────────────────────────────────
    # Usamos score.write() en lugar de ScoreExporter directamente.
    # write() llama internamente a makeNotation (beams, ties, rests) de forma
    # segura y escribe el archivo XML bien formado.
    try:
        score.write("musicxml", fp=str(output_path))
    except Exception as e:
        # Fallback: si write() falla (por ejemplo por notas problemáticas),
        # desactivar makeBeams y exportar de todos modos.
        import warnings
        warnings.warn(f"write() falló ({e}), intentando exportación sin beams…")
        from music21.musicxml.m21ToXml import ScoreExporter
        import xml.etree.ElementTree as ET
        try:
            exporter = ScoreExporter(score)
            exporter.makeBeams = False          # desactivar beaming automático
            root_element = exporter.parse()
            xml_bytes    = ET.tostring(root_element, encoding="unicode",
                                       xml_declaration=False)
            xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes
            with open(str(output_path), "w", encoding="utf-8") as f:
                f.write(xml_str)
        except Exception as e2:
            raise RuntimeError(
                f"No se pudo exportar a MusicXML: {e} / {e2}"
            ) from e2

    # ── Reporte ────────────────────────────────────────────────────────
    mel_notes = len(mel_part.flatten().notes)
    acc_notes = len(acc_part.flatten().notes)
    mel_bars  = len(mel_part.getElementsByClass(stream.Measure))
    acc_bars  = len(acc_part.getElementsByClass(stream.Measure))

    print(f"  MusicXML guardado en: {output_path}")
    print(f"  Melodía:        {mel_notes} notas, {mel_bars} compases")
    print(f"  Acompañamiento: {acc_notes} notas, {acc_bars} compases")

    return score
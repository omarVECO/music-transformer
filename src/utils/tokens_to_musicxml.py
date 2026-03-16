# src/utils/tokens_to_musicxml.py
"""
Convierte secuencias de tokens (encoder + decoder) a MusicXML
usando music21 como intermediario.

Partitura resultante:
  - Pentagrama 1: melodía de entrada (encoder tokens)
  - Pentagrama 2: acompañamiento generado (decoder tokens)
  - Cifrado de acordes sobre el pentagrama 1
  - Indicaciones de dinámica por velocidad
  - Indicación de tempo al inicio

Uso:
  from utils.tokens_to_musicxml import tokens_to_musicxml

  tokens_to_musicxml(
      enc_tokens=["<SOS>", "<TIMESIG_4_4>", ...],
      dec_tokens=["<SOS>", "<GENRE_FUNK>", ...],
      output_path="partitura.xml"
  )
"""
import music21
from music21 import stream, note, chord, meter, tempo, key, dynamics
from music21 import expressions, harmony, instrument, layout
from collections import defaultdict
from music21 import stream, note, chord, meter, tempo, key, dynamics
from music21 import expressions, harmony, instrument, layout, clef

# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────
PPQ           = 16    # subdivisiones por negra
TICKS_PER_BAR = 64    # 4/4

# Mapa velocity → símbolo de dinámica music21
VELOCITY_DYNAMICS = {
    16:  "ppp",
    32:  "pp",
    48:  "p",
    64:  "mp",
    80:  "mf",
    96:  "f",
    112: "ff",
    127: "fff",
}

# Duración en ticks → nombre music21
TICKS_TO_DURATION = {
    1:  "16th",
    2:  "eighth",
    3:  "eighth",        # corchea con puntillo (aprox)
    4:  "quarter",
    6:  "quarter",       # negra con puntillo (aprox)
    8:  "half",
    12: "half",          # blanca con puntillo (aprox)
    16: "whole",
}

DOTTED_TICKS = {3, 6, 12}   # duraciones con puntillo

DUR_TOKEN_TICKS = {
    "<DUR_1>":1,  "<DUR_2>":2,  "<DUR_3>":3,  "<DUR_4>":4,
    "<DUR_6>":6,  "<DUR_8>":8,  "<DUR_12>":12,"<DUR_16>":16,
    "<DUR_T2>":2, "<DUR_T4>":4,
}

TEMPO_TOKEN_BPM = {
    "<TEMPO_60>":60,  "<TEMPO_80>":80,  "<TEMPO_100>":100,
    "<TEMPO_120>":120,"<TEMPO_140>":140,"<TEMPO_160>":160,
    "<TEMPO_180>":180,"<TEMPO_200>":200,
}

# ─────────────────────────────────────────────────────────────
# Parser de tokens → eventos
# ─────────────────────────────────────────────────────────────
def parse_tokens(tokens):
    """
    Convierte lista de tokens en lista de eventos:
    {
        "type":     "note" | "rest" | "chord_symbol" | "meta",
        "bar":      int,
        "pos":      int,   # en ticks dentro del compás
        "pitch":    int,   # MIDI pitch (solo para "note")
        "duration": int,   # en ticks
        "velocity": int,
        "chord":    str,   # e.g. "C MAJ7"  (solo para "chord_symbol")
        "key":      str,
        "tempo":    int,
        "timesig":  str,
        "genre":    str,
        "mood":     str,
    }
    """
    events     = []
    meta       = {
        "tempo": 120, "key": "C", "mode": "major",
        "timesig": "4/4", "genre": None, "mood": None
    }

    current_bar   = 1
    current_pos   = 0
    current_pitch = None
    current_dur   = None
    current_vel   = 64
    bar_chords    = {}   # bar → chord_string

    skip_prefixes = ("<GENRE_","<MOOD_","<ENERGY_","<INST_","<BEAT_",
                     "<SOS>","<EOS>","<PAD>","<UNK>","<SEP>","<MASK>")

    note_names_sharp = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    def flush():
        nonlocal current_pitch, current_dur, current_vel
        if current_pitch is None or current_dur is None:
            return
        events.append({
            "type":     "note",
            "bar":      current_bar,
            "pos":      current_pos,
            "pitch":    current_pitch,
            "duration": current_dur,
            "velocity": current_vel,
        })
        current_pitch = None
        current_dur   = None

    for tok in tokens:
        # Meta — contexto global
        if tok.startswith("<TEMPO_"):
            meta["tempo"] = TEMPO_TOKEN_BPM.get(tok, 120)
            continue
        if tok.startswith("<TIMESIG_"):
            parts = tok[9:-1].split("_")
            meta["timesig"] = f"{parts[0]}/{parts[1]}"
            continue
        if tok.startswith("<KEY_"):
            parts = tok[5:-1].split("_")
            note_raw = parts[0].replace("s", "#")
            meta["key"]  = note_raw
            meta["mode"] = "major" if parts[1] == "MAJ" else "minor"
            continue
        if tok.startswith("<GENRE_"):
            meta["genre"] = tok[7:-1].capitalize()
            continue
        if tok.startswith("<MOOD_"):
            meta["mood"] = tok[6:-1].capitalize()
            continue

        # Estructurales
        if tok.startswith("<BAR_"):
            flush()
            current_bar = int(tok[5:-1])
            current_pos = 0
            continue
        if tok.startswith("<POS_"):
            flush()
            current_pos = int(tok[5:-1])
            continue

        # Acordes — guardar por compás
        if tok.startswith("<CHORD_"):
            inner = tok[7:-1]
            parts = inner.split("_")
            root  = parts[0].replace("s", "#")
            qual  = parts[1] if len(parts) > 1 else "MAJ"
            chord_str = f"{root} {qual}"
            bar_chords[current_bar] = chord_str
            continue

        # Notas
        if tok.startswith("<PITCH_"):
            flush()
            current_pitch = int(tok[7:-1])
            current_dur   = None
            current_vel   = 64
            continue
        if tok == "<REST>":
            flush()
            current_pitch = None
            continue
        if tok in DUR_TOKEN_TICKS:
            current_dur = DUR_TOKEN_TICKS[tok]
            continue
        if tok.startswith("<VEL_"):
            current_vel = int(tok[5:-1])
            flush()
            continue

    flush()

    # Añadir chord_symbols como eventos
    for bar, chord_str in bar_chords.items():
        events.append({
            "type":  "chord_symbol",
            "bar":   bar,
            "pos":   0,
            "chord": chord_str,
        })

    # Ordenar por (bar, pos, type) — chord_symbol antes que note
    type_order = {"chord_symbol": 0, "note": 1, "rest": 2}
    events.sort(key=lambda e: (e["bar"], e["pos"], type_order.get(e["type"], 9)))

    return events, meta

# ─────────────────────────────────────────────────────────────
# Eventos → music21 Part
# ─────────────────────────────────────────────────────────────
def events_to_part(events, meta, part_name="Part", inst_obj=None):
    """Convierte lista de eventos a un music21.stream.Part."""
    p = stream.Part()
    # Normalizar tempo ilegible
    if meta.get("tempo", 120) > 160:
        meta = meta.copy()
        meta["tempo"] = meta["tempo"] // 2  # halftime notation
    p.id       = part_name
    p.partName = part_name

    if inst_obj:
        p.insert(0, inst_obj)

    quarter_per_tick = 1.0 / PPQ
    ts_parts         = meta["timesig"].split("/")
    beats_per_bar    = int(ts_parts[0])
    bar_duration_qn  = beats_per_bar  # en 4/4, cada compás = 4 quarter notes

    # Agrupar eventos por compás
    by_bar = defaultdict(list)
    for ev in events:
        by_bar[ev["bar"]].append(ev)

    if not by_bar:
        return p

    max_bar      = max(by_bar.keys())
    prev_dynamic = None

    for bar_idx in range(1, max_bar + 1):
        measure    = stream.Measure(number=bar_idx)
        bar_events = by_bar.get(bar_idx, [])

        # Insertar indicaciones de metro y tonalidad en el primer compás
        if bar_idx == 1:
            measure.insert(0, meter.TimeSignature(meta["timesig"]))
            measure.insert(0, key.Key(meta["key"], meta["mode"]))

            # Clave según instrumento
            clef_obj = music21.clef.TrebleClef()  # default
            if inst_obj is not None:
                inst_name = type(inst_obj).__name__
                if inst_name in ("ElectricBass", "Bass", "BassGuitar",
                                 "Tuba", "Contrabass", "Violoncello"):
                    clef_obj = music21.clef.BassClef()
                elif inst_name in ("Viola",):
                    clef_obj = music21.clef.AltoClef()
            measure.insert(0, clef_obj)

        # Construir mapa de posición → lista de notas para este compás
        # Necesitamos rellenar huecos con silencios
        note_map = {}   # pos_qn → (note_obj o chord_symbol, dynamic)

        for ev in bar_events:
            pos_qn = round(ev["pos"] * quarter_per_tick, 6)

            if ev["type"] == "chord_symbol":
                raw = (ev["chord"]
                    .replace("MAJ7","maj7").replace("MIN7","m7")
                    .replace("DOM7","7").replace("DIM7","dim7")
                    .replace("MAJ","").replace("MIN","m")
                    .replace("DIM","dim").replace("AUG","aug"))
                try:
                    cs = harmony.ChordSymbol(raw)
                    measure.insert(pos_qn, cs)
                except Exception:
                    pass

            elif ev["type"] == "note":
                ticks    = ev["duration"]
                # Clamp duración para que no exceda el compás
                max_ticks = TICKS_PER_BAR - ev["pos"]
                ticks     = min(ticks, max_ticks) if max_ticks > 0 else ticks
                ticks     = max(ticks, 1)

                dur_name = TICKS_TO_DURATION.get(ticks, "quarter")
                dotted   = ticks in DOTTED_TICKS

                d = music21.duration.Duration(dur_name)
                if dotted:
                    d.dots = 1

                # Asegurar que la duración no excede el compás
                dur_qn = d.quarterLength
                remaining = bar_duration_qn - pos_qn
                if dur_qn > remaining and remaining > 0:
                    d = music21.duration.Duration(quarterLength=remaining)

                n = note.Note(ev["pitch"])
                n.duration = d
                n.volume.velocity = ev["velocity"]

                # Dinámica — solo emitir si cambia
                vel_bin = min(VELOCITY_DYNAMICS.keys(),
                              key=lambda x: abs(x - ev["velocity"]))
                dyn_str = VELOCITY_DYNAMICS[vel_bin]
                if dyn_str != prev_dynamic:
                    dyn = dynamics.Dynamic(dyn_str)
                    measure.insert(pos_qn, dyn)
                    prev_dynamic = dyn_str

                measure.insert(pos_qn, n)

        # Rellenar el compás con silencios si está vacío o incompleto
        measure.makeRests(fillGaps=True, inPlace=True)

        p.append(measure)

    return p

# ─────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────
def tokens_to_musicxml(enc_tokens, dec_tokens, output_path,
                        melody_name="Melodía", accomp_name="Acompañamiento"):

    enc_events, enc_meta = parse_tokens(enc_tokens)
    dec_events, dec_meta = parse_tokens(dec_tokens)

    shared_meta = enc_meta.copy()
    if dec_meta["tempo"] != 120:
        shared_meta["tempo"] = dec_meta["tempo"]

    # Clamp tempo — 252 BPM es ilegible en partitura, normalizar a rango útil
    shared_meta["tempo"] = max(60, min(shared_meta["tempo"], 160))

    score = stream.Score()
    score.metadata = music21.metadata.Metadata()
    score.metadata.title = "Acompañamiento Generado"
    if shared_meta.get("genre"):
        score.metadata.title += f" — {shared_meta['genre']}"
    if shared_meta.get("mood"):
        score.metadata.title += f" ({shared_meta['mood']})"

    bpm = shared_meta["tempo"]
    mm  = tempo.MetronomeMark(number=bpm)

    # Pentagrama 1 — Melodía
    melody_inst          = instrument.Guitar()
    melody_inst.partName = melody_name
    melody_part = events_to_part(
        enc_events, shared_meta,
        part_name=melody_name,
        inst_obj=melody_inst
    )

    # Pentagrama 2 — Acompañamiento
    inst_obj = instrument.ElectricBass()
    for tok in dec_tokens:
        if tok == "<INST_PIANO>":
            inst_obj = instrument.Piano()
        elif tok == "<INST_GUITAR>":
            inst_obj = instrument.Guitar()
        elif tok == "<INST_BASS>":
            inst_obj = instrument.ElectricBass()
    inst_obj.partName = accomp_name

    accomp_part = events_to_part(
        dec_events, shared_meta,
        part_name=accomp_name,
        inst_obj=inst_obj
    )

    # Filtrar compases vacíos del acompañamiento
    # (compases que solo tienen silencios)
    non_empty_bars = set()
    for ev in dec_events:
        if ev["type"] == "note":
            non_empty_bars.add(ev["bar"])

    # Insertar marca de tempo en primer compás con notas
    mel_measures = melody_part.getElementsByClass(stream.Measure)
    if mel_measures:
        mel_measures[0].insert(0, mm)

    score.append(melody_part)
    score.append(accomp_part)

    # Escribir XML manualmente con encoding correcto
    import xml.etree.ElementTree as ET
    from music21.musicxml.m21ToXml import ScoreExporter

    exporter     = ScoreExporter(score)
    root_element = exporter.parse()

    xml_bytes = ET.tostring(root_element, encoding="unicode", xml_declaration=False)
    xml_str   = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes

    output_path = str(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"  MusicXML guardado en: {output_path}")

    mel_notes = len(melody_part.flatten().notes)
    acc_notes = len(accomp_part.flatten().notes)
    mel_bars  = len(melody_part.getElementsByClass(stream.Measure))
    acc_bars  = len(accomp_part.getElementsByClass(stream.Measure))
    print(f"  Melodía:        {mel_notes} notas, {mel_bars} compases")
    print(f"  Acompañamiento: {acc_notes} notas, {acc_bars} compases")

    return score
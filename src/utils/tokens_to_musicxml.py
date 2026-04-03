# src/utils/tokens_to_musicxml.py
import music21
from music21 import stream, note, chord, meter, tempo, key, dynamics
from music21 import harmony, instrument, clef
from collections import defaultdict

PPQ           = 16
TICKS_PER_BAR = 64

VELOCITY_DYNAMICS = {
    16:"ppp", 32:"pp", 48:"p", 64:"mp",
    80:"mf",  96:"f", 112:"ff", 127:"fff",
}

DUR_TOKEN_TICKS = {
    "<DUR_1>":1,  "<DUR_2>":2,  "<DUR_3>":3,  "<DUR_4>":4,
    "<DUR_6>":6,  "<DUR_8>":8,  "<DUR_12>":12, "<DUR_16>":16,
    "<DUR_T2>":2, "<DUR_T4>":4,
}

TEMPO_TOKEN_BPM = {
    "<TEMPO_60>":60,  "<TEMPO_80>":80,  "<TEMPO_100>":100,
    "<TEMPO_120>":120,"<TEMPO_140>":140,"<TEMPO_160>":160,
    "<TEMPO_180>":180,"<TEMPO_200>":200,
}

def ticks_to_quarterLength(ticks):
    """Convierte ticks de semicorchea a quarter notes para music21."""
    return ticks / PPQ

def parse_tokens(tokens):
    """
    Convierte tokens a lista de eventos nota con posición absoluta en ticks.
    Retorna (events, meta).
    Cada evento: { bar, pos_ticks, pitch, dur_ticks, velocity }
    """
    meta = { "tempo": 120, "key": "C", "mode": "major",
             "timesig": "4/4", "genre": None, "mood": None }

    events      = []
    bar_chords  = {}
    note_names  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    current_bar   = 1
    current_pos   = 0
    current_pitch = None
    current_dur   = None
    current_vel   = 80

    def flush():
        nonlocal current_pitch, current_dur, current_vel
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
            meta["key"]  = p[0].replace("s","#")
            meta["mode"] = "major" if p[1]=="MAJ" else "minor"
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
        elif tok.startswith("<CHORD_"):
            inner = tok[7:-1].split("_")
            root  = inner[0].replace("s","#")
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
            current_vel = int(tok[5:-1])
            flush()
        elif tok in ("<SOS>","<EOS>","<PAD>","<UNK>"):
            pass

    flush()
    return events, meta, bar_chords


def events_to_part(events, meta, bar_chords, part_name, inst_obj):
    """Construye un music21 Part a partir de eventos con posición en ticks."""
    p = stream.Part()
    p.id       = part_name
    p.partName = part_name

    if inst_obj:
        p.insert(0, inst_obj)

    ts_num = int(meta["timesig"].split("/")[0])

    # Agrupar por compás
    by_bar = defaultdict(list)
    for ev in events:
        by_bar[ev["bar"]].append(ev)

    if not by_bar:
        return p

    max_bar      = max(by_bar.keys())
    prev_dynamic = None

    for bar_idx in range(1, max_bar + 1):
        m          = stream.Measure(number=bar_idx)
        bar_events = by_bar.get(bar_idx, [])

        if bar_idx == 1:
            m.insert(0, meter.TimeSignature(meta["timesig"]))
            m.insert(0, key.Key(meta["key"], meta["mode"]))
            # Clave según instrumento
            if inst_obj and type(inst_obj).__name__ in (
                "ElectricBass","Bass","BassGuitar","Contrabass","Tuba"):
                m.insert(0, clef.BassClef())
            else:
                m.insert(0, clef.TrebleClef())

        # Cifrado de acordes
        if bar_idx in bar_chords:
            root, qual = bar_chords[bar_idx]
            chord_str = (root + qual
                .replace("MAJ7","maj7").replace("MIN7","m7")
                .replace("DOM7","7").replace("DIM7","dim7")
                .replace("MAJ","").replace("MIN","m")
                .replace("DIM","dim").replace("AUG","aug"))
            try:
                cs = harmony.ChordSymbol(chord_str)
                m.insert(0, cs)
            except Exception:
                pass

        # Insertar notas
        for ev in bar_events:
            pos_ticks = ev["pos_ticks"]
            dur_ticks = ev["dur_ticks"]

            # Convertir a quarter notes
            pos_qn = ticks_to_quarterLength(pos_ticks)
            dur_qn = ticks_to_quarterLength(dur_ticks)

            # Clamp: la nota no puede salirse del compás
            bar_dur_qn = ts_num  # en 4/4 = 4 quarter notes
            pos_qn     = min(pos_qn, bar_dur_qn - 0.0625)  # mínimo 1 tick antes del fin
            dur_qn     = min(dur_qn, bar_dur_qn - pos_qn)
            dur_qn     = max(dur_qn, 0.0625)  # mínimo una semicorchea

            n = note.Note(ev["pitch"])
            n.duration         = music21.duration.Duration(quarterLength=dur_qn)
            n.volume.velocity  = ev["velocity"]

            # Dinámica — emitir solo cuando cambia
            vel_bin = min(VELOCITY_DYNAMICS.keys(),
                          key=lambda x: abs(x - ev["velocity"]))
            dyn_str = VELOCITY_DYNAMICS[vel_bin]
            if dyn_str != prev_dynamic:
                m.insert(pos_qn, dynamics.Dynamic(dyn_str))
                prev_dynamic = dyn_str

            m.insert(pos_qn, n)

        # Rellenar huecos con silencios
        m.makeRests(fillGaps=True, inPlace=True)
        p.append(m)

    return p


def tokens_to_musicxml(enc_tokens, dec_tokens, output_path,
                        melody_name="Melodía", accomp_name="Acompañamiento"):

    enc_events, enc_meta, enc_chords = parse_tokens(enc_tokens)
    dec_events, dec_meta, dec_chords = parse_tokens(dec_tokens)

    # Usar meta del encoder, clampear tempo
    shared_meta = enc_meta.copy()
    shared_meta["tempo"] = max(60, min(shared_meta["tempo"], 160))

    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata()
    title = "Acompañamiento Generado"
    if shared_meta.get("genre"): title += f" — {shared_meta['genre']}"
    if shared_meta.get("mood"):  title += f" ({shared_meta['mood']})"
    score.metadata.title = title

    # Pentagrama melodía
    mel_inst          = instrument.Guitar()
    mel_inst.partName = melody_name
    mel_part = events_to_part(enc_events, shared_meta, enc_chords, melody_name, mel_inst)

    # Insertar tempo en primer compás
    mel_measures = mel_part.getElementsByClass(stream.Measure)
    if mel_measures:
        mel_measures[0].insert(0, tempo.MetronomeMark(number=shared_meta["tempo"]))

    # Pentagrama acompañamiento
    acc_inst = instrument.ElectricBass()
    for tok in dec_tokens:
        if tok == "<INST_PIANO>":   acc_inst = instrument.Piano()
        elif tok == "<INST_GUITAR>": acc_inst = instrument.Guitar()
        elif tok == "<INST_BASS>":   acc_inst = instrument.ElectricBass()
    acc_inst.partName = accomp_name
    acc_part = events_to_part(dec_events, shared_meta, dec_chords, accomp_name, acc_inst)

    score.append(mel_part)
    score.append(acc_part)

    # Exportar
    import xml.etree.ElementTree as ET
    from music21.musicxml.m21ToXml import ScoreExporter
    exporter     = ScoreExporter(score)
    root_element = exporter.parse()
    xml_bytes    = ET.tostring(root_element, encoding="unicode", xml_declaration=False)
    xml_str      = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes

    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write(xml_str)

    mel_notes = len(mel_part.flatten().notes)
    acc_notes = len(acc_part.flatten().notes)
    print(f"  MusicXML guardado en: {output_path}")
    print(f"  Melodía:        {mel_notes} notas, {len(mel_part.getElementsByClass(stream.Measure))} compases")
    print(f"  Acompañamiento: {acc_notes} notas, {len(acc_part.getElementsByClass(stream.Measure))} compases")

    return score
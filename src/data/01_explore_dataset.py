# src/data/01_explore_dataset.py
import json
import h5py
import numpy as np
from collections import defaultdict
from pathlib import Path
import pretty_midi

DATA_DIR    = Path("data/raw")
LMD_DIR     = DATA_DIR / "lmd_matched"
H5_DIR      = DATA_DIR / "lmd_matched_h5"
SCORES_PATH = DATA_DIR / "match_scores.json"

# ─────────────────────────────────────────────
# 1. Conteo general
# ─────────────────────────────────────────────
midi_files = list(LMD_DIR.rglob("*.mid"))
h5_files   = list(H5_DIR.rglob("*.h5"))

print("=" * 55)
print(f"  MIDIs encontrados:         {len(midi_files)}")
print(f"  Archivos HDF5 encontrados: {len(h5_files)}")

# ─────────────────────────────────────────────
# 2. Distribución de tags — path corregido
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DISTRIBUCIÓN DE TAGS")
print("=" * 55)

tag_counter      = defaultdict(float)  # acumulamos frecuencia, no solo count
tag_appearances  = defaultdict(int)    # cuántos archivos tienen ese tag
h5_with_tags     = 0
h5_errors        = 0

for h5_path in h5_files:
    try:
        with h5py.File(h5_path, "r") as f:
            tags  = f["metadata"]["artist_terms"][:]
            freqs = f["metadata"]["artist_terms_freq"][:]
            if len(tags) > 0:
                h5_with_tags += 1
                for t, freq in zip(tags, freqs):
                    tag = t.decode() if isinstance(t, bytes) else str(t)
                    tag = tag.lower().strip()
                    tag_counter[tag]     += float(freq)
                    tag_appearances[tag] += 1
    except Exception:
        h5_errors += 1

print(f"  HDF5 con tags:    {h5_with_tags} / {len(h5_files)}")
print(f"  HDF5 con errores: {h5_errors}")
print(f"  Tags únicos:      {len(tag_counter)}")

print(f"\n  Top 60 tags por frecuencia acumulada:")
print(f"  {'Tag':<35} {'Archivos':>8}  {'Freq acum':>10}")
print(f"  {'-'*35} {'-'*8}  {'-'*10}")
for tag, freq in sorted(tag_counter.items(), key=lambda x: -x[1])[:60]:
    bar = "█" * min(int(tag_appearances[tag] // 80), 20)
    print(f"  {tag:<35} {tag_appearances[tag]:>8}  {freq:>10.1f}  {bar}")

# ─────────────────────────────────────────────
# 3. Cuántos HDF5 caen en nuestros géneros objetivo
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  COBERTURA DE GÉNEROS/MOODS OBJETIVO")
print("=" * 55)

GENRE_KEYWORDS = {
    "ROCK":        ["rock", "classic rock", "hard rock", "alternative", "indie", "grunge", "punk"],
    "POP":         ["pop", "pop rock", "dance pop", "teen pop", "synth pop"],
    "FUNK":        ["funk", "soul", "r&b", "groove", "motown", "neo soul"],
    "JAZZ":        ["jazz", "bebop", "fusion", "swing", "bossa nova", "cool jazz"],
    "BLUES":       ["blues", "blues rock", "electric blues", "chicago blues"],
    "LATIN":       ["latin", "salsa", "bossa nova", "samba", "latin pop"],
    "ELECTRONIC":  ["electronic", "edm", "techno", "house", "ambient", "downtempo"],
    "CLASSICAL":   ["classical", "orchestra", "chamber music", "baroque", "piano classical"],
}

MOOD_KEYWORDS = {
    "HAPPY":   ["happy", "upbeat", "cheerful", "fun", "feel good", "uplifting"],
    "SAD":     ["sad", "melancholy", "depressing", "heartbreak", "emotional"],
    "DARK":    ["dark", "gloomy", "evil", "aggressive", "heavy"],
    "RELAXED": ["relaxed", "chill", "mellow", "calm", "smooth", "easy listening"],
    "TENSE":   ["tense", "intense", "powerful", "energetic", "driving"],
    "BRIGHT":  ["bright", "positive", "optimistic", "lively"],
}

genre_counts = defaultdict(int)
mood_counts  = defaultdict(int)
both_counts  = 0
none_counts  = 0

for h5_path in h5_files:
    try:
        with h5py.File(h5_path, "r") as f:
            tags = f["metadata"]["artist_terms"][:]
            tags_str = {t.decode().lower().strip() for t in tags}

        assigned_genre = None
        assigned_mood  = None

        for genre, keywords in GENRE_KEYWORDS.items():
            if any(kw in tags_str for kw in keywords):
                assigned_genre = genre
                break

        for mood, keywords in MOOD_KEYWORDS.items():
            if any(kw in tags_str for kw in keywords):
                assigned_mood = mood
                break

        if assigned_genre:
            genre_counts[assigned_genre] += 1
        if assigned_mood:
            mood_counts[assigned_mood] += 1
        if assigned_genre and assigned_mood:
            both_counts += 1
        if not assigned_genre and not assigned_mood:
            none_counts += 1

    except Exception:
        pass

print(f"\n  Géneros:")
for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count // 30)
    print(f"    {genre:<15} {count:>5}  {bar}")

print(f"\n  Moods:")
for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count // 20)
    print(f"    {mood:<15} {count:>5}  {bar}")

print(f"\n  Con género + mood: {both_counts}")
print(f"  Sin ninguno:       {none_counts}")

# ─────────────────────────────────────────────
# 4. Match scores
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MATCH SCORES")
print("=" * 55)

with open(SCORES_PATH) as f:
    scores = json.load(f)

all_scores = np.array([
    score
    for midis in scores.values()
    for score in midis.values()
])

print(f"  Total matches:  {len(all_scores)}")
print(f"  Promedio:       {all_scores.mean():.4f}")
print(f"  Mediana:        {np.median(all_scores):.4f}")
for threshold in [0.5, 0.6, 0.7, 0.75, 0.8]:
    count = (all_scores >= threshold).sum()
    print(f"  >= {threshold:.2f}:  {count:>6} matches  ({100*count/len(all_scores):.1f}%)")

# ─────────────────────────────────────────────
# 5. Instrumentos en muestra de MIDIs
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ANÁLISIS DE INSTRUMENTOS (muestra 500 MIDIs)")
print("=" * 55)

instrument_counter = defaultdict(int)
has_bass           = 0
has_guitar         = 0
has_piano          = 0
has_melody_target  = 0  # tiene al menos uno de los 3
midi_errors        = 0

for midi_path in midi_files[:500]:
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        non_drum = [i for i in pm.instruments if not i.is_drum]
        classes  = {pretty_midi.program_to_instrument_class(i.program) for i in non_drum}

        for inst in non_drum:
            instrument_counter[pretty_midi.program_to_instrument_class(inst.program)] += 1

        b = "Bass"     in classes
        g = "Guitar"   in classes
        p = "Piano"    in classes
        if b: has_bass   += 1
        if g: has_guitar += 1
        if p: has_piano  += 1
        if b or g or p:
            has_melody_target += 1

    except Exception:
        midi_errors += 1

print(f"  MIDIs analizados:              500")
print(f"  Con error:                     {midi_errors}")
print(f"  Tienen Bass:                   {has_bass}")
print(f"  Tienen Guitar:                 {has_guitar}")
print(f"  Tienen Piano:                  {has_piano}")
print(f"  Tienen al menos uno de los 3:  {has_melody_target}")
print(f"\n  Todas las clases encontradas:")
for inst, count in sorted(instrument_counter.items(), key=lambda x: -x[1]):
    print(f"    {inst:<35} {count:>5}")

print("\n" + "=" * 55)
print("  EXPLORACIÓN COMPLETA")
print("=" * 55)
# src/data/02_filter_and_label.py
"""
Cruza LMD-matched con HDF5 del MSD, asigna etiquetas de género/mood,
aplica filtros de calidad y guarda un CSV con los MIDIs aptos.
"""
import json
import h5py
import csv
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

DATA_DIR    = Path("data/raw")
LMD_DIR     = DATA_DIR / "lmd_matched"
H5_DIR      = DATA_DIR / "lmd_matched_h5"
SCORES_PATH = DATA_DIR / "match_scores.json"
OUT_CSV     = Path("data/processed/labeled_midi.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
MATCH_SCORE_THRESHOLD = 0.6
CAP_PER_GENRE         = 1500

# Orden importa: más específico primero
# El primer género que hace match gana
GENRE_MAP = [
    ("FUNK",       ["funk", "soul", "r&b", "groove", "motown", "neo soul",
                    "blues", "electric blues", "chicago blues", "blues rock"]),
    ("JAZZ",       ["jazz", "bebop", "fusion", "swing", "cool jazz", "soul jazz"]),
    ("LATIN",      ["latin", "salsa", "samba", "latin pop", "bossa nova", "latin jazz"]),
    ("CLASSICAL",  ["classical", "orchestra", "chamber music", "baroque", "opera"]),
    ("ELECTRONIC", ["electronic", "techno", "house", "trance", "electronica",
                    "synthpop", "downtempo", "ambient", "edm", "electro",
                    "progressive house", "tech house", "deep house",
                    "progressive trance", "drum and bass", "dubstep"]),
    ("POP",        ["pop", "dance pop", "teen pop", "europop", "bubblegum"]),
    ("ROCK", ["rock", "classic rock", "hard rock", "alternative rock",
          "indie rock", "punk", "grunge", "folk rock", "country rock",
          "alternative", "indie", "new wave", "soft rock", "pop rock",
          "metal", "heavy metal", "death metal", "black metal",
          "progressive rock", "psychedelic rock", "garage rock"]),
]

MOOD_MAP = [
    ("HAPPY",   ["happy", "upbeat", "cheerful", "fun", "feel good", "dance",
                 "party", "disco", "feel-good", "uplifting", "80s", "pop dance",
                 "eurodance", "bubblegum"]),

    ("SAD",     ["sad", "melancholy", "depressing", "heartbreak", "emotional",
                 "ballad", "sorrow", "grief", "tearjerker", "torch song",
                 "singer-songwriter", "acoustic"]),  # acústico/songwriter → tendencia sad/introspectivo

    ("DARK",    ["dark", "gloomy", "evil", "aggressive", "heavy", "gothic",
                 "menacing", "sinister", "brooding", "metal", "heavy metal",
                 "death metal", "black metal", "doom", "industrial", "noise"]),

    ("TENSE",   ["tense", "intense", "powerful", "energetic", "driving",
                 "adrenaline", "epic", "punk", "hardcore", "thrash",
                 "soundtrack", "cinematic"]),         # soundtracks → tensión/drama

    ("RELAXED", ["relaxed", "chill", "mellow", "calm", "smooth", "easy listening",
                 "chill-out", "peaceful", "soothing", "ambient", "downtempo",
                 "lounge", "bossa nova", "folk", "country", "acoustic guitar",
                 "new age", "instrumental"]),
]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def assign_genre(tags_set):
    for genre, keywords in GENRE_MAP:
        if any(kw in tags_set for kw in keywords):
            return genre
    return None

def assign_mood(tags_set):
    for mood, keywords in MOOD_MAP:
        if any(kw in tags_set for kw in keywords):
            return mood
    return None

def get_h5_path(msd_id):
    """
    Los HDF5 están organizados como:
    lmd_matched_h5 / A / A / A / TRAAAAW128F429D538.h5
    donde las subcarpetas son los primeros 3 caracteres del MSD ID
    """
    return H5_DIR / msd_id[2] / msd_id[3] / msd_id[4] / f"{msd_id}.h5"

# ─────────────────────────────────────────────────────────────
# 1. Cargar match scores y construir índice msd_id → mejores MIDIs
# ─────────────────────────────────────────────────────────────
print("Cargando match scores...")
with open(SCORES_PATH) as f:
    scores = json.load(f)

# Para cada MSD ID, guardar lista de (midi_md5, score) sobre el umbral
msd_to_midis = {}
for msd_id, midi_dict in scores.items():
    valid = [
        (md5, score)
        for md5, score in midi_dict.items()
        if score >= MATCH_SCORE_THRESHOLD
    ]
    if valid:
        # Ordenar por score descendente — tomar el mejor MIDI por canción
        valid.sort(key=lambda x: -x[1])
        msd_to_midis[msd_id] = valid

print(f"  MSD IDs con al menos un MIDI válido: {len(msd_to_midis)}")

# ─────────────────────────────────────────────────────────────
# 2. Construir índice md5 → path en disco
# ─────────────────────────────────────────────────────────────
print("Indexando archivos MIDI en disco...")
md5_to_path = {}
for midi_path in LMD_DIR.rglob("*.mid"):
    md5 = midi_path.stem  # nombre del archivo sin extensión = md5
    md5_to_path[md5] = midi_path

print(f"  MIDIs indexados: {len(md5_to_path)}")

# ─────────────────────────────────────────────────────────────
# 3. Cruzar con HDF5, asignar etiquetas, aplicar cap
# ─────────────────────────────────────────────────────────────
print("\nFiltrando y etiquetando...")

genre_counts   = defaultdict(int)
mood_counts    = defaultdict(int)
results        = []

stats = {
    "total_msd":        len(msd_to_midis),
    "no_h5":            0,
    "no_tags":          0,
    "no_genre":         0,
    "no_mood":          0,
    "no_midi_on_disk":  0,
    "cap_rejected":     0,
    "accepted":         0,
}

for msd_id, midi_list in tqdm(msd_to_midis.items(), desc="Procesando"):

    # Buscar HDF5
    h5_path = get_h5_path(msd_id)
    if not h5_path.exists():
        stats["no_h5"] += 1
        continue

    # Leer tags
    try:
        with h5py.File(h5_path, "r") as f:
            raw_tags  = f["metadata"]["artist_terms"][:]
            raw_freqs = f["metadata"]["artist_terms_freq"][:]
            title     = f["metadata"]["songs"]["title"][0]
            artist    = f["metadata"]["songs"]["artist_name"][0]
            tempo     = float(f["analysis"]["songs"]["tempo"][0])
            energy    = float(f["analysis"]["songs"]["energy"][0])

        if len(raw_tags) == 0:
            stats["no_tags"] += 1
            continue

        tags_set = {
            t.decode().lower().strip() if isinstance(t, bytes) else t.lower().strip()
            for t in raw_tags
        }
        title  = title.decode()  if isinstance(title,  bytes) else str(title)
        artist = artist.decode() if isinstance(artist, bytes) else str(artist)

    except Exception as e:
        stats["no_tags"] += 1
        continue

    # Asignar género y mood
    genre = assign_genre(tags_set)
    mood  = assign_mood(tags_set)

    if not genre:
        stats["no_genre"] += 1
        continue
    if not mood:
        stats["no_mood"] += 1
        continue

    # Aplicar cap
    if genre_counts[genre] >= CAP_PER_GENRE:
        stats["cap_rejected"] += 1
        continue

    # Verificar que el mejor MIDI esté en disco
    midi_path = None
    best_score = None
    for md5, score in midi_list:
        if md5 in md5_to_path:
            midi_path  = md5_to_path[md5]
            best_score = score
            break

    if midi_path is None:
        stats["no_midi_on_disk"] += 1
        continue

    # Aceptar
    genre_counts[genre] += 1
    mood_counts[mood]   += 1
    stats["accepted"]   += 1

    results.append({
        "msd_id":     msd_id,
        "midi_path":  str(midi_path),
        "title":      title,
        "artist":     artist,
        "genre":      genre,
        "mood":       mood,
        "tempo":      round(tempo, 1),
        "energy":     round(energy, 4),
        "score":      round(best_score, 4),
    })

# ─────────────────────────────────────────────────────────────
# 4. Guardar CSV
# ─────────────────────────────────────────────────────────────
fieldnames = ["msd_id", "midi_path", "title", "artist",
              "genre", "mood", "tempo", "energy", "score"]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# ─────────────────────────────────────────────────────────────
# 5. Reporte final
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  REPORTE FINAL")
print("=" * 55)
print(f"  MSD IDs procesados:       {stats['total_msd']}")
print(f"  Sin HDF5:                 {stats['no_h5']}")
print(f"  Sin tags:                 {stats['no_tags']}")
print(f"  Sin género asignable:     {stats['no_genre']}")
print(f"  Sin mood asignable:       {stats['no_mood']}")
print(f"  Sin MIDI en disco:        {stats['no_midi_on_disk']}")
print(f"  Rechazados por cap:       {stats['cap_rejected']}")
print(f"  ── ACEPTADOS:             {stats['accepted']}")

print(f"\n  Distribución por género:")
for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count // 30)
    print(f"    {genre:<15} {count:>5}  {bar}")

print(f"\n  Distribución por mood:")
for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count // 20)
    print(f"    {mood:<15} {count:>5}  {bar}")

print(f"\n  CSV guardado en: {OUT_CSV}")
print("=" * 55)
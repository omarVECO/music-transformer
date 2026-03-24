"""
batch.py
Procesa N audios de una carpeta y genera un .mid por cada uno.
Solo procesa audios que aún no tienen su .mid generado — los que
ya existen se saltan automáticamente.

Uso:
    python src/input_process/batch.py
    python src/input_process/batch.py --input_dir data/input_temp --genre FUNK --mood DARK --instrument BASS
    python src/input_process/batch.py --forzar   ← reprocesa todo aunque ya exista el .mid
"""
import os
import sys
import argparse
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))
os.chdir(BASE_DIR)

from input_process.pipeline import run_input_pipeline

EXTENSIONES_VALIDAS = {".wav", ".mp3", ".flac"}
GENEROS      = ["ROCK", "POP", "FUNK", "JAZZ", "LATIN", "CLASSICAL", "ELECTRONIC"]
MOODS        = ["HAPPY", "SAD", "DARK", "RELAXED", "TENSE"]
INSTRUMENTOS = ["BASS", "PIANO", "GUITAR"]
OUTPUT_DIR   = os.path.join("data", "processed", "input_midis")

def _ya_procesado(audio_path):
    """Retorna True si ya existe el .mid correspondiente a este audio."""
    base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    midi_path = os.path.join(OUTPUT_DIR, f"{base_name}.mid")
    return os.path.isfile(midi_path)

def run_batch(input_dir, genre, mood, instrument, forzar=False):
    """
    Procesa todos los audios nuevos en input_dir.
    Si forzar=False, salta los que ya tienen .mid generado.
    Si un audio falla, continúa con los demás y reporta al final.
    """
    # Recopilar audios
    todos = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in EXTENSIONES_VALIDAS
    ]

    if not todos:
        print(f"No se encontraron audios en: {input_dir}")
        return []

    # Separar nuevos de ya procesados
    pendientes = [a for a in todos if not _ya_procesado(a)] if not forzar else todos
    saltados   = [a for a in todos if _ya_procesado(a)]     if not forzar else []

    print(f"\n{'='*55}")
    print(f"  PROCESAMIENTO POR LOTE")
    print(f"{'='*55}")
    print(f"  Carpeta    : {input_dir}")
    print(f"  Total      : {len(todos)} audios")
    print(f"  Ya listos  : {len(saltados)} (se saltan)")
    print(f"  Pendientes : {len(pendientes)} (a procesar)")
    print(f"  Género     : {genre}")
    print(f"  Mood       : {mood}")
    print(f"  Instrumento: {instrument}")
    if forzar:
        print(f"  Modo       : FORZAR — reprocesa todo")
    print(f"{'='*55}\n")

    if not pendientes:
        print("Todos los audios ya tienen su MIDI generado.")
        return []

    resultados = []

    for i, audio_path in enumerate(pendientes, 1):
        nombre = os.path.basename(audio_path)
        print(f"[{i}/{len(pendientes)}] {nombre}")

        try:
            resultado = run_input_pipeline(audio_path, genre, mood, instrument)
        except Exception as e:
            resultado = {
                "midi_path":  None,
                "genre":      genre,
                "mood":       mood,
                "instrument": instrument,
                "error":      f"Error inesperado: {str(e)}",
            }

        resultado["audio"] = nombre
        resultados.append(resultado)

        if resultado["error"]:
            print(f"  {resultado['error']}\n")
        else:
            print(f"  {resultado['midi_path']}\n")

    _imprimir_reporte(resultados, saltados)
    _guardar_reporte(resultados, genre, mood, instrument)

    return resultados


def _imprimir_reporte(resultados, saltados):
    exitosos = [r for r in resultados if r["error"] is None]
    fallidos  = [r for r in resultados if r["error"] is not None]

    print(f"\n{'='*55}")
    print(f"  REPORTE FINAL")
    print(f"{'='*55}")
    print(f"  Saltados (ya existían) : {len(saltados)}")
    print(f"  Procesados             : {len(resultados)}")
    print(f"  Exitosos            : {len(exitosos)}")
    print(f"  Fallidos            : {len(fallidos)}")

    if saltados:
        print(f"\n  Saltados:")
        for a in saltados:
            print(f"    ⏭  {os.path.basename(a)}")

    if fallidos:
        print(f"\n  Errores:")
        for r in fallidos:
            print(f"    {r['audio']}: {r['error']}")

    if exitosos:
        print(f"\n  MIDIs nuevos generados:")
        for r in exitosos:
            print(f"    {r['audio']} → {r['midi_path']}")

    print(f"{'='*55}\n")


def _guardar_reporte(resultados, genre, mood, instrument):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    reporte_path = os.path.join(OUTPUT_DIR, f"batch_report_{timestamp}.json")

    reporte = {
        "timestamp":  timestamp,
        "genre":      genre,
        "mood":       mood,
        "instrument": instrument,
        "total":      len(resultados),
        "exitosos":   len([r for r in resultados if r["error"] is None]),
        "fallidos":   len([r for r in resultados if r["error"] is not None]),
        "resultados": resultados,
    }

    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"  Reporte guardado en: {reporte_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa audios nuevos → MIDIs")
    parser.add_argument("--input_dir",  default=os.path.join("data", "input_temp"),
                        help="Carpeta con los audios de entrada")
    parser.add_argument("--genre",      default="FUNK",  choices=GENEROS)
    parser.add_argument("--mood",       default="HAPPY", choices=MOODS)
    parser.add_argument("--instrument", default="BASS",  choices=INSTRUMENTOS)
    parser.add_argument("--forzar",     action="store_true",
                        help="Reprocesa todos aunque ya exista el .mid")
    args = parser.parse_args()

    run_batch(args.input_dir, args.genre, args.mood, args.instrument, args.forzar)
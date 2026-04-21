# src/utils/generate_score.py
"""
Pipeline completo: MIDI de entrada → inferencia → MusicXML

CAMBIOS respecto a la versión anterior:
- Se eliminaron las importaciones de top_p_sampling, apply_repetition_penalty y
  apply_pitch_range_boost desde inference.py — esas funciones son ahora internas
  al módulo de inferencia y ya no forman parte de la API pública.
- generate() acepta dos nuevos parámetros opcionales: top_k y repetition_penalty.
- Se añadió --top_k y --repetition_penalty al CLI para dar control completo al usuario.
- tokens_to_musicxml recibe ahora gen_ids (lista de enteros) en lugar de dec_tokens
  (lista de strings) — la conversión ID→token se hace dentro de tokens_to_musicxml
  para mantener un único punto de verdad.

Uso:
  PYTHONPATH=src python src/utils/generate_score.py \
    --input_midi mi_guitarra.mid \
    --genre FUNK \
    --mood DARK \
    --instrument BASS \
    --output partitura.xml
"""
import argparse
import torch
import pretty_midi
from pathlib import Path

from model.config import ModelConfig
from model.transformer import MusicTransformer
from model.inference import generate  # API pública: solo generate()
from data.midi_tokenizer import (
    detect_key, select_tracks, notes_to_token_sequence,
    inst_to_token, TOKEN2ID, ID2TOKEN, MIN_PITCH, MAX_PITCH,
    SECONDS_PER_TICK, decode_event_tokens_to_midi,
)
from utils.tokens_to_musicxml import tokens_to_musicxml


def main():
    parser = argparse.ArgumentParser(
        description="Genera acompañamiento MIDI y lo exporta a MusicXML"
    )
    parser.add_argument("--input_midi",         required=True,
                        help="Ruta al MIDI de melodía de entrada")
    parser.add_argument("--genre",              default="FUNK",
                        choices=["ROCK","POP","FUNK","JAZZ","LATIN","CLASSICAL","ELECTRONIC"])
    parser.add_argument("--mood",               default="HAPPY",
                        choices=["HAPPY","SAD","DARK","RELAXED","TENSE"])
    parser.add_argument("--instrument",         default="BASS",
                        choices=["BASS","PIANO","GUITAR"])
    parser.add_argument("--output",             default="partitura.xml",
                        help="Ruta de salida del MusicXML")
    parser.add_argument("--temperature",        type=float, default=0.9,
                        help="Temperatura de muestreo (0.7–1.1 recomendado)")
    parser.add_argument("--top_p",              type=float, default=0.92,
                        help="Nucleus sampling threshold")
    parser.add_argument("--top_k",              type=int,   default=50,
                        help="Top-k sampling (0 = desactivado)")
    parser.add_argument("--repetition_penalty", type=float, default=1.3,
                        help="Penalización de pitch repetido (1.0 = sin penalización)")
    parser.add_argument("--max_tokens",         type=int,   default=1024,
                        help="Máximo de tokens a generar")
    parser.add_argument("--output_midi",        default=None,
                        help="Ruta opcional para guardar acompañamiento como MIDI (.mid)")
    parser.add_argument("--include_melody",     action="store_true",
                        help="Incluir la melodía original (normalizada a 120 BPM) en el MIDI de salida")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Cargar modelo ──────────────────────────────────────────────────
    config = ModelConfig()
    model  = MusicTransformer(config).to(device)
    ckpt   = torch.load("checkpoints/v2/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Modelo: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # ── Tokenizar melodía de entrada ───────────────────────────────────
    pm = pretty_midi.PrettyMIDI(args.input_midi)
    melody_inst, _ = select_tracks(pm)
    if melody_inst is None:
        melody_inst = next((i for i in pm.instruments if not i.is_drum), None)
    if melody_inst is None:
        raise ValueError("No se encontró pista melódica en el MIDI de entrada")

    tempo_bpm = pm.estimate_tempo()
    if not (30 < tempo_bpm < 300):
        tempo_bpm = 120.0
    key_token = detect_key(pm)
    print(f"Entrada: {key_token}  {tempo_bpm:.0f} BPM")

    # El encoder sigue usando la representación posicional (BAR/POS/PITCH/DUR/VEL)
    enc_tokens = notes_to_token_sequence(
        melody_inst, pm, tempo_bpm, key_token,
        args.genre, args.mood, 0.5,
        inst_to_token(melody_inst), is_encoder=True
    )
    if enc_tokens is None:
        raise ValueError("La melodía es demasiado corta para tokenizar")

    enc_tokens = enc_tokens[:config.max_seq_len]
    enc_ids    = torch.tensor(
        [TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in enc_tokens],
        dtype=torch.long,
    )
    enc_mask = torch.ones(len(enc_ids), dtype=torch.bool)

    # Padding hasta max_seq_len
    pad_len  = config.max_seq_len - len(enc_ids)
    enc_ids  = torch.cat([enc_ids,  torch.zeros(pad_len, dtype=torch.long)])
    enc_mask = torch.cat([enc_mask, torch.zeros(pad_len, dtype=torch.bool)])

    # ── Generar acompañamiento ─────────────────────────────────────────
    prompt = [
        TOKEN2ID["<SOS>"],
        TOKEN2ID.get(f"<GENRE_{args.genre}>", TOKEN2ID["<UNK>"]),
        TOKEN2ID.get(f"<MOOD_{args.mood}>",   TOKEN2ID["<UNK>"]),
        TOKEN2ID.get("<ENERGY_MED>",           TOKEN2ID["<UNK>"]),
        TOKEN2ID.get(f"<INST_{args.instrument}>", TOKEN2ID["<UNK>"]),
    ]
    print(f"Generando {args.genre}/{args.mood}/{args.instrument}…")
    print(f"  temperature={args.temperature}  top_p={args.top_p}  "
          f"top_k={args.top_k}  repetition_penalty={args.repetition_penalty}")

    # generate() devuelve una lista de IDs de tokens (incluyendo el prompt)
    gen_ids = generate(
        model, enc_ids, enc_mask, prompt, config, device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"  {len(gen_ids)} tokens generados")

    # ── Exportar acompañamiento a MIDI (opcional) ──────────────────────
    if args.output_midi:
        from data.midi_tokenizer import TARGET_TEMPO, SECONDS_PER_TICK
        inst_map = {"BASS": 33, "PIANO": 0, "GUITAR": 25}
        program  = inst_map.get(args.instrument.upper(), 33)
        pm_out   = decode_event_tokens_to_midi(gen_ids, instrument_program=program)

        if args.include_melody and melody_inst is not None:
            # Normalize melody timestamps from original BPM to TARGET_TEMPO (120 BPM)
            scale = TARGET_TEMPO / tempo_bpm
            mel_pm_inst = pretty_midi.Instrument(
                program=melody_inst.program,
                name=melody_inst.name or "Melody",
            )
            for orig_note in melody_inst.notes:
                mel_pm_inst.notes.append(pretty_midi.Note(
                    velocity=orig_note.velocity,
                    pitch=orig_note.pitch,
                    start=orig_note.start * scale,
                    end=orig_note.end   * scale,
                ))
            pm_out.instruments.insert(0, mel_pm_inst)

        pm_out.write(args.output_midi)
        n_notes = sum(len(i.notes) for i in pm_out.instruments)
        print(f"  MIDI guardado en: {args.output_midi} ({n_notes} notas)")

    # ── Convertir a MusicXML ───────────────────────────────────────────
    # tokens_to_musicxml recibe los tokens como strings para el encoder
    # y como IDs para el decoder (event-based), y hace la conversión internamente.
    print("Convirtiendo a MusicXML…")
    tokens_to_musicxml(
        enc_tokens=enc_tokens,            # lista de strings (encoder, posicional)
        dec_token_ids=gen_ids,            # lista de ints (decoder, event-based)
        output_path=args.output,
        tempo_bpm=tempo_bpm,
        melody_name="Melodía",
        accomp_name=f"{args.instrument.capitalize()} ({args.genre})",
    )


if __name__ == "__main__":
    main()
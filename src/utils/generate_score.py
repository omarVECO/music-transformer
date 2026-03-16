# src/utils/generate_score.py
"""
Pipeline completo: MIDI de entrada → inferencia → MusicXML

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
from model.inference import generate, top_p_sampling, apply_repetition_penalty, apply_pitch_range_boost
from data.midi_tokenizer import (
    detect_key, select_tracks, notes_to_token_sequence,
    inst_to_token, TOKEN2ID, ID2TOKEN, MIN_PITCH, MAX_PITCH
)
from utils.tokens_to_musicxml import tokens_to_musicxml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_midi",  required=True)
    parser.add_argument("--genre",       default="FUNK",
                        choices=["ROCK","POP","FUNK","JAZZ","LATIN","CLASSICAL","ELECTRONIC"])
    parser.add_argument("--mood",        default="HAPPY",
                        choices=["HAPPY","SAD","DARK","RELAXED","TENSE"])
    parser.add_argument("--instrument",  default="BASS",
                        choices=["BASS","PIANO","GUITAR"])
    parser.add_argument("--output",      default="partitura.xml")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p",       type=float, default=0.92)
    parser.add_argument("--max_tokens",  type=int,   default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Cargar modelo ─────────────────────────────────────────
    config = ModelConfig()
    model  = MusicTransformer(config).to(device)
    ckpt   = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Modelo: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # ── Tokenizar entrada ─────────────────────────────────────
    pm = pretty_midi.PrettyMIDI(args.input_midi)
    melody_inst, _ = select_tracks(pm)
    if melody_inst is None:
        melody_inst = next((i for i in pm.instruments if not i.is_drum), None)
    if melody_inst is None:
        raise ValueError("No se encontró pista melódica")

    tempo_bpm = pm.estimate_tempo()
    if not (30 < tempo_bpm < 300):
        tempo_bpm = 120.0
    key_token = detect_key(pm)
    print(f"Entrada: {key_token}  {tempo_bpm:.0f} BPM")

    enc_tokens = notes_to_token_sequence(
        melody_inst, pm, tempo_bpm, key_token,
        args.genre, args.mood, 0.5,
        inst_to_token(melody_inst), is_encoder=True
    )
    enc_tokens = enc_tokens[:config.max_seq_len]

    enc_ids  = torch.tensor(
        [TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in enc_tokens],
        dtype=torch.long
    )
    enc_mask = torch.ones(len(enc_ids), dtype=torch.bool)
    pad_len  = config.max_seq_len - len(enc_ids)
    enc_ids  = torch.cat([enc_ids,  torch.zeros(pad_len, dtype=torch.long)])
    enc_mask = torch.cat([enc_mask, torch.zeros(pad_len, dtype=torch.bool)])

    # ── Generar acompañamiento ────────────────────────────────
    prompt = [
        TOKEN2ID["<SOS>"],
        TOKEN2ID[f"<GENRE_{args.genre}>"],
        TOKEN2ID[f"<MOOD_{args.mood}>"],
        TOKEN2ID["<ENERGY_MED>"],
        TOKEN2ID[f"<INST_{args.instrument}>"],
    ]
    print(f"Generando {args.genre}/{args.mood}/{args.instrument}...")

    gen_ids = generate(
        model, enc_ids, enc_mask, prompt, config, device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    dec_tokens = [ID2TOKEN.get(i, "<UNK>") for i in gen_ids]
    print(f"  {len(dec_tokens)} tokens generados")

    # ── Convertir a MusicXML ──────────────────────────────────
    print("Convirtiendo a MusicXML...")
    tokens_to_musicxml(
        enc_tokens=enc_tokens,
        dec_tokens=dec_tokens,
        output_path=args.output,
        melody_name="Melodía",
        accomp_name=f"{args.instrument.capitalize()} ({args.genre})"
    )

if __name__ == "__main__":
    main()
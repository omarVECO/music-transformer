import argparse
import torch
import pretty_midi
import numpy as np
from pathlib import Path

from model.config import ModelConfig
from model.transformer import MusicTransformer
from data.midi_tokenizer import (
    detect_key, select_tracks, notes_to_token_sequence,
    inst_to_token, TOKEN2ID, ID2TOKEN, PPQ, TICKS_PER_BAR,
    MIN_PITCH, MAX_PITCH, VELOCITY_BINS
)

def top_p_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs  = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative - sorted_probs > p] = 0
    sorted_probs /= sorted_probs.sum()
    chosen = torch.multinomial(sorted_probs, 1)
    return sorted_idx[chosen]

def apply_repetition_penalty(logits, generated_ids, penalty=1.3):
    """
    Reduce la probabilidad de tokens ya generados recientemente.
    penalty > 1.0 = penaliza repetición
    Solo aplica a tokens de pitch para no afectar la estructura.
    """
    pitch_ids = {TOKEN2ID[f"<PITCH_{i}>"] for i in range(MIN_PITCH, MAX_PITCH + 1)}
    seen = set(generated_ids[-32:])  # ventana de los últimos 32 tokens
    for tid in seen:
        if tid in pitch_ids:
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty
    return logits

def apply_pitch_range_boost(logits, generated_ids, boost=2.0):
    """
    Si el modelo lleva muchas notas en el mismo rango de octava,
    boost las probabilidades de notas fuera de ese rango.
    """
    pitch_ids = {TOKEN2ID[f"<PITCH_{i}>"]: i
                 for i in range(MIN_PITCH, MAX_PITCH + 1)}

    recent_pitches = [
        pitch_ids[tid]
        for tid in generated_ids[-48:]
        if tid in pitch_ids
    ]
    if len(recent_pitches) < 4:
        return logits

    avg_pitch = sum(recent_pitches) / len(recent_pitches)

    for tid, pitch in pitch_ids.items():
        distance = abs(pitch - avg_pitch)
        if distance > 7:  # más de una quinta desde el promedio
            logits[tid] *= boost

    return logits

@torch.no_grad()
def generate(model, enc_ids, enc_mask, prompt_ids, config, device,
             max_new_tokens=1024, temperature=0.9, top_p=0.9):
    model.eval()
    enc_ids  = enc_ids.unsqueeze(0).to(device)
    enc_mask = enc_mask.unsqueeze(0).to(device)
    memory   = model.encode(enc_ids, enc_mask)
    gen_ids  = list(prompt_ids)
    eos_id   = TOKEN2ID["<EOS>"]

    for _ in range(max_new_tokens):
        tgt      = torch.tensor([gen_ids], dtype=torch.long, device=device)
        tgt_mask = torch.ones(1, len(gen_ids), dtype=torch.bool, device=device)

        if tgt.size(1) > config.max_seq_len:
            tgt      = tgt[:, -config.max_seq_len:]
            tgt_mask = tgt_mask[:, -config.max_seq_len:]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model.decode(tgt, memory, tgt_mask, enc_mask)

        next_logits = logits[0, -1, :]
        next_logits[TOKEN2ID["<PAD>"]] = -float("inf")
        next_logits[TOKEN2ID["<UNK>"]] = -float("inf")

        next_logits = apply_repetition_penalty(
            next_logits, gen_ids, penalty=1.3
        )
        next_logits = apply_pitch_range_boost(next_logits, gen_ids, boost=2.0)

        next_id = top_p_sampling(next_logits, p=top_p, temperature=temperature).item()
        gen_ids.append(next_id)

        if next_id == eos_id:
            break

    return gen_ids

def tokens_to_midi(token_ids, tempo_bpm=120.0, instrument_program=33):
    pm   = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=instrument_program)

    seconds_per_tick = 60.0 / (tempo_bpm * PPQ)
    current_bar   = 0
    current_pos   = 0
    current_pitch = None
    current_dur   = None
    current_vel   = None

    DUR_TICKS = {
        "<DUR_1>":1,"<DUR_2>":2,"<DUR_3>":3,"<DUR_4>":4,
        "<DUR_6>":6,"<DUR_8>":8,"<DUR_12>":12,"<DUR_16>":16,
        "<DUR_T2>":2,"<DUR_T4>":4
    }
    VEL_MAP = {f"<VEL_{v}>": v for v in VELOCITY_BINS}

    def flush_note():
        nonlocal current_pitch, current_dur, current_vel
        if current_pitch is None or current_dur is None or current_vel is None:
            return
        tick_start = current_bar * TICKS_PER_BAR + current_pos
        tick_end   = tick_start + current_dur
        t_start    = tick_start * seconds_per_tick
        t_end      = tick_end   * seconds_per_tick
        if MIN_PITCH <= current_pitch <= MAX_PITCH and t_end > t_start:
            inst.notes.append(pretty_midi.Note(
                velocity=current_vel, pitch=current_pitch,
                start=t_start, end=t_end
            ))
        current_pitch = None
        current_dur   = None
        current_vel   = None

    skip_prefixes = (
        "<GENRE_","<MOOD_","<ENERGY_","<INST_",
        "<TIMESIG_","<KEY_","<TEMPO_","<CHORD_","<BEAT_"
    )

    for tid in token_ids:
        tok = ID2TOKEN.get(tid, "<UNK>")

        if tok in ("<SOS>","<EOS>","<PAD>","<UNK>","<SEP>","<MASK>"):
            continue
        if any(tok.startswith(p) for p in skip_prefixes):
            continue

        if tok.startswith("<BAR_"):
            flush_note()
            current_bar = int(tok[5:-1]) - 1
            current_pos = 0

        elif tok.startswith("<POS_"):
            flush_note()
            current_pos = int(tok[5:-1])

        elif tok.startswith("<PITCH_"):
            flush_note()
            current_pitch = int(tok[7:-1])

        elif tok == "<REST>":
            flush_note()
            current_pitch = None

        elif tok in DUR_TICKS:
            current_dur = DUR_TICKS[tok]

        elif tok in VEL_MAP:
            current_vel = VEL_MAP[tok]
            flush_note()

    flush_note()
    pm.instruments.append(inst)
    return pm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_midi",  required=True)
    parser.add_argument("--genre",       default="FUNK",
                        choices=["ROCK","POP","FUNK","JAZZ","LATIN","CLASSICAL","ELECTRONIC"])
    parser.add_argument("--mood",        default="HAPPY",
                        choices=["HAPPY","SAD","DARK","RELAXED","TENSE"])
    parser.add_argument("--instrument",  default="BASS",
                        choices=["BASS","PIANO","GUITAR"])
    parser.add_argument("--output",      default="output_accompaniment.mid")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--max_tokens",  type=int,   default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = ModelConfig()
    model  = MusicTransformer(config).to(device)
    ckpt   = torch.load("checkpoints/best_model_cic.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Modelo cargado — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    pm = pretty_midi.PrettyMIDI(args.input_midi)
    melody_inst, _ = select_tracks(pm)
    if melody_inst is None:
        melody_inst = next((i for i in pm.instruments if not i.is_drum), None)
    if melody_inst is None:
        raise ValueError("El MIDI no tiene pistas utilizables")

    tempo_bpm = pm.estimate_tempo()
    if not (30 < tempo_bpm < 300):
        tempo_bpm = 120.0
    key_token = detect_key(pm)
    print(f"Tonalidad: {key_token}  Tempo: {tempo_bpm:.0f} BPM")

    enc_tokens = notes_to_token_sequence(
        melody_inst, pm, tempo_bpm, key_token,
        args.genre, args.mood, 0.5,
        inst_to_token(melody_inst), is_encoder=True
    )
    if enc_tokens is None:
        raise ValueError("Melodía demasiado corta")

    enc_tokens = enc_tokens[:config.max_seq_len]
    enc_ids    = torch.tensor(
        [TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in enc_tokens],
        dtype=torch.long
    )
    enc_mask = torch.ones(len(enc_ids), dtype=torch.bool)

    pad_len  = config.max_seq_len - len(enc_ids)
    enc_ids  = torch.cat([enc_ids,  torch.zeros(pad_len, dtype=torch.long)])
    enc_mask = torch.cat([enc_mask, torch.zeros(pad_len, dtype=torch.bool)])

    prompt = [
        TOKEN2ID["<SOS>"],
        TOKEN2ID[f"<GENRE_{args.genre}>"],
        TOKEN2ID[f"<MOOD_{args.mood}>"],
        TOKEN2ID["<ENERGY_MED>"],
        TOKEN2ID[f"<INST_{args.instrument}>"],
    ]

    print(f"Generando — género:{args.genre} mood:{args.mood} inst:{args.instrument}")
    print(f"  temperature={args.temperature}  top_p={args.top_p}  max_tokens={args.max_tokens}")

    gen_ids = generate(
        model, enc_ids, enc_mask, prompt, config, device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"  Tokens generados: {len(gen_ids)}")

    INST_PROGRAMS = {"BASS": 33, "PIANO": 0, "GUITAR": 25}
    output_pm = tokens_to_midi(gen_ids, tempo_bpm=tempo_bpm,
                                instrument_program=INST_PROGRAMS[args.instrument])

    if not output_pm.instruments or len(output_pm.instruments[0].notes) == 0:
        print("Sin notas — prueba con temperature=0.7 o top_p=0.85")
    else:
        n     = len(output_pm.instruments[0].notes)
        dur   = output_pm.get_end_time()
        print(f"  Notas generadas: {n}  Duración: {dur:.1f}s")
        output_pm.write(args.output)
        print(f"  Guardado en: {args.output}")

if __name__ == "__main__":
    main()
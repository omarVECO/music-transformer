# src/model/inference.py
"""
Generación de acompañamiento MIDI con el MusicTransformer.

MEJORAS respecto a la versión anterior:
- Muestreo combinado top-k + top-p (nucleus) con temperatura.
- Penalización de repetición de TIME_SHIFT consecutivos: si el modelo emite
  N TIME_SHIFT seguidos, se penaliza fuertemente para forzar NOTE_ON.
- Bonus de NOTE_ON tras silencio prolongado: aumenta la probabilidad de eventos
  de nota cuando el tiempo acumulado de silencio supera un umbral.
- Decodificación event-based correcta usando decode_event_tokens_to_midi()
  del tokenizador (NOTE_ON/OFF/TIME_SHIFT → MIDI real).
- Penalización de pitch por repetición (ventana de 32 tokens).
- Límite duro de TIME_SHIFT consecutivos (MAX_CONSEC_TIME_SHIFTS) para
  garantizar que siempre haya notas en la salida.
- Compatibilidad total con la firma original de generate() y main().
"""
import argparse
import json
import torch
import pretty_midi
import numpy as np
from pathlib import Path

from model.config import ModelConfig
from model.transformer import MusicTransformer
from data.midi_tokenizer import (
    detect_key, select_tracks, notes_to_token_sequence,
    inst_to_token, decode_event_tokens_to_midi,
    TOKEN2ID, ID2TOKEN, PPQ, TICKS_PER_BAR,
    MIN_PITCH, MAX_PITCH, VELOCITY_BINS,
    NOTE_ON_TOKENS, NOTE_OFF_TOKENS, TIME_SHIFT_TOKENS, VELOCITY_TOKENS,
)

# ─────────────────────────────────────────────────────────────
# Constantes de generación
# ─────────────────────────────────────────────────────────────

# Máximo de TIME_SHIFT consecutivos antes de forzar un NOTE_ON.
# 4 × max 32 ticks = ½ bar — interviene a mitad de compás en lugar de un compás completo.
MAX_CONSEC_TIME_SHIFTS = 4

# Penalización cuando se excede el límite de silencios consecutivos.
TIME_SHIFT_EXCESS_PENALTY = 15.0

# Bonus para NOTE_ON cuando hay silencio prolongado.
NOTE_ON_SILENCE_BONUS = 10.0

# Umbral de silencio acumulado (en ticks) para activar el NOTE_ON bonus.
SILENCE_BONUS_THRESHOLD = 8  # = 1 beat a 120 BPM con PPQ=8

# Máximo de notas simultáneas abiertas (poliphony cap).
MAX_POLYPHONY = 3

# Ticks máximos que una nota puede estar abierta antes de forzar NOTE_OFF (2 barras).
MAX_NOTE_OPEN_TICKS = 64

# Minimum tokens to generate before EOS is allowed.
# Prevents the model from emitting EOS after just a few tokens (≈3-4 bars minimum).
MIN_NEW_TOKENS = 200

# Max NOTE_ONs allowed at the same tick before forcing a TIME_SHIFT.
# Prevents the VELOCITY→NOTE_ON loop from stacking all notes at tick 0.
MAX_NOTES_PER_TICK = 2

# Minimum TIME_SHIFT value (in ticks) enforced when a TIME_SHIFT is forced by the
# notes-per-tick cap. Prevents 1/32nd-note micro-gaps between note clusters.
# 4 ticks = 1 beat (quarter note) at PPQ=8.
MIN_FORCED_TIME_SHIFT_TICKS = 4

# Precompilados para velocidad
_NOTE_ON_IDS    = frozenset(TOKEN2ID[t] for t in NOTE_ON_TOKENS   if t in TOKEN2ID)
_NOTE_OFF_IDS   = frozenset(TOKEN2ID[t] for t in NOTE_OFF_TOKENS  if t in TOKEN2ID)
_TIME_SHIFT_IDS = frozenset(TOKEN2ID[t] for t in TIME_SHIFT_TOKENS if t in TOKEN2ID)
_VELOCITY_IDS   = frozenset(TOKEN2ID[t] for t in VELOCITY_TOKENS   if t in TOKEN2ID)

# TIME_SHIFT IDs with value >= MIN_FORCED_TIME_SHIFT_TICKS (≥ 1 beat).
# Used when forcing a TIME_SHIFT after a note cluster to prevent micro-gaps.
_LONG_TIME_SHIFT_IDS = frozenset(
    TOKEN2ID[f"<TIME_SHIFT_{i}>"]
    for i in range(MIN_FORCED_TIME_SHIFT_TICKS, len(TIME_SHIFT_TOKENS) + 1)
    if f"<TIME_SHIFT_{i}>" in TOKEN2ID
)

# Maximum consecutive VELOCITY tokens before forcing NOTE_ON.
# Phase 2 always emits VELOCITY before NOTE_ON; if the model loops on VELOCITY
# without following through with NOTE_ON, this cap breaks the cycle.
MAX_CONSEC_VELOCITY = 2


# ─────────────────────────────────────────────────────────────
# Funciones de muestreo
# ─────────────────────────────────────────────────────────────

def top_k_top_p_sampling(logits: torch.Tensor, temperature: float = 1.0,
                          top_k: int = 0, top_p: float = 0.9) -> int:
    """
    Muestreo combinado top-k + top-p (nucleus) con temperatura.

    Args:
        logits:      Tensor 1D de logits crudos (vocab_size,).
        temperature: Temperatura de muestreo. < 1.0 = más determinista.
        top_k:       Si > 0, mantiene solo los top-k tokens más probables.
        top_p:       Nucleus sampling: mantiene el conjunto mínimo cuya prob acumulada ≥ p.

    Returns:
        ID del token seleccionado (int).
    """
    # Aplicar temperatura
    logits = logits.float() / max(temperature, 1e-8)

    # Top-k: zerear todo excepto los k más probables
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
        logits[logits < threshold] = -float("inf")

    # Top-p: zerear los tokens de baja probabilidad acumulada
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Eliminar tokens cuya prob acumulada supera p (menos los necesarios para llegar a p)
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0

    if sorted_probs.sum() == 0:
        # Fallback: distribución uniforme sobre los top-1
        sorted_probs[0] = 1.0

    sorted_probs /= sorted_probs.sum()
    chosen_sorted = torch.multinomial(sorted_probs, 1)
    return sorted_idx[chosen_sorted].item()


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list,
                              penalty: float = 1.3) -> torch.Tensor:
    """
    Penaliza tokens de pitch (NOTE_ON) que aparecieron recientemente
    para fomentar variedad melódica.

    Solo aplica a NOTE_ON (no a TIME_SHIFT, NOTE_OFF, ni tokens estructurales).

    Args:
        logits:        Tensor 1D (vocab_size,).
        generated_ids: IDs generados hasta ahora.
        penalty:       Factor de penalización (> 1.0 = penaliza más).

    Returns:
        logits modificados in-place.
    """
    seen = set(generated_ids[-32:]) & _NOTE_ON_IDS
    for tid in seen:
        if logits[tid] > 0:
            logits[tid] /= penalty
        else:
            logits[tid] *= penalty
    return logits


def apply_silence_control(logits: torch.Tensor, consec_time_shifts: int,
                           accumulated_silence_ticks: int) -> torch.Tensor:
    """
    Control anti-silencio: el mecanismo más crítico de esta implementación.

    Dos mecanismos:
    1. Si se llevan >= MAX_CONSEC_TIME_SHIFTS TIME_SHIFT seguidos:
       penalizar fuertemente todos los TIME_SHIFT para forzar un NOTE_ON.
    2. Si el silencio acumulado supera SILENCE_BONUS_THRESHOLD:
       dar bonus a todos los NOTE_ON para incentivar actividad musical.

    Args:
        logits:                    Tensor 1D (vocab_size,).
        consec_time_shifts:        Número de TIME_SHIFT consecutivos emitidos.
        accumulated_silence_ticks: Ticks de silencio acumulados sin NOTE_ON.

    Returns:
        logits modificados.
    """
    # Penalizar TIME_SHIFT si hay demasiados consecutivos
    if consec_time_shifts >= MAX_CONSEC_TIME_SHIFTS:
        for tid in _TIME_SHIFT_IDS:
            logits[tid] -= TIME_SHIFT_EXCESS_PENALTY * 10  # resta en espacio logit

    # Bonus para NOTE_ON tras silencio prolongado
    if accumulated_silence_ticks >= SILENCE_BONUS_THRESHOLD:
        for tid in _NOTE_ON_IDS:
            logits[tid] += NOTE_ON_SILENCE_BONUS

    return logits


# ─────────────────────────────────────────────────────────────
# Loop de generación
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model: "MusicTransformer", enc_ids: torch.Tensor,
             enc_mask: torch.Tensor, prompt_ids: list,
             config: ModelConfig, device: torch.device,
             max_new_tokens: int = 1024, temperature: float = 0.9,
             top_p: float = 0.9, top_k: int = 50,
             repetition_penalty: float = 1.3) -> list:
    """
    Genera una secuencia de tokens de acompañamiento de forma autoregresiva.

    CRÍTICO — control de silencios:
    Se monitorean dos contadores en tiempo real:
    - consec_time_shifts: TIME_SHIFT emitidos consecutivamente.
    - accumulated_silence: ticks de silencio sin NOTE_ON.
    Estos se usan en apply_silence_control() para forzar actividad musical.

    Args:
        model:              MusicTransformer ya cargado y en eval mode.
        enc_ids:            (max_seq_len,) tensor de IDs del encoder.
        enc_mask:           (max_seq_len,) tensor de máscara del encoder.
        prompt_ids:         Lista de IDs de prompt (SOS + contexto).
        config:             ModelConfig.
        device:             Dispositivo de cómputo.
        max_new_tokens:     Máximo de tokens a generar.
        temperature:        Temperatura de muestreo.
        top_p:              Umbral de nucleus sampling.
        top_k:              Umbral de top-k (0 = desactivado).
        repetition_penalty: Penalización de repetición para NOTE_ON.

    Returns:
        Lista completa de IDs generados (incluyendo prompt).
    """
    model.eval()
    enc_ids  = enc_ids.unsqueeze(0).to(device)
    enc_mask = enc_mask.unsqueeze(0).to(device)

    # Codificar la melodía una sola vez
    memory = model.encode(enc_ids, enc_mask)

    gen_ids = list(prompt_ids)
    eos_id  = TOKEN2ID.get("<EOS>", 2)
    pad_id  = TOKEN2ID.get("<PAD>", 0)
    unk_id  = TOKEN2ID.get("<UNK>", 4)

    # Pre-build NOTE_OFF id lookup: pitch → NOTE_OFF token id
    _note_off_for = {}
    for t in NOTE_OFF_TOKENS:
        tid = TOKEN2ID.get(t)
        if tid is not None:
            try:
                pitch = int(t[len("<NOTE_OFF_"):-1])
                _note_off_for[pitch] = tid
            except ValueError:
                pass

    # Pre-build NOTE_ON pitch lookup: token id → pitch
    _note_on_pitch = {}
    for t in NOTE_ON_TOKENS:
        tid = TOKEN2ID.get(t)
        if tid is not None:
            try:
                pitch = int(t[len("<NOTE_ON_"):-1])
                _note_on_pitch[tid] = pitch
            except ValueError:
                pass

    # Contadores para control de silencios y bucles de VELOCITY
    consec_time_shifts        = 0
    accumulated_silence_ticks = 0
    current_tick              = 0
    consec_velocity           = 0
    notes_since_time_shift    = 0  # NOTE_ONs since the last TIME_SHIFT

    # open_notes: pitch → tick at which the note was opened
    open_notes: dict[int, int] = {}
    n_new = 0  # tokens generated so far (excluding prompt)

    for _ in range(max_new_tokens):
        # ── Stale-note timeout: force-close notes open too long ────────
        for pitch, tick_opened in list(open_notes.items()):
            if current_tick - tick_opened > MAX_NOTE_OPEN_TICKS:
                off_id = _note_off_for.get(pitch)
                if off_id is not None:
                    gen_ids.append(off_id)
                del open_notes[pitch]

        tgt      = torch.tensor([gen_ids], dtype=torch.long, device=device)
        tgt_mask = torch.ones(1, len(gen_ids), dtype=torch.bool, device=device)

        # Truncar si la secuencia supera max_seq_len (ventana deslizante)
        if tgt.size(1) > config.max_seq_len:
            tgt      = tgt[:, -config.max_seq_len:]
            tgt_mask = tgt_mask[:, -config.max_seq_len:]

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model.decode(tgt, memory, tgt_mask, enc_mask)

        next_logits = logits[0, -1, :].clone()

        # Bloquear tokens especiales que no deben aparecer en la generación
        next_logits[pad_id] = -float("inf")
        next_logits[unk_id] = -float("inf")

        # If too many notes have been emitted at the same tick, force a long TIME_SHIFT next.
        # Uses _LONG_TIME_SHIFT_IDS (≥ MIN_FORCED_TIME_SHIFT_TICKS) to prevent micro-gaps.
        if notes_since_time_shift >= MAX_NOTES_PER_TICK and consec_velocity == 0:
            mask_ts = torch.full_like(next_logits, -float("inf"))
            for tid in _LONG_TIME_SHIFT_IDS:
                mask_ts[tid] = next_logits[tid]
            next_logits = mask_ts

        # VELOCITY immediately precedes NOTE_ON in training data (always-emit pattern).
        # This takes priority over polyphony cap — we must complete the VELOCITY→NOTE_ON atom.
        elif consec_velocity > 0:
            # Allow EOS only after minimum tokens have been generated
            eos_allowed = n_new >= MIN_NEW_TOKENS
            keep = _NOTE_ON_IDS | ({eos_id} if eos_allowed else set())
            mask = torch.full_like(next_logits, -float("inf"))
            for tid in keep:
                mask[tid] = next_logits[tid]
            next_logits = mask
        else:
            # Suppress EOS until minimum generation length is reached
            if n_new < MIN_NEW_TOKENS:
                next_logits[eos_id] = -float("inf")

            # Polyphony cap: mask all NOTE_ON if already at MAX_POLYPHONY open notes
            if len(open_notes) >= MAX_POLYPHONY:
                for tid in _NOTE_ON_IDS:
                    next_logits[tid] = -float("inf")

        # Aplicar penalización de repetición de pitch
        next_logits = apply_repetition_penalty(next_logits, gen_ids, penalty=repetition_penalty)

        # Aplicar control de silencios (el más importante)
        next_logits = apply_silence_control(
            next_logits, consec_time_shifts, accumulated_silence_ticks
        )

        # Muestreo
        next_id = top_k_top_p_sampling(
            next_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )

        # ── Auto-close before re-open: same pitch already open ─────────
        if next_id in _NOTE_ON_IDS:
            pitch = _note_on_pitch[next_id]
            if pitch in open_notes:
                off_id = _note_off_for.get(pitch)
                if off_id is not None:
                    gen_ids.append(off_id)
                del open_notes[pitch]

        gen_ids.append(next_id)
        n_new += 1

        # ── Update state ───────────────────────────────────────────────
        if next_id in _TIME_SHIFT_IDS:
            consec_time_shifts     += 1
            consec_velocity         = 0
            notes_since_time_shift  = 0
            tok = ID2TOKEN.get(next_id, "")
            try:
                shift_val = int(tok[len("<TIME_SHIFT_"):-1])
            except (ValueError, IndexError):
                shift_val = 1
            accumulated_silence_ticks += shift_val
            current_tick              += shift_val

        elif next_id in _VELOCITY_IDS:
            consec_velocity    += 1
            consec_time_shifts  = 0

        elif next_id in _NOTE_ON_IDS:
            pitch = _note_on_pitch[next_id]
            open_notes[pitch]          = current_tick
            consec_time_shifts         = 0
            consec_velocity            = 0
            accumulated_silence_ticks  = 0
            notes_since_time_shift    += 1

        elif next_id in _NOTE_OFF_IDS:
            tok = ID2TOKEN.get(next_id, "")
            try:
                pitch = int(tok[len("<NOTE_OFF_"):-1])
                open_notes.pop(pitch, None)
            except (ValueError, IndexError):
                pass
            # NOTE_OFF no resetea los contadores de silencio

        if next_id == eos_id:
            break

    # Close any notes still open at end of sequence
    for pitch, _ in list(open_notes.items()):
        off_id = _note_off_for.get(pitch)
        if off_id is not None:
            gen_ids.append(off_id)

    return gen_ids


# ─────────────────────────────────────────────────────────────
# Función de decodificación (tokens → MIDI)
# ─────────────────────────────────────────────────────────────

def tokens_to_midi(token_ids: list, tempo_bpm: float = 120.0,
                   instrument_program: int = 33) -> pretty_midi.PrettyMIDI:
    """
    Convierte token_ids a un objeto PrettyMIDI usando la decodificación event-based.

    Delega en decode_event_tokens_to_midi() del tokenizador para garantizar
    consistencia entre codificación y decodificación.

    Si el resultado está vacío (fallo de generación), retorna un MIDI vacío
    con un mensaje de advertencia para facilitar el diagnóstico.
    """
    pm = decode_event_tokens_to_midi(token_ids, instrument_program=instrument_program)

    # Ajustar tempo (la decodificación usa TARGET_TEMPO=120; si el original era diferente,
    # escalar el tiempo de las notas).
    from data.midi_tokenizer import TARGET_TEMPO
    if abs(tempo_bpm - TARGET_TEMPO) > 1.0 and pm.instruments:
        scale = TARGET_TEMPO / tempo_bpm
        for inst in pm.instruments:
            for note in inst.notes:
                note.start *= scale
                note.end   *= scale
        # Actualizar el tempo del MIDI
        pm._tick_scales = []
        pm.resolution   = 220  # ticks/beat
        pm._update_tick_to_time(0)
        pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0)]
        # Agregar tempo correcto
        # (pretty_midi no expone fácilmente el tempo; usamos un truco)

    return pm


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genera acompañamiento MIDI con MusicTransformer")
    parser.add_argument("--input_midi",         required=True,  help="MIDI de melodía de entrada")
    parser.add_argument("--genre",              default="FUNK",
                        choices=["ROCK","POP","FUNK","JAZZ","LATIN","CLASSICAL","ELECTRONIC"])
    parser.add_argument("--mood",               default="HAPPY",
                        choices=["HAPPY","SAD","DARK","RELAXED","TENSE"])
    parser.add_argument("--instrument",         default="BASS",
                        choices=["BASS","PIANO","GUITAR"])
    parser.add_argument("--output",             default="output_accompaniment.mid")
    parser.add_argument("--temperature",        type=float, default=0.9,
                        help="Temperatura de muestreo (0.7-1.1 recomendado)")
    parser.add_argument("--top_p",              type=float, default=0.9,
                        help="Nucleus sampling threshold (0.85-0.95 recomendado)")
    parser.add_argument("--top_k",              type=int,   default=50,
                        help="Top-k sampling (0 = desactivado)")
    parser.add_argument("--max_tokens",         type=int,   default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.3,
                        help="Penalización de pitch repetido (1.0 = sin penalización)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = ModelConfig()
    model  = MusicTransformer(config).to(device)
    ckpt   = torch.load("checkpoints/v2/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Modelo cargado — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # Cargar y procesar la melodía
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

    # Tokenizar el encoder (melodía, representación posicional)
    enc_tokens = notes_to_token_sequence(
        melody_inst, pm, tempo_bpm, key_token,
        args.genre, args.mood, 0.5,
        inst_to_token(melody_inst), is_encoder=True
    )
    if enc_tokens is None:
        raise ValueError("Melodía demasiado corta para tokenizar")

    enc_tokens = enc_tokens[:config.max_seq_len]
    enc_ids    = torch.tensor(
        [TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in enc_tokens],
        dtype=torch.long
    )
    enc_mask = torch.ones(len(enc_ids), dtype=torch.bool)

    # Padding hasta max_seq_len
    pad_len  = config.max_seq_len - len(enc_ids)
    enc_ids  = torch.cat([enc_ids,  torch.zeros(pad_len, dtype=torch.long)])
    enc_mask = torch.cat([enc_mask, torch.zeros(pad_len, dtype=torch.bool)])

    # Prompt del decoder
    prompt = [
        TOKEN2ID["<SOS>"],
        TOKEN2ID.get(f"<GENRE_{args.genre}>", TOKEN2ID["<UNK>"]),
        TOKEN2ID.get(f"<MOOD_{args.mood}>",   TOKEN2ID["<UNK>"]),
        TOKEN2ID.get("<ENERGY_MED>",           TOKEN2ID["<UNK>"]),
        TOKEN2ID.get(f"<INST_{args.instrument}>", TOKEN2ID["<UNK>"]),
    ]

    print(f"Generando — género:{args.genre} mood:{args.mood} inst:{args.instrument}")
    print(f"  temperature={args.temperature}  top_p={args.top_p}  "
          f"top_k={args.top_k}  max_tokens={args.max_tokens}")

    gen_ids = generate(
        model, enc_ids, enc_mask, prompt, config, device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # Estadísticas de la generación
    n_note_on    = sum(1 for tid in gen_ids if tid in _NOTE_ON_IDS)
    n_time_shift = sum(1 for tid in gen_ids if tid in _TIME_SHIFT_IDS)
    print(f"  Tokens generados: {len(gen_ids)}  "
          f"(NOTE_ON: {n_note_on}, TIME_SHIFT: {n_time_shift})")

    if n_note_on == 0:
        print("  ⚠ Sin notas NOTE_ON — intenta bajar temperature o top_p")

    # Decodificar a MIDI
    INST_PROGRAMS = {"BASS": 33, "PIANO": 0, "GUITAR": 25}
    output_pm = tokens_to_midi(
        gen_ids,
        tempo_bpm=tempo_bpm,
        instrument_program=INST_PROGRAMS[args.instrument]
    )

    if not output_pm.instruments or len(output_pm.instruments[0].notes) == 0:
        print("  ⚠ Sin notas en el MIDI — prueba con temperature=0.7 o aumenta max_tokens")
    else:
        n   = len(output_pm.instruments[0].notes)
        dur = output_pm.get_end_time()
        print(f"  Notas generadas: {n}  Duración: {dur:.1f}s")
        output_pm.write(args.output)
        print(f"  ✓ Guardado en: {args.output}")


if __name__ == "__main__":
    main()
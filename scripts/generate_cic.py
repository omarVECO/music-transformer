"""
Self-contained generation pipeline for the CIC checkpoint
(checkpoints/best_model_cic.pt).

This checkpoint uses a custom architecture and the original 355-token
positional vocabulary — neither of which matches the current codebase.
Everything needed (vocab, model, tokenizer, MIDI decoder) is embedded here
so this script can run without touching any current source files.

Usage (from project root):
    python scripts/generate_cic.py --input_midi <path.mid> \
        --genre FUNK --mood HAPPY --instrument BASS --output out.mid

Batch usage:
    python scripts/batch_generate_cic.py
"""

import argparse
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# 1. Vocabulary (355 tokens, original positional scheme)
# ─────────────────────────────────────────────────────────────────────────────

PPQ           = 16        # ticks per quarter note
TICKS_PER_BAR = 64        # 4/4 at PPQ=16
MAX_BARS      = 32
MIN_BARS      = 4
MAX_PITCH     = 108
MIN_PITCH     = 28
VELOCITY_BINS = [16, 32, 48, 64, 80, 96, 112, 127]

SPECIAL   = ["<PAD>", "<SOS>", "<EOS>", "<SEP>", "<UNK>", "<MASK>"]
GENRES    = ["<GENRE_ROCK>", "<GENRE_POP>", "<GENRE_FUNK>", "<GENRE_JAZZ>",
             "<GENRE_LATIN>", "<GENRE_CLASSICAL>", "<GENRE_ELECTRONIC>"]
MOODS     = ["<MOOD_HAPPY>", "<MOOD_SAD>", "<MOOD_DARK>", "<MOOD_RELAXED>", "<MOOD_TENSE>"]
ENERGIES  = ["<ENERGY_LOW>", "<ENERGY_MED>", "<ENERGY_HIGH>"]
TIMESIGS  = ["<TIMESIG_3_4>", "<TIMESIG_4_4>", "<TIMESIG_6_8>"]
KEYS      = [f"<KEY_{n}_{m}>" for n in
             ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
             for m in ["MAJ","MIN"]]
TEMPOS    = ["<TEMPO_60>","<TEMPO_80>","<TEMPO_100>","<TEMPO_120>",
             "<TEMPO_140>","<TEMPO_160>","<TEMPO_180>","<TEMPO_200>"]
BARS      = [f"<BAR_{i}>" for i in range(1, MAX_BARS + 1)]
BEATS     = [f"<BEAT_{i}>" for i in range(1, 5)]
POSITIONS = [f"<POS_{i}>" for i in range(TICKS_PER_BAR)]   # 0..63
INSTS     = ["<INST_PIANO>", "<INST_BASS>", "<INST_GUITAR>"]
PITCHES   = [f"<PITCH_{i}>" for i in range(MIN_PITCH, MAX_PITCH + 1)] + ["<REST>"]
DURATIONS = ["<DUR_1>","<DUR_2>","<DUR_3>","<DUR_4>","<DUR_6>",
             "<DUR_8>","<DUR_12>","<DUR_16>","<DUR_T2>","<DUR_T4>"]
VELOCITIES = [f"<VEL_{v}>" for v in VELOCITY_BINS]
CHORDS    = [f"<CHORD_{n}_{q}>"
             for n in ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]
             for q in ["MAJ","MIN","DIM","AUG","MAJ7","MIN7","DOM7","DIM7"]]

ALL_TOKENS = (SPECIAL + GENRES + MOODS + ENERGIES + TIMESIGS + KEYS +
              TEMPOS + BARS + BEATS + POSITIONS + INSTS + PITCHES +
              DURATIONS + VELOCITIES + CHORDS)

TOKEN2ID = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}

VOCAB_SIZE = len(TOKEN2ID)   # 355

# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom model architecture (matches checkpoint key names exactly)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN self-attention
        x2, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                key_padding_mask=key_padding_mask)
        x = x + self.drop(x2)
        # Pre-LN FFN
        x2 = self.ff2(self.drop(self.act(self.ff1(self.norm2(x)))))
        return x + self.drop(x2)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor | None = None,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN masked self-attention
        x2, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                attn_mask=tgt_mask)
        x = x + self.drop(x2)
        # Pre-LN cross-attention
        x2, _ = self.cross_attn(self.norm2(x), memory, memory,
                                 key_padding_mask=memory_key_padding_mask)
        x = x + self.drop(x2)
        # Pre-LN FFN
        x2 = self.ff2(self.drop(self.act(self.ff1(self.norm3(x)))))
        return x + self.drop(x2)


class MusicTransformerCIC(nn.Module):
    """
    Custom encoder-decoder transformer matching checkpoints/best_model_cic.pt.
    Hyper-params: vocab=355, d_model=256, n_heads=4, d_ff=1024,
                  n_enc_layers=4, n_dec_layers=4, max_seq_len=1536.
    """
    D_MODEL    = 256
    N_HEADS    = 4
    D_FF       = 1024
    N_ENC      = 4
    N_DEC      = 4
    MAX_SEQ    = 1536
    DROPOUT    = 0.0   # eval-mode; no effect, but keep architecture consistent

    def __init__(self):
        super().__init__()
        d, h, ff, e, dc = (self.D_MODEL, self.N_HEADS, self.D_FF,
                           self.N_ENC, self.N_DEC)
        self.embedding  = nn.Embedding(VOCAB_SIZE, d, padding_idx=0)
        self.pos_enc    = PositionalEncoding(d, self.MAX_SEQ, self.DROPOUT)
        self.enc_layers = nn.ModuleList([EncoderLayer(d, h, ff, self.DROPOUT)
                                         for _ in range(e)])
        self.enc_norm   = nn.LayerNorm(d)
        self.dec_layers = nn.ModuleList([DecoderLayer(d, h, ff, self.DROPOUT)
                                         for _ in range(dc)])
        self.dec_norm   = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, VOCAB_SIZE, bias=False)

    def encode(self, src_ids: torch.Tensor,
               src_mask: torch.Tensor) -> torch.Tensor:
        """src_mask: True where valid (we invert for MHA key_padding_mask)."""
        x = self.pos_enc(self.embedding(src_ids) * math.sqrt(self.D_MODEL))
        pad_mask = ~src_mask   # True = ignore in PyTorch MHA
        for layer in self.enc_layers:
            x = layer(x, key_padding_mask=pad_mask)
        return self.enc_norm(x)

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
               tgt_mask_bool: torch.Tensor,
               memory_mask: torch.Tensor) -> torch.Tensor:
        seq_len = tgt_ids.size(1)
        causal  = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tgt_ids.device),
            diagonal=1,
        )
        x = self.pos_enc(self.embedding(tgt_ids) * math.sqrt(self.D_MODEL))
        mem_pad = ~memory_mask
        for layer in self.dec_layers:
            x = layer(x, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return self.output_proj(self.dec_norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tokenizer helpers (original positional scheme)
# ─────────────────────────────────────────────────────────────────────────────

_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

MELODY_CLASSES = {"Guitar", "Piano", "Reed", "Synth Lead", "Chromatic Percussion"}
ACCOMP_CLASSES = {"Bass", "Piano", "Guitar", "Strings", "Ensemble"}

_CHORD_TEMPLATES = {
    "MAJ":  [1,0,0,0,1,0,0,1,0,0,0,0],
    "MIN":  [1,0,0,1,0,0,0,1,0,0,0,0],
    "DIM":  [1,0,0,1,0,0,1,0,0,0,0,0],
    "AUG":  [1,0,0,0,1,0,0,0,1,0,0,0],
    "MAJ7": [1,0,0,0,1,0,0,1,0,0,0,1],
    "MIN7": [1,0,0,1,0,0,0,1,0,0,1,0],
    "DOM7": [1,0,0,0,1,0,0,1,0,0,1,0],
    "DIM7": [1,0,0,1,0,0,1,0,0,1,0,0],
}
_NOTE_NAMES = ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]


def _detect_key_cic(pm: pretty_midi.PrettyMIDI) -> str:
    pitch_classes = np.zeros(12)
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                pitch_classes[note.pitch % 12] += note.end - note.start
    best_key, best_mode, best_score = 0, "MAJ", -np.inf
    for root in range(12):
        for profile, mode in ((_KS_MAJOR, "MAJ"), (_KS_MINOR, "MIN")):
            rot = np.roll(profile, root)
            r   = np.corrcoef(pitch_classes, rot)[0, 1]
            if r > best_score:
                best_score, best_key, best_mode = r, root, mode
    return f"<KEY_{_NOTE_NAMES[best_key]}_{best_mode}>"


def _detect_chord_cic(pitches: list) -> str | None:
    if not pitches:
        return None
    pc = np.zeros(12)
    for p in pitches:
        pc[p % 12] += 1
    best_root, best_qual, best_score = 0, "MAJ", -1
    for root in range(12):
        for qual, template in _CHORD_TEMPLATES.items():
            rot   = np.roll(template, root)
            score = np.dot(pc / (pc.sum() or 1), rot)
            if score > best_score:
                best_score, best_root, best_qual = score, root, qual
    return f"<CHORD_{_NOTE_NAMES[best_root]}_{best_qual}>"


def _select_tracks_cic(pm: pretty_midi.PrettyMIDI):
    non_drum = [i for i in pm.instruments if not i.is_drum and len(i.notes) > 0]
    if len(non_drum) < 2:
        return None, None
    mel_cands, acc_cands = [], []
    for inst in non_drum:
        cls = pretty_midi.program_to_instrument_class(inst.program)
        avg = np.mean([n.pitch for n in inst.notes])
        if cls in MELODY_CLASSES:
            mel_cands.append((inst, avg))
        if cls in ACCOMP_CLASSES:
            acc_cands.append((inst, avg))
    if not mel_cands or not acc_cands:
        return None, None
    melody = max(mel_cands, key=lambda x: x[1])[0]
    acc_cands = [(i, p) for i, p in acc_cands if i != melody]
    if not acc_cands:
        return None, None
    return melody, min(acc_cands, key=lambda x: x[1])[0]


def _inst_to_token_cic(inst: pretty_midi.Instrument) -> str:
    cls = pretty_midi.program_to_instrument_class(inst.program)
    if cls == "Bass":   return "<INST_BASS>"
    if cls == "Guitar": return "<INST_GUITAR>"
    return "<INST_PIANO>"


def _quantize_vel(vel: int) -> str:
    for b in VELOCITY_BINS:
        if vel <= b:
            return f"<VEL_{b}>"
    return "<VEL_127>"


def _quantize_dur(ticks: int) -> str:
    DUR_MAP = {1:"<DUR_1>",2:"<DUR_2>",3:"<DUR_3>",4:"<DUR_4>",
               6:"<DUR_6>",8:"<DUR_8>",12:"<DUR_12>",16:"<DUR_16>"}
    if ticks <= 0:
        return "<DUR_1>"
    return DUR_MAP[min(DUR_MAP, key=lambda x: abs(x - ticks))]


def _quantize_tempo(bpm: float) -> str:
    for thresh, label in zip([70,90,110,130,150,170,190],
                              ["<TEMPO_60>","<TEMPO_80>","<TEMPO_100>","<TEMPO_120>",
                               "<TEMPO_140>","<TEMPO_160>","<TEMPO_180>","<TEMPO_200>"]):
        if bpm < thresh:
            return label
    return "<TEMPO_200>"


def tokenize_melody_cic(inst: pretty_midi.Instrument,
                        pm: pretty_midi.PrettyMIDI,
                        tempo_bpm: float,
                        key_token: str) -> list | None:
    """Encode melody to positional tokens (old scheme, is_encoder=True)."""
    spt       = 60.0 / (tempo_bpm * PPQ)
    total_bars = min(int(pm.get_end_time() / (spt * TICKS_PER_BAR)), MAX_BARS)
    if total_bars < MIN_BARS:
        return None

    notes_by_tick: dict = defaultdict(list)
    for note in inst.notes:
        tick  = int(round(note.start / spt))
        dur_t = max(1, int(round((note.end - note.start) / spt)))
        notes_by_tick[tick].append((note.pitch, dur_t, note.velocity))

    tokens = ["<SOS>", "<TIMESIG_4_4>", key_token, _quantize_tempo(tempo_bpm)]
    prev_beat = -1

    for bar_idx in range(total_bars):
        bar_start = bar_idx * TICKS_PER_BAR
        bar_notes = [p for off in range(TICKS_PER_BAR)
                     for p, _, _ in notes_by_tick.get(bar_start + off, [])]
        chord_tok   = _detect_chord_cic(bar_notes)
        bar_emitted = chord_emitted = False

        for pos in range(TICKS_PER_BAR):
            tick = bar_start + pos
            if tick not in notes_by_tick:
                continue
            if not bar_emitted:
                tokens.append(f"<BAR_{bar_idx+1}>")
                bar_emitted = True
            if not chord_emitted and chord_tok:
                tokens.append(chord_tok)
                chord_emitted = True
            beat_idx = pos // PPQ
            if beat_idx != prev_beat:
                tokens.append(f"<BEAT_{beat_idx+1}>")
                prev_beat = beat_idx
            tokens.append(f"<POS_{pos}>")
            for pitch, dur_t, vel in notes_by_tick[tick]:
                if MIN_PITCH <= pitch <= MAX_PITCH:
                    tokens.append(f"<PITCH_{pitch}>")
                    tokens.append(_quantize_dur(dur_t))
                    tokens.append(_quantize_vel(vel))

    tokens.append("<EOS>")
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────────────────────────────────────

def _top_p_sampling(logits: torch.Tensor, p: float, temperature: float) -> int:
    logits = logits / temperature
    probs  = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumul = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumul - sorted_probs > p] = 0.0
    sorted_probs /= sorted_probs.sum()
    return sorted_idx[torch.multinomial(sorted_probs, 1)].item()


def _repetition_penalty(logits: torch.Tensor, gen_ids: list, penalty: float):
    """Apply penalty to any token that has appeared too many times in the last window."""
    window = gen_ids[-64:]
    # Hard-penalise structural tokens that appear more than 3 times consecutively
    if len(gen_ids) >= 4 and len(set(gen_ids[-4:])) == 1:
        logits[gen_ids[-1]] = float("-inf")
    # Soft penalty for any token seen in the recent window
    from collections import Counter
    counts = Counter(window)
    for tid, cnt in counts.items():
        if cnt >= 3:
            factor = penalty ** (cnt - 2)
            logits[tid] = logits[tid] / factor if logits[tid] > 0 else logits[tid] * factor
    return logits


@torch.no_grad()
def generate_cic(model: MusicTransformerCIC,
                 enc_ids: torch.Tensor, enc_mask: torch.Tensor,
                 prompt: list, device: torch.device,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.9,
                 top_p: float = 0.92,
                 repetition_penalty: float = 1.3) -> list:
    model.eval()
    enc_ids  = enc_ids.unsqueeze(0).to(device)
    enc_mask = enc_mask.unsqueeze(0).to(device)
    memory   = model.encode(enc_ids, enc_mask)

    gen_ids = list(prompt)
    eos_id  = TOKEN2ID["<EOS>"]

    for _ in range(max_new_tokens):
        tgt = torch.tensor([gen_ids], dtype=torch.long, device=device)
        tmask = torch.ones(1, len(gen_ids), dtype=torch.bool, device=device)
        if tgt.size(1) > MusicTransformerCIC.MAX_SEQ:
            tgt   = tgt[:, -MusicTransformerCIC.MAX_SEQ:]
            tmask = tmask[:, -MusicTransformerCIC.MAX_SEQ:]

        logits = model.decode(tgt, memory, tmask, enc_mask)
        next_logits = logits[0, -1, :].clone()
        next_logits[TOKEN2ID["<PAD>"]] = float("-inf")
        next_logits[TOKEN2ID["<UNK>"]] = float("-inf")
        next_logits = _repetition_penalty(next_logits, gen_ids, repetition_penalty)

        next_id = _top_p_sampling(next_logits, p=top_p, temperature=temperature)
        gen_ids.append(next_id)
        if next_id == eos_id:
            break

    return gen_ids


# ─────────────────────────────────────────────────────────────────────────────
# 5. Positional tokens → MIDI decoder
# ─────────────────────────────────────────────────────────────────────────────

def tokens_to_midi_cic(token_ids: list,
                       tempo_bpm: float = 120.0,
                       instrument_program: int = 33) -> pretty_midi.PrettyMIDI:
    pm   = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=instrument_program)
    spt  = 60.0 / (tempo_bpm * PPQ)   # seconds per tick

    DUR_TICKS = {"<DUR_1>":1,"<DUR_2>":2,"<DUR_3>":3,"<DUR_4>":4,
                 "<DUR_6>":6,"<DUR_8>":8,"<DUR_12>":12,"<DUR_16>":16,
                 "<DUR_T2>":2,"<DUR_T4>":4}
    VEL_MAP   = {f"<VEL_{v}>": v for v in VELOCITY_BINS}
    SKIP      = ("<GENRE_","<MOOD_","<ENERGY_","<INST_","<TIMESIG_",
                 "<KEY_","<TEMPO_","<CHORD_","<BEAT_")

    current_bar = current_pos = 0
    current_pitch = current_dur = current_vel = None

    def flush(force: bool = False):
        nonlocal current_pitch, current_dur, current_vel
        if current_pitch is None:
            return
        # Use defaults for dur/vel when tokens are missing (model sometimes skips them)
        dur = current_dur if current_dur is not None else 4   # default: quarter note
        vel = current_vel if current_vel is not None else 64
        if force or (current_dur is not None and current_vel is not None):
            ts = (current_bar * TICKS_PER_BAR + current_pos) * spt
            te = ts + dur * spt
            if MIN_PITCH <= current_pitch <= MAX_PITCH and te > ts:
                inst.notes.append(pretty_midi.Note(
                    velocity=vel, pitch=current_pitch, start=ts, end=te))
        current_pitch = current_dur = current_vel = None

    for tid in token_ids:
        tok = ID2TOKEN.get(tid, "<UNK>")
        if tok in ("<SOS>","<EOS>","<PAD>","<UNK>","<SEP>","<MASK>"):
            if tok == "<EOS>":
                flush(force=True)
            continue
        if any(tok.startswith(p) for p in SKIP):
            continue
        if tok.startswith("<BAR_"):
            flush(force=True); current_bar = int(tok[5:-1]) - 1; current_pos = 0
        elif tok.startswith("<POS_"):
            flush(force=True); current_pos = int(tok[5:-1])
        elif tok.startswith("<PITCH_"):
            flush(force=True); current_pitch = int(tok[7:-1])
        elif tok == "<REST>":
            flush(force=True); current_pitch = None
        elif tok in DUR_TICKS:
            current_dur = DUR_TICKS[tok]
        elif tok in VEL_MAP:
            current_vel = VEL_MAP[tok]; flush()

    flush(force=True)
    pm.instruments.append(inst)
    return pm


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_midi",         required=True)
    parser.add_argument("--genre",              default="FUNK",
                        choices=["ROCK","POP","FUNK","JAZZ","LATIN","CLASSICAL","ELECTRONIC"])
    parser.add_argument("--mood",               default="HAPPY",
                        choices=["HAPPY","SAD","DARK","RELAXED","TENSE"])
    parser.add_argument("--instrument",         default="BASS",
                        choices=["BASS","PIANO","GUITAR"])
    parser.add_argument("--output",             default="output_cic.mid")
    parser.add_argument("--temperature",        type=float, default=0.9)
    parser.add_argument("--top_p",              type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--max_tokens",         type=int,   default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = MusicTransformerCIC().to(device)
    ckpt  = torch.load("checkpoints/best_model_cic.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"CIC model loaded — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # Tokenize input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(args.input_midi)

    melody_inst, _ = _select_tracks_cic(pm)
    if melody_inst is None:
        melody_inst = next((i for i in pm.instruments if not i.is_drum), None)
    if melody_inst is None:
        raise ValueError("No usable tracks in input MIDI")

    tempo_bpm = pm.estimate_tempo()
    if not (30 < tempo_bpm < 300):
        tempo_bpm = 120.0
    key_token = _detect_key_cic(pm)
    print(f"Key: {key_token}  Tempo: {tempo_bpm:.0f} BPM")

    enc_tokens = tokenize_melody_cic(melody_inst, pm, tempo_bpm, key_token)
    if enc_tokens is None:
        # Fallback: use any non-drum track
        for inst in pm.instruments:
            if not inst.is_drum and len(inst.notes) >= 4:
                enc_tokens = tokenize_melody_cic(inst, pm, tempo_bpm, key_token)
                if enc_tokens:
                    break
    if enc_tokens is None:
        raise ValueError("Melody too short to tokenize")

    enc_tokens = enc_tokens[:MusicTransformerCIC.MAX_SEQ]
    enc_ids    = torch.tensor([TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in enc_tokens],
                               dtype=torch.long)
    enc_mask   = torch.ones(len(enc_ids), dtype=torch.bool)

    pad_len  = MusicTransformerCIC.MAX_SEQ - len(enc_ids)
    enc_ids  = torch.cat([enc_ids,  torch.zeros(pad_len, dtype=torch.long)])
    enc_mask = torch.cat([enc_mask, torch.zeros(pad_len, dtype=torch.bool)])

    prompt = [
        TOKEN2ID["<SOS>"],
        TOKEN2ID[f"<GENRE_{args.genre}>"],
        TOKEN2ID[f"<MOOD_{args.mood}>"],
        TOKEN2ID["<ENERGY_MED>"],
        TOKEN2ID[f"<INST_{args.instrument}>"],
    ]

    print(f"Generating {args.genre}/{args.mood}/{args.instrument} …")
    gen_ids = generate_cic(model, enc_ids, enc_mask, prompt, device,
                           max_new_tokens=args.max_tokens,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty)
    print(f"  {len(gen_ids)} tokens generated")

    INST_PROGRAMS = {"BASS": 33, "PIANO": 0, "GUITAR": 25}
    out_pm = tokens_to_midi_cic(gen_ids, tempo_bpm=tempo_bpm,
                                 instrument_program=INST_PROGRAMS[args.instrument])

    n_notes = sum(len(i.notes) for i in out_pm.instruments)
    if n_notes == 0:
        print("  Warning: 0 notes generated")
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_pm.write(args.output)
        print(f"  {n_notes} notes → {args.output}")


if __name__ == "__main__":
    main()

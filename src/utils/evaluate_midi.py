"""
Objective evaluation metrics for AI-generated MIDI accompaniment.

Covers 4 musical dimensions (12 metrics total):
  Pitch   — pitch class histogram, pitch range, pitch entropy, avg pitch interval
  Rhythm  — note density, IOI entropy, beat alignment ratio
  Harmony — in-key rate, tonal distance (KL divergence vs reference)
  Structure — polyphony rate, velocity entropy, empty bar rate

Usage:
  # No reference (uses uniform PCH baseline for tonal distance)
  PYTHONPATH=src python src/utils/evaluate_midi.py --generated results/batch/

  # With LMD reference sample
  PYTHONPATH=src python src/utils/evaluate_midi.py \
    --generated results/batch/ \
    --reference data/raw/lmd_matched/ \
    --max_ref 100 \
    --output results/metrics.csv
"""
import argparse
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

# ─────────────────────────────────────────────────────────────────────────────
# Krumhansl-Schmuckler key profiles (12-element, sum-normalised later)
# ─────────────────────────────────────────────────────────────────────────────
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Pitch classes in each major key (for in-key rate)
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) from a raw count or probability array."""
    p = counts.astype(float)
    total = p.sum()
    if total == 0:
        return float("nan")
    p = p / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _all_notes(midi: pretty_midi.PrettyMIDI) -> list:
    """Return all notes across non-drum instruments."""
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Key detection (Krumhansl-Schmuckler)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_key(pch: np.ndarray):
    """
    Returns (root_int, 'major'|'minor') for the best-matching key.
    pch: 12-element pitch class histogram (raw counts or probs).
    """
    if pch.sum() == 0:
        return 0, "major"
    p = pch / pch.sum()
    best_r, best_root, best_mode = -999, 0, "major"
    for root in range(12):
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            rotated = np.roll(profile, root)
            r = np.corrcoef(p, rotated / rotated.sum())[0, 1]
            if r > best_r:
                best_r, best_root, best_mode = r, root, mode
    return best_root, best_mode


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def pitch_class_histogram(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    """12-element normalised pitch class histogram."""
    hist = np.zeros(12)
    for n in _all_notes(midi):
        hist[n.pitch % 12] += 1
    total = hist.sum()
    return hist / total if total > 0 else hist


def pitch_range(midi: pretty_midi.PrettyMIDI) -> float:
    """Span between highest and lowest pitch in semitones."""
    pitches = [n.pitch for n in _all_notes(midi)]
    if len(pitches) < 2:
        return float("nan")
    return float(max(pitches) - min(pitches))


def pitch_entropy(midi: pretty_midi.PrettyMIDI) -> float:
    """Shannon entropy of the pitch distribution (over 128 MIDI pitches)."""
    counts = np.zeros(128)
    for n in _all_notes(midi):
        counts[n.pitch] += 1
    return _entropy(counts)


def avg_pitch_interval(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Mean absolute semitone interval between consecutive note onsets
    (sorted by start time, per instrument).
    """
    intervals = []
    for inst in midi.instruments:
        if inst.is_drum or len(inst.notes) < 2:
            continue
        pitches = [n.pitch for n in sorted(inst.notes, key=lambda x: x.start)]
        intervals.extend(abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1))
    if not intervals:
        return float("nan")
    return float(np.mean(intervals))


def note_density(midi: pretty_midi.PrettyMIDI) -> float:
    """Notes per beat (uses midi.estimate_tempo for beat duration)."""
    notes = _all_notes(midi)
    if not notes:
        return 0.0
    duration = midi.get_end_time()
    if duration <= 0:
        return float("nan")
    try:
        beats = midi.get_beats()
        n_beats = max(len(beats), 1)
    except Exception:
        tempo, _ = midi.estimate_tempo()
        n_beats = max(duration / (60.0 / max(tempo, 1)), 1)
    return float(len(notes) / n_beats)


def ioi_entropy(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Entropy of the inter-onset interval (IOI) distribution.
    IOIs are quantised to 32nd-note bins (31.25 ms at 120 BPM).
    """
    onsets = sorted(n.start for n in _all_notes(midi))
    if len(onsets) < 2:
        return float("nan")
    iois = np.diff(onsets)
    iois = iois[iois > 0.001]          # ignore simultaneous onsets
    if len(iois) == 0:
        return float("nan")
    # Quantise to 31.25 ms bins (1/32 note at 120 BPM)
    bin_size = 0.03125
    bins = np.round(iois / bin_size).astype(int)
    bins = np.clip(bins, 1, 128)
    counts = np.bincount(bins, minlength=129)[1:]
    return _entropy(counts)


def beat_alignment_ratio(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Fraction of note onsets within ±15 ms of an 8th-note grid position.
    """
    notes = _all_notes(midi)
    if not notes:
        return float("nan")
    try:
        beats = midi.get_beats()
    except Exception:
        return float("nan")
    if len(beats) < 2:
        return float("nan")

    # Build 8th-note grid (midpoints between beats + beat positions)
    grid = []
    for i, b in enumerate(beats):
        grid.append(b)
        if i + 1 < len(beats):
            grid.append((b + beats[i+1]) / 2.0)
    grid = np.array(grid)

    tol = 0.015  # 15 ms tolerance
    onsets = np.array([n.start for n in notes])
    dists = np.min(np.abs(onsets[:, None] - grid[None, :]), axis=1)
    return float((dists <= tol).mean())


def in_key_rate(midi: pretty_midi.PrettyMIDI) -> float:
    """Fraction of notes whose pitch class belongs to the detected key."""
    pch = pitch_class_histogram(midi) * 1e6   # un-normalise for detection
    root, mode = _detect_key(pch)
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    key_pcs = set((root + s) % 12 for s in scale)
    notes = _all_notes(midi)
    if not notes:
        return float("nan")
    in_key = sum(1 for n in notes if n.pitch % 12 in key_pcs)
    return float(in_key / len(notes))


def tonal_distance(pch_gen: np.ndarray, pch_ref: np.ndarray) -> float:
    """
    Symmetric KL divergence between two pitch class histograms.
    Both arrays are smoothed with a small epsilon to avoid log(0).
    """
    eps = 1e-8
    p = pch_gen + eps;  p /= p.sum()
    q = pch_ref + eps;  q /= q.sum()
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return (kl_pq + kl_qp) / 2.0


def polyphony_rate(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Average number of simultaneously active notes per active frame.
    Sampled at 100 Hz (10 ms resolution) using get_piano_roll().
    """
    notes = _all_notes(midi)
    if not notes:
        return float("nan")
    try:
        roll = midi.get_piano_roll(fs=100)   # shape (128, T)
        active = roll > 0
        active_cols = active.sum(axis=0)
        nonzero = active_cols[active_cols > 0]
        if len(nonzero) == 0:
            return float("nan")
        return float(nonzero.mean())
    except Exception:
        return float("nan")


def velocity_entropy(midi: pretty_midi.PrettyMIDI) -> float:
    """Shannon entropy of the velocity distribution (8-bin quantisation)."""
    notes = _all_notes(midi)
    if not notes:
        return float("nan")
    bins = np.array([n.velocity for n in notes])
    # 8 bins matching the tokenizer: 0-15, 16-31, ..., 112-127
    counts, _ = np.histogram(bins, bins=[0,16,32,48,64,80,96,112,128])
    return _entropy(counts)


def empty_bar_rate(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Fraction of 4-beat bars containing no note events.
    """
    notes = _all_notes(midi)
    if not notes:
        return 1.0
    try:
        beats = midi.get_beats()
    except Exception:
        return float("nan")
    if len(beats) < 4:
        return float("nan")

    onsets = set()
    # Build bar index for each onset
    beat_times = np.array(beats)
    note_starts = np.array([n.start for n in notes])

    n_bars = len(beats) // 4
    if n_bars == 0:
        return float("nan")

    empty = 0
    for bar in range(n_bars):
        bar_start = beat_times[bar * 4]
        bar_end   = beat_times[min(bar * 4 + 4, len(beat_times) - 1)]
        if not np.any((note_starts >= bar_start) & (note_starts < bar_end)):
            empty += 1
    return float(empty / n_bars)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_file(path: str, ref_pch: np.ndarray | None = None) -> dict:
    """
    Compute all 12 metrics for one MIDI file.
    ref_pch: reference pitch class histogram (12,); uses uniform if None.
    """
    result = {"file": Path(path).name}
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as e:
        warnings.warn(f"Cannot load {path}: {e}")
        nan = float("nan")
        for k in ("pitch_range","pitch_entropy","avg_pitch_interval",
                  "note_density","ioi_entropy","beat_alignment_ratio",
                  "in_key_rate","tonal_distance","polyphony_rate",
                  "velocity_entropy","empty_bar_rate"):
            result[k] = nan
        return result

    pch = pitch_class_histogram(midi)

    # Reference PCH: use provided or fall back to uniform
    if ref_pch is None or ref_pch.sum() == 0:
        ref = np.ones(12) / 12.0
    else:
        ref = ref_pch.copy()

    result["pitch_range"]          = pitch_range(midi)
    result["pitch_entropy"]        = pitch_entropy(midi)
    result["avg_pitch_interval"]   = avg_pitch_interval(midi)
    result["note_density"]         = note_density(midi)
    result["ioi_entropy"]          = ioi_entropy(midi)
    result["beat_alignment_ratio"] = beat_alignment_ratio(midi)
    result["in_key_rate"]          = in_key_rate(midi)
    result["tonal_distance"]       = tonal_distance(pch, ref)
    result["polyphony_rate"]       = polyphony_rate(midi)
    result["velocity_entropy"]     = velocity_entropy(midi)
    result["empty_bar_rate"]       = empty_bar_rate(midi)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reference PCH from LMD
# ─────────────────────────────────────────────────────────────────────────────

def build_reference_pch(ref_dir: str, max_files: int = 100) -> np.ndarray:
    """
    Sample up to max_files MIDIs from ref_dir, accumulate their PCHs,
    and return a normalised mean histogram.
    """
    midi_files = list(Path(ref_dir).rglob("*.mid"))
    if len(midi_files) > max_files:
        random.seed(42)
        midi_files = random.sample(midi_files, max_files)

    acc = np.zeros(12)
    loaded = 0
    for p in midi_files:
        try:
            midi = pretty_midi.PrettyMIDI(str(p))
            pch  = pitch_class_histogram(midi)
            if pch.sum() > 0:
                acc += pch
                loaded += 1
        except Exception:
            continue

    print(f"Reference PCH built from {loaded}/{len(midi_files)} LMD files.")
    return acc / acc.sum() if acc.sum() > 0 else np.ones(12) / 12.0


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation & display
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "pitch_range", "pitch_entropy", "avg_pitch_interval",
    "note_density", "ioi_entropy", "beat_alignment_ratio",
    "in_key_rate", "tonal_distance",
    "polyphony_rate", "velocity_entropy", "empty_bar_rate",
]

METRIC_LABELS = {
    "pitch_range":          "Pitch Range (semitones)",
    "pitch_entropy":        "Pitch Entropy (nats)",
    "avg_pitch_interval":   "Avg Pitch Interval (semitones)",
    "note_density":         "Note Density (notes/beat)",
    "ioi_entropy":          "IOI Entropy (nats)",
    "beat_alignment_ratio": "Beat Alignment Ratio",
    "in_key_rate":          "In-Key Rate",
    "tonal_distance":       "Tonal Distance (sym-KL)",
    "polyphony_rate":       "Polyphony Rate",
    "velocity_entropy":     "Velocity Entropy (nats)",
    "empty_bar_rate":       "Empty Bar Rate",
}

DIMENSIONS = {
    "Pitch":     ["pitch_range","pitch_entropy","avg_pitch_interval"],
    "Rhythm":    ["note_density","ioi_entropy","beat_alignment_ratio"],
    "Harmony":   ["in_key_rate","tonal_distance"],
    "Structure": ["polyphony_rate","velocity_entropy","empty_bar_rate"],
}


def print_report(df: pd.DataFrame, ref_pch: np.ndarray | None = None) -> None:
    valid = df.dropna(subset=METRIC_COLS, how="all")
    n = len(valid)
    print(f"\n{'='*68}")
    print(f"  MUSIC GENERATION EVALUATION REPORT  ({n} files)")
    print(f"{'='*68}")

    for dim, metrics in DIMENSIONS.items():
        print(f"\n── {dim} ─────────────────────────────────────────────")
        for m in metrics:
            col = valid[m].dropna()
            if col.empty:
                print(f"  {METRIC_LABELS[m]:<38}  N/A")
                continue
            mean, std = col.mean(), col.std()
            mn,   mx  = col.min(), col.max()
            print(f"  {METRIC_LABELS[m]:<38}  {mean:6.3f} ± {std:.3f}  "
                  f"[{mn:.3f} – {mx:.3f}]")

    # Per-file table (compact)
    print(f"\n── Per-file summary ─────────────────────────────────────────")
    header = f"{'File':<35} {'Range':>6} {'InKey':>6} {'Density':>8} {'EmptyBr':>8} {'Poly':>6}"
    print(header)
    print("-" * len(header))
    for _, row in valid.iterrows():
        def fmt(v): return f"{v:6.3f}" if not (isinstance(v, float) and math.isnan(v)) else "   NaN"
        print(f"{str(row['file'])[:34]:<35} "
              f"{fmt(row['pitch_range'])} "
              f"{fmt(row['in_key_rate'])} "
              f"{fmt(row['note_density']):>8} "
              f"{fmt(row['empty_bar_rate']):>8} "
              f"{fmt(row['polyphony_rate'])}")

    print(f"\n{'='*68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate objective metrics for AI-generated MIDI files."
    )
    parser.add_argument("--generated", required=True,
                        help="Directory containing generated .mid files")
    parser.add_argument("--reference", default=None,
                        help="Directory of reference MIDIs (LMD). "
                             "If omitted, uses uniform PCH for tonal distance.")
    parser.add_argument("--max_ref", type=int, default=100,
                        help="Max reference MIDI files to sample (default 100)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save per-file results as CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Reference PCH ─────────────────────────────────────────────────────
    ref_pch = None
    if args.reference:
        print(f"Building reference PCH from {args.reference} …")
        ref_pch = build_reference_pch(args.reference, max_files=args.max_ref)

    # ── Find generated files ───────────────────────────────────────────────
    gen_dir   = Path(args.generated)
    mid_files = sorted(gen_dir.rglob("*.mid"))
    if not mid_files:
        print(f"No .mid files found in {gen_dir}")
        return

    print(f"Evaluating {len(mid_files)} generated MIDI files …")

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = []
    for p in mid_files:
        r = evaluate_file(str(p), ref_pch=ref_pch)
        results.append(r)

    df = pd.DataFrame(results)

    # ── Report ────────────────────────────────────────────────────────────
    print_report(df, ref_pch=ref_pch)

    # ── Save CSV ──────────────────────────────────────────────────────────
    if args.output:
        df.to_csv(args.output, index=False, float_format="%.4f")
        print(f"Per-file results saved to {args.output}")


if __name__ == "__main__":
    main()

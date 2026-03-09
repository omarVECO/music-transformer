import csv, json
from pathlib import Path
from collections import defaultdict

TOKEN2ID = json.load(open("data/tokens/vocabulary.json"))["token2id"]
PAD_ID   = TOKEN2ID["<PAD>"]
MIN_WIN_LEN = 32
MAX_SEQ_LEN = 1536

def find_bar_boundaries(token_ids):
    bar_ids = {TOKEN2ID[f"<BAR_{i}>"] for i in range(1, 33)}
    return [i for i, tid in enumerate(token_ids) if tid in bar_ids]

rows = list(csv.DictReader(open("data/tokens/index.csv")))

skip_reasons = defaultdict(int)
short_lens   = []

for row in rows:
    try:
        data    = json.load(open(row["token_path"]))
        enc_ids = [TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in data["encoder_tokens"]]
        dec_ids = [TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in data["decoder_tokens"]]
    except Exception as e:
        skip_reasons["json_error"] += 1
        continue

    enc_bars = find_bar_boundaries(enc_ids)
    dec_bars = find_bar_boundaries(dec_ids)

    if len(enc_bars) < 2:
        skip_reasons["enc_no_bars"] += 1
        short_lens.append(len(enc_ids))
        continue

    # Simular primera ventana
    enc_start = enc_bars[0]
    enc_win   = enc_ids[enc_start:enc_start + MAX_SEQ_LEN]
    dec_start = dec_bars[0] if dec_bars else 0
    dec_win   = dec_ids[dec_start:dec_start + MAX_SEQ_LEN]

    if len(enc_win) < MIN_WIN_LEN:
        skip_reasons["enc_win_too_short"] += 1
        short_lens.append(len(enc_win))
    elif len(dec_win) < MIN_WIN_LEN:
        skip_reasons["dec_win_too_short"] += 1
        short_lens.append(len(dec_win))
    else:
        skip_reasons["should_be_ok"] += 1

import numpy as np
print("=== RAZONES DE SKIP ===")
for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason:<25} {count}")

if short_lens:
    print(f"\n=== LONGITUDES DE CASOS CORTOS ===")
    arr = np.array(short_lens)
    print(f"  min={arr.min()}  max={arr.max()}  avg={arr.mean():.0f}  median={np.median(arr):.0f}")
    print(f"  < 10 tokens: {(arr < 10).sum()}")
    print(f"  < 32 tokens: {(arr < 32).sum()}")
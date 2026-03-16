import torch, sys
sys.path.insert(0, 'src')
from model.config import ModelConfig
from model.transformer import MusicTransformer
from data.midi_tokenizer import TOKEN2ID, ID2TOKEN, detect_key, select_tracks, notes_to_token_sequence, inst_to_token
import pretty_midi
from collections import Counter

device = torch.device("cuda")
config = ModelConfig()
model  = MusicTransformer(config).to(device)
ckpt   = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

midi_path = "data/raw/lmd_matched/T/T/T/TRTTTXG128F93274CA/94327c13c6adfce6158ee27cbfb4ecee.mid"
pm = pretty_midi.PrettyMIDI(midi_path)
melody_inst, _ = select_tracks(pm)
tempo_bpm = pm.estimate_tempo()
key_token = detect_key(pm)

enc_tokens = notes_to_token_sequence(
    melody_inst, pm, tempo_bpm, key_token,
    "FUNK", "DARK", 0.5, inst_to_token(melody_inst), is_encoder=True
)
enc_tokens = enc_tokens[:config.max_seq_len]
enc_ids  = torch.tensor([TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in enc_tokens], dtype=torch.long)
enc_mask = torch.ones(len(enc_ids), dtype=torch.bool)
pad_len  = config.max_seq_len - len(enc_ids)
enc_ids  = torch.cat([enc_ids,  torch.zeros(pad_len, dtype=torch.long)])
enc_mask = torch.cat([enc_mask, torch.zeros(pad_len, dtype=torch.bool)])

from model.inference import generate

prompt = [TOKEN2ID["<SOS>"], TOKEN2ID["<GENRE_FUNK>"],
          TOKEN2ID["<MOOD_DARK>"], TOKEN2ID["<ENERGY_MED>"], TOKEN2ID["<INST_BASS>"]]

print("=" * 55)
for temp in [0.6, 0.8, 1.0, 1.2]:
    torch.manual_seed(42)
    gen_ids = generate(model, enc_ids, enc_mask, prompt, config, device,
                       max_new_tokens=400, temperature=temp, top_p=0.9)
    gen_tokens = [ID2TOKEN.get(i, "<UNK>") for i in gen_ids]

    pitches = [t for t in gen_tokens if t.startswith("<PITCH_")]
    unique  = set(pitches)
    notes   = [int(t[7:-1]) for t in pitches]
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    pitch_counter = Counter(pitches).most_common(6)

    print(f"\ntemperature={temp}")
    print(f"  Pitches únicos: {len(unique)} / {len(pitches)} notas")
    print(f"  Rango: MIDI {min(notes)} ({note_names[min(notes)%12]}) — MIDI {max(notes)} ({note_names[max(notes)%12]})")
    print(f"  Top 6: {[(t[7:-1], c) for t, c in pitch_counter]}")
    if len(notes) > 1:
        import numpy as np
        intervals = [abs(notes[i+1]-notes[i]) for i in range(len(notes)-1)]
        print(f"  Intervalo promedio: {np.mean(intervals):.1f} semitonos")
print("=" * 55)
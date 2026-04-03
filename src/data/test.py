import torch, sys, json
sys.path.insert(0, 'src')
from model.config import ModelConfig
from model.transformer import MusicTransformer
from model.inference import generate
from data.midi_tokenizer import (
    TOKEN2ID, ID2TOKEN, detect_key, select_tracks,
    notes_to_token_sequence, inst_to_token
)
import pretty_midi

device = torch.device("cuda")
config = ModelConfig()
model  = MusicTransformer(config).to(device)
ckpt   = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

midi_path = "data/raw/lmd_matched/T/T/T/TRTTTXG128F93274CA/94327c13c6adfce6158ee27cbfb4ecee.mid"
pm        = pretty_midi.PrettyMIDI(midi_path)
melody, _ = select_tracks(pm)
tempo_bpm = pm.estimate_tempo()
key_token = detect_key(pm)

enc_tokens = notes_to_token_sequence(
    melody, pm, tempo_bpm, key_token,
    "FUNK", "DARK", 0.5, inst_to_token(melody), is_encoder=True
)
enc_tokens = enc_tokens[:config.max_seq_len]
enc_ids  = torch.tensor([TOKEN2ID.get(t, TOKEN2ID["<UNK>"]) for t in enc_tokens], dtype=torch.long)
enc_mask = torch.ones(len(enc_ids), dtype=torch.bool)
pad_len  = config.max_seq_len - len(enc_ids)
enc_ids  = torch.cat([enc_ids,  torch.zeros(pad_len, dtype=torch.long)])
enc_mask = torch.cat([enc_mask, torch.zeros(pad_len, dtype=torch.bool)])

torch.manual_seed(42)
prompt  = [TOKEN2ID["<SOS>"], TOKEN2ID["<GENRE_FUNK>"],
           TOKEN2ID["<MOOD_DARK>"], TOKEN2ID["<ENERGY_MED>"], TOKEN2ID["<INST_BASS>"]]
gen_ids = generate(model, enc_ids, enc_mask, prompt, config, device,
                   max_new_tokens=300, temperature=0.8, top_p=0.92)
gen_tokens = [ID2TOKEN.get(i, "<UNK>") for i in gen_ids]

# Mostrar los primeros 120 tokens del decoder con contexto
print("=== TOKENS DECODER (primeros 120) ===")
for i, t in enumerate(gen_tokens[:120]):
    print(f"  {i:>3}: {t}")

# Mostrar también encoder — primeros 2 compases
print("\n=== TOKENS ENCODER (primeros 2 compases) ===")
bar_count = 0
for i, t in enumerate(enc_tokens):
    if t.startswith("<BAR_"):
        bar_count += 1
        if bar_count > 2:
            break
    print(f"  {i:>3}: {t}")

# Guardar tokens para análisis
with open("/tmp/debug_tokens.json", "w") as f:
    json.dump({"enc": enc_tokens[:200], "dec": gen_tokens}, f, indent=2)
print("\nTokens guardados en /tmp/debug_tokens.json")
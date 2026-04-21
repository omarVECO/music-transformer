"""
Batch generation: runs generate_score.py on a diverse set of LMD input MIDIs,
producing one MIDI + MusicXML output per input.

Usage (from project root):
    PYTHONPATH=src python scripts/batch_generate.py [--max N]
"""
import argparse
import subprocess
import sys
from pathlib import Path

INPUTS = [
    # (midi_path, genre, mood, instrument)
    ("data/raw/lmd_matched/G/D/F/TRGDFBB12903C9963A/468be2f5dd31a1ba444b8018d8e8c7ad.mid",  "FUNK",       "HAPPY",   "BASS"),
    ("data/raw/lmd_matched/O/Q/E/TROQERS128F4263D1C/0adda16c56e2d7698aa86fba6c3df744.mid",  "FUNK",       "SAD",     "BASS"),
    ("data/raw/lmd_matched/E/S/N/TRESNUQ128F42ACF91/039839682a31386adda152110defb0d8.mid",  "FUNK",       "DARK",    "PIANO"),
    ("data/raw/lmd_matched/O/X/M/TROXMVV128F92E47A0/db426aa15937ba4f477233ac2f72a13e.mid",  "FUNK",       "RELAXED", "GUITAR"),
    ("data/raw/lmd_matched/I/Z/R/TRIZRTY128F4226906/18e844944d410b378fffc9185bdfd9b5.mid",  "FUNK",       "TENSE",   "BASS"),
    ("data/raw/lmd_matched/W/M/H/TRWMHMP128EF34293F/d8392424ea57a0fe6f65447680924d37.mid",  "JAZZ",       "HAPPY",   "PIANO"),
    ("data/raw/lmd_matched/E/G/G/TREGGML12903CC0740/33e3ed0356f8356f0a89a4dce5be88ad.mid",  "JAZZ",       "SAD",     "PIANO"),
    ("data/raw/lmd_matched/N/W/N/TRNWNJI128E0783C42/d8a7cc1e64c27695c42a781bb86adfc5.mid",  "JAZZ",       "DARK",    "BASS"),
    ("data/raw/lmd_matched/Q/U/I/TRQUIXQ128F92E89D2/25ca38ed61bbf131e983f7da28b70801.mid",  "JAZZ",       "RELAXED", "PIANO"),
    ("data/raw/lmd_matched/O/G/N/TROGNMP128F148C4A0/6de22e60c6cc161b49b7a37394a30b98.mid",  "JAZZ",       "TENSE",   "BASS"),
    ("data/raw/lmd_matched/E/M/T/TREMTWP128F4250CFB/f478f7a62006a5b935cc05a103f11a44.mid",  "ROCK",       "HAPPY",   "GUITAR"),
    ("data/raw/lmd_matched/X/T/T/TRXTTPP128F92F5595/9a732292bc21a930dfcb4a417cde29d5.mid",  "ROCK",       "SAD",     "BASS"),
    ("data/raw/lmd_matched/L/T/L/TRLTLWT12903CFB613/558ae7abee2e050e6f823af8d5ee27c2.mid",  "ROCK",       "DARK",    "GUITAR"),
    ("data/raw/lmd_matched/A/S/B/TRASBQL128F92F0777/8bbaebad0e103bdb71148806fb200ef8.mid",  "ROCK",       "RELAXED", "PIANO"),
    ("data/raw/lmd_matched/N/G/B/TRNGBSC128F424508F/4a8719e3345312b5683d01a89865363c.mid",  "ROCK",       "TENSE",   "GUITAR"),
    ("data/raw/lmd_matched/V/F/F/TRVFFKS128F4289A7D/2ac8563e4883aed6d75900a1b979ac31.mid",  "POP",        "HAPPY",   "PIANO"),
    ("data/raw/lmd_matched/S/U/T/TRSUTUD128F42333F4/ed50c73d40d44c78004f6a1275c5a87d.mid",  "POP",        "SAD",     "PIANO"),
    ("data/raw/lmd_matched/C/G/S/TRCGSSV128EF35EFFB/6fbb50c61e19f8892d007a36c1b88311.mid",  "POP",        "DARK",    "BASS"),
    ("data/raw/lmd_matched/H/V/E/TRHVEET12903CCE9EF/cfa7b0c800addeae37362e53c21d6ea4.mid",  "POP",        "RELAXED", "GUITAR"),
    ("data/raw/lmd_matched/P/P/L/TRPPLIU128F4260C6E/b9314b56aa9167360a1004d9d2f56089.mid",  "POP",        "TENSE",   "BASS"),
    ("data/raw/lmd_matched/I/N/U/TRINUBW128F4249FAA/b66d828db4826ab01fa2f7f7fb8084de.mid",  "CLASSICAL",  "HAPPY",   "PIANO"),
    ("data/raw/lmd_matched/O/I/E/TROIEWC128F425AD35/dbe2c3f88f818862dbf1d921c23e4f75.mid",  "CLASSICAL",  "SAD",     "PIANO"),
    ("data/raw/lmd_matched/O/O/S/TROOSVK128F931B9A6/2f18ff3e10cec11156d3bd7b9696c892.mid",  "CLASSICAL",  "DARK",    "BASS"),
    ("data/raw/lmd_matched/U/D/F/TRUDFHM128F426D114/de861589c5b1358f72a0fd6e4694e010.mid",  "CLASSICAL",  "RELAXED", "PIANO"),
    ("data/raw/lmd_matched/W/Z/Q/TRWZQBG128F149EF5F/42cdc8fc90d9b8311e1b9df1199ead37.mid",  "CLASSICAL",  "TENSE",   "PIANO"),
    ("data/raw/lmd_matched/S/N/X/TRSNXNI128F147C9D8/7b4bb90c2c573beda0f74c14ab570e20.mid",  "ELECTRONIC", "HAPPY",   "BASS"),
    ("data/raw/lmd_matched/Q/X/V/TRQXVIW128F42A9B31/b0fb2fb5318a9d1a0b70467833e655cd.mid",  "ELECTRONIC", "SAD",     "PIANO"),
    ("data/raw/lmd_matched/E/F/I/TREFIDV128F930A23A/42edf9d577bd899860deb6c9b85046e4.mid",  "ELECTRONIC", "DARK",    "BASS"),
    ("data/raw/lmd_matched/P/J/Q/TRPJQIR12903CBE9BC/709a519120605e551efa8d903b35f19b.mid",  "ELECTRONIC", "RELAXED", "GUITAR"),
    ("data/raw/lmd_matched/E/I/S/TREISJY128F428C9A7/8f71b771d4fbeb8c0e355593589f6413.mid",  "ELECTRONIC", "TENSE",   "BASS"),
    ("data/raw/lmd_matched/V/E/I/TRVEICJ12903CD8D92/2329f83374b97bf935af6818ea5e364e.mid",  "LATIN",      "HAPPY",   "GUITAR"),
    ("data/raw/lmd_matched/V/A/N/TRVANTV128F4264525/6d19edbf4b355c8104be20ffd5f38abc.mid",  "LATIN",      "SAD",     "PIANO"),
    ("data/raw/lmd_matched/V/P/C/TRVPCCY128F92CED12/95be6ec14abbb6120592e65216445f8e.mid",  "LATIN",      "DARK",    "BASS"),
    ("data/raw/lmd_matched/V/R/O/TRVROEK128F149A6DE/0c6c54e865e00bd4790bbb6634fc10f4.mid",  "LATIN",      "RELAXED", "GUITAR"),
    ("data/raw/lmd_matched/B/B/E/TRBBERA128F93474E1/c657f3a599245af9a2ff6537ee5945f1.mid",  "LATIN",      "TENSE",   "BASS"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=len(INPUTS),
                        help="Max number of files to generate")
    args = parser.parse_args()

    out_dir = Path("results/batch")
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = INPUTS[: args.max]
    ok, failed = 0, []

    for i, (midi_path, genre, mood, instrument) in enumerate(inputs, 1):
        stem = f"{i:02d}_{genre}_{mood}_{instrument}".lower()
        out_xml  = str(out_dir / f"{stem}.xml")
        out_midi = str(out_dir / f"{stem}.mid")

        print(f"\n[{i}/{len(inputs)}] {genre}/{mood}/{instrument} — {Path(midi_path).name}")

        cmd = [
            sys.executable, "src/utils/generate_score.py",
            "--input_midi",         midi_path,
            "--genre",              genre,
            "--mood",               mood,
            "--instrument",         instrument,
            "--output",             out_xml,
            "--output_midi",        out_midi,
            "--temperature",        "0.9",
            "--top_p",              "0.92",
            "--top_k",              "50",
            "--repetition_penalty", "1.3",
            "--max_tokens",         "1024",
        ]

        env = {"PYTHONPATH": "src"}
        import os
        full_env = {**os.environ, **env}

        result = subprocess.run(cmd, env=full_env, capture_output=False)
        if result.returncode == 0:
            ok += 1
            print(f"  ✓ saved {out_midi}")
        else:
            failed.append(stem)
            print(f"  ✗ FAILED (return code {result.returncode})")

    print(f"\n{'='*50}")
    print(f"Done: {ok}/{len(inputs)} succeeded")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()

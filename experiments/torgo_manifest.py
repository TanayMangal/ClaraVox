import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

FIELDS = ["id", "speaker", "etiology", "category", "subcategory", "path", "duration", "text"]


def usable(text):
    t = str(text).strip()
    if not t or "[" in t or "]" in t or t.lower() == "xxx" or t.endswith(".jpg"):
        return False
    return bool(re.search(r"[A-Za-z]", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="torgo_pairs.csv")
    ap.add_argument("--out", default="torgo_work")
    ap.add_argument("--train-split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    df = df[df["is_dysarthric"] == True].reset_index(drop=True)  # noqa: E712  (same as script 3)
    speakers = df["speaker"].unique().tolist()
    np.random.seed(args.seed)
    np.random.shuffle(speakers)
    k = int(len(speakers) * args.train_split)
    train_spk, eval_spk = speakers[:k], speakers[k:]
    print("train speakers:", train_spk, "| eval speakers:", eval_spk)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "dev": []}
    skipped = 0
    for _, r in df.iterrows():
        if not usable(r["transcript"]):
            skipped += 1
            continue
        try:
            dur = sf.info(r["audio_path"]).duration
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if not (0.2 <= dur <= 30):
            skipped += 1
            continue
        split = "train" if r["speaker"] in train_spk else "dev"
        rows[split].append({
            "id": f"{r['speaker']}_{r['session']}_{Path(r['audio_path']).stem}",
            "speaker": r["speaker"], "etiology": "TORGO", "category": "TORGO", "subcategory": "",
            "path": str(Path(r["audio_path"]).resolve()), "duration": f"{dur:.3f}",
            "text": str(r["transcript"]).strip(),
        })
    for name, rr in rows.items():
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rr)
        print(f"{name}: {len(rr)} utterances, {sum(float(x['duration']) for x in rr)/3600:.2f} h")
    print(f"skipped {skipped} rows (instructions, missing files, too long/short)")


if __name__ == "__main__":
    main()

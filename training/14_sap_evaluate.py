import argparse
import csv
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch

SR = 16000
_norm_re = re.compile(r"[^a-z0-9' ]+")


def norm_text(s):
    s = (s or "").lower().replace("’", "'")
    return " ".join(_norm_re.sub(" ", s).split())


def load_audio(path):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a[: SR * 30]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--work", default="sap_work")
    ap.add_argument("--csv", default=None, help="default: <work>/test.csv")
    ap.add_argument("--per-etiology", type=int, default=0, help="sample at most N utterances per etiology (0 = all)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import jiwer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if args.threads:
        torch.set_num_threads(args.threads)
    work = Path(args.work)
    rows = list(csv.DictReader(open(args.csv or work / "test.csv", newline="", encoding="utf-8-sig")))
    if not rows:
        sys.exit("No rows to evaluate (is test.csv empty? With one audio tar there may be few test speakers).")
    if args.per_etiology:
        rng, g = random.Random(0), defaultdict(list)
        for r in rows:
            g[r["etiology"]].append(r)
        rows = [x for v in g.values() for x in rng.sample(v, min(args.per_etiology, len(v)))]
    print(f"Evaluating on {len(rows)} utterances from {len({r['speaker'] for r in rows})} speakers")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = work / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.models:
        proc = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name).to(device).eval()
        gen_kw = {"max_new_tokens": 128, "num_beams": args.beams}
        if getattr(model.generation_config, "is_multilingual", True):
            gen_kw.update(language="en", task="transcribe")
        preds, t0 = [], time.time()
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            audios = []
            for r in chunk:
                p = Path(r["path"])
                audios.append(load_audio(str(p if p.is_absolute() else work / p)))
            feats = proc.feature_extractor(audios, sampling_rate=SR, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                ids = model.generate(input_features=feats, **gen_kw)
            preds.extend(proc.batch_decode(ids, skip_special_tokens=True))
            if (i // args.batch) % 20 == 0:
                print(f"  {name}: {i + len(chunk)}/{len(rows)}")
        secs = (time.time() - t0) / len(rows)

        tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[-60:]
        with open(out_dir / f"{tag}_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "speaker", "etiology", "reference", "prediction"])
            for r, pr in zip(rows, preds):
                w.writerow([r["id"], r["speaker"], r["etiology"], r["text"], pr.strip()])

        groups = defaultdict(lambda: ([], []))
        for r, pr in zip(rows, preds):
            ref, hyp = norm_text(r["text"]), norm_text(pr)
            if not ref:
                continue
            for key in (r["etiology"], "ALL"):
                groups[key][0].append(ref)
                groups[key][1].append(hyp)
        for key, (refs, hyps) in groups.items():
            summary.append({"model": name, "etiology": key, "utterances": len(refs),
                            "wer": round(100 * jiwer.wer(refs, hyps), 2),
                            "sec_per_utt": round(secs, 2)})
        del model

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "etiology", "utterances", "wer", "sec_per_utt"])
        w.writeheader()
        w.writerows(summary)

    ets = sorted({s["etiology"] for s in summary}, key=lambda e: (e == "ALL", e))
    print("\nWER % (lower is better)")
    print(f"{'model':<45}" + "".join(f"{e:>9}" for e in ets))
    for name in args.models:
        cells = {s["etiology"]: s["wer"] for s in summary if s["model"] == name}
        print(f"{name[-45:]:<45}" + "".join(f"{cells.get(e, float('nan')):>9.1f}" for e in ets))
    print(f"\nSaved {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()

import argparse, csv, random, re, sys
from collections import defaultdict
import torch, jiwer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PREFIX = "Fix grammar: "
norm = lambda s: " ".join(re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower().replace("’", "'")).split())

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--manifest", default="sap_work/manifest_all.csv")
ap.add_argument("--gec", default="models/SpeechCorrection/gec-model")
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="sap_work/eval/gec_overcorrection.csv")
ap.add_argument("--guard", action="store_true", help="apply app.py's safe_correction() guard")
a = ap.parse_args()

cat = {r["id"]: r["category"] for r in csv.DictReader(open(a.manifest, encoding="utf-8-sig"))}
rows = list(csv.DictReader(open(a.pred, encoding="utf-8-sig")))
random.Random(0).shuffle(rows)
rows = [r for r in rows if norm(r["reference"])][: a.n]
tok = AutoTokenizer.from_pretrained(a.gec)
model = AutoModelForSeq2SeqLM.from_pretrained(a.gec).to(a.device).eval()

def gec(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([PREFIX + t for t in texts[i:i+bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(a.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_length=128, num_beams=5, early_stopping=True)
        for t in tok.batch_decode(ids, skip_special_tokens=True):
            t = t.strip()
            if t.lower().startswith(PREFIX.lower()):
                t = t[len(PREFIX):].strip()
            out.append(re.sub(r"\s+([.,!?;:])", r"\1", t))
        print(f"  {min(i+bs, len(texts))}/{len(texts)}", file=sys.stderr, flush=True)
    return out

refs = [r["reference"] for r in rows]
preds = [r["prediction"] for r in rows]
gec_ref = gec(refs)
gec_pred = gec(preds)
if a.guard:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "space"))
    from app import safe_correction
    cap = lambda raw: raw.strip()[:1].upper() + raw.strip()[1:]
    fix = lambda raw, c: cap(raw) if len(raw.split()) <= 1 or not safe_correction(raw, c) else c
    gec_ref = [fix(r, c) for r, c in zip(refs, gec_ref)]
    gec_pred = [fix(r, c) for r, c in zip(preds, gec_pred)]
    print("guard: on", file=sys.stderr)

groups = defaultdict(lambda: defaultdict(list))
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "etiology", "category", "reference", "gec(reference)", "prediction", "gec(prediction)"])
    for r, gr, gp in zip(rows, gec_ref, gec_pred):
        c = cat.get(r["id"], "?")
        w.writerow([r["id"], r["etiology"], c, r["reference"], gr, r["prediction"], gp])
        for key in (f"cat:{c}", f"et:{r['etiology']}", "ALL"):
            g = groups[key]
            g["ref"].append(norm(r["reference"])); g["gref"].append(norm(gr) or "x")
            g["pred"].append(norm(r["prediction"]) or "x"); g["gpred"].append(norm(gp) or "x")
            g["changed"].append(norm(gr) != norm(r["reference"]))

print(f"\n{'group':<38}{'n':>5}{'ref changed':>13}{'WER gec(ref)':>14}{'WER raw':>10}{'WER corrected':>15}")
for key in sorted(groups, key=lambda k: (k != "ALL", k)):
    g = groups[key]
    n = len(g["ref"])
    print(f"{key:<38}{n:>5}{100*sum(g['changed'])/n:>12.1f}%{100*jiwer.wer(g['ref'], g['gref']):>13.1f}%"
          f"{100*jiwer.wer(g['ref'], g['pred']):>9.1f}%{100*jiwer.wer(g['ref'], g['gpred']):>14.1f}%")
print(f"\nper-utterance details: {a.out}")

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PREFIX = "Fix grammar: "


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower().replace("’", "'")).split())


def read(p, limit=0):
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def generate(model, tok, texts, device, bs=32):
    out = []
    model.eval()
    for i in range(0, len(texts), bs):
        enc = tok([PREFIX + t for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_length=128, num_beams=4, early_stopping=True)
        for t in tok.batch_decode(ids, skip_special_tokens=True):
            t = t.strip()
            if t.lower().startswith(PREFIX.lower()):
                t = t[len(PREFIX):].strip()
            out.append(re.sub(r"\s+([.,!?;:])", r"\1", t))
    return out


def test(model_dir, test_csv, device, limit):
    import jiwer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    rows = read(test_csv)
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    rows = [r for v in by.values() for r in v[:limit]]
    preds = generate(model, tok, [r["source"] for r in rows], device)
    g = defaultdict(lambda: {"n": 0, "exact": 0, "same": 0, "refs": [], "hyps": []})
    for r, p in zip(rows, preds):
        for key in (r["kind"], "ALL"):
            d = g[key]
            d["n"] += 1
            d["exact"] += norm(p) == norm(r["target"])
            d["same"] += norm(p) == norm(r["source"])
            d["refs"].append(norm(r["target"]) or "x")
            d["hyps"].append(norm(p) or "x")
    print(f"\n{model_dir}")
    print(f"{'kind':<12}{'n':>6}{'exact match':>13}{'unchanged':>11}{'WER':>8}")
    for k in sorted(g, key=lambda k: (k == "ALL", k)):
        d = g[k]
        print(f"{k:<12}{d['n']:>6}{100*d['exact']/d['n']:>12.1f}%{100*d['same']/d['n']:>10.1f}%"
              f"{100*jiwer.wer(d['refs'], d['hyps']):>7.1f}%")
    print("(for 'identity' rows, unchanged = left a correct sentence alone; higher is better)")
    for r, p in list(zip(rows, preds))[:10]:
        print(f"  [{r['kind']}] {r['source']!r} -> {p[:120]!r}   (target {r['target']!r})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="sap_work")
    ap.add_argument("--base-model", default="google/flan-t5-base")
    ap.add_argument("--out", default=None, help="default: <work>/runs/gec-v2")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--model", default=None, help="model to test with --test-only")
    ap.add_argument("--test-per-kind", type=int, default=500)
    args = ap.parse_args()
    work = Path(args.work)
    data = work / "gec_v2"
    out = Path(args.out or work / "runs" / "gec-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_only:
        test(args.model or out / "final", data / "test.csv", device, args.test_per_kind)
        return

    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    def prep(batch):
        enc = tok([PREFIX + s for s in batch["source"]], max_length=128, truncation=True)
        enc["labels"] = tok(text_target=batch["target"], max_length=128, truncation=True)["input_ids"]
        return enc

    train = Dataset.from_list(read(data / "train.csv")).map(prep, batched=True, remove_columns=["source", "target", "kind"])
    dev = Dataset.from_list(read(data / "dev.csv", 1000)).map(prep, batched=True, remove_columns=["source", "target", "kind"])

    def metrics(pred):
        p, l = pred.predictions, pred.label_ids
        p = p[0] if isinstance(p, tuple) else p
        p = np.where(p != -100, p, tok.pad_token_id)
        l = np.where(l != -100, l, tok.pad_token_id)
        ps = [norm(x) for x in tok.batch_decode(p, skip_special_tokens=True)]
        ls = [norm(x) for x in tok.batch_decode(l, skip_special_tokens=True)]
        return {"exact": round(float(np.mean([a == b for a, b in zip(ps, ls)])), 4)}

    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch, per_device_eval_batch_size=32,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_steps=200,
        eval_strategy="steps", eval_steps=1000, save_strategy="steps", save_steps=1000, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="exact", greater_is_better=True,
        predict_with_generate=True, generation_max_length=128, logging_steps=100, report_to="none",
        bf16=bf16, dataloader_num_workers=0, seed=42)
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train, eval_dataset=dev,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
                             compute_metrics=metrics, processing_class=tok)
    trainer.train()
    final = out / "final"
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"saved {final} (best dev exact match {trainer.state.best_metric})")
    test(final, data / "test.csv", device, args.test_per_kind)


if __name__ == "__main__":
    main()

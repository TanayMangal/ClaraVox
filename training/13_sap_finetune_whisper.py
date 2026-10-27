import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
LAYER_MODULES = "q_proj|k_proj|v_proj|out_proj|fc1|fc2"


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio[: SR * 30]


class SpeechCSV(Dataset):


    def __init__(self, rows, work_dir, processor):
        self.rows, self.work, self.proc = rows, Path(work_dir), processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = Path(r["path"])
        p = p if p.is_absolute() else self.work / p
        audio = load_audio(str(p))
        feats = self.proc.feature_extractor(audio, sampling_rate=SR).input_features[0]
        labels = self.proc.tokenizer(r["text"]).input_ids[:448]
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        lab = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)


        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def cpu_supports_bf16():
    try:
        return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except Exception:  # noqa: BLE001
        return False


def prepare_model(model, method, train_from, lora_r, lora_alpha):
    n_enc = model.config.encoder_layers
    train_from = max(0, min(train_from, n_enc))
    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model
    if method == "partial":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.model.decoder.parameters():
            p.requires_grad = True
        for i in range(train_from, n_enc):
            for p in model.model.encoder.layers[i].parameters():
                p.requires_grad = True
        for p in model.model.encoder.layer_norm.parameters():
            p.requires_grad = True
        return model

    from peft import LoraConfig, get_peft_model
    enc_layers = "|".join(str(i) for i in range(train_from, n_enc)) or "NONE"
    target = (rf".*encoder\.layers\.({enc_layers})\..*\.({LAYER_MODULES})"
              rf"|.*decoder\.layers\.\d+\..*\.({LAYER_MODULES})")
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
                     target_modules=target)
    return get_peft_model(model, cfg)


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tot


_norm_re = re.compile(r"[^a-z0-9' ]+")


def norm_text(s):
    s = s.lower().replace("’", "'")
    s = _norm_re.sub(" ", s)
    return " ".join(s.split())


def benchmark(model, ds, collator, args, n_train_rows):
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collator,
                    num_workers=args.workers)
    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"
    model.to(dev)
    if args.grad_ckpt and args.method != "lora":
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    use_bf16 = (not gpu) and (args.bf16 == "on" or (args.bf16 == "auto" and cpu_supports_bf16()))
    amp = dict(device_type="cuda", dtype=torch.float16, enabled=True) if gpu else \
        dict(device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16)
    scaler = torch.amp.GradScaler("cuda", enabled=gpu)
    model.train()
    times, it = [], iter(dl)
    print(f"\nBenchmark on {'GPU (fp16)' if gpu else 'CPU'}: {args.benchmark} steps, batch {args.batch}, "
          f"bf16={use_bf16}, threads={torch.get_num_threads()}")
    for step in range(args.benchmark + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: v.to(dev) for k, v in batch.items()}
        t0 = time.time()
        with torch.autocast(**amp):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if gpu:
            torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 0:
            times.append(dt)
        print(f"  step {step}: {dt:.1f}s  loss {loss.item():.3f}")
    sec_per_ex = float(np.median(times)) / args.batch
    model.eval()
    gen_model = model
    t0 = time.time()
    b = collator([ds[i] for i in range(min(4, len(ds)))])
    with torch.no_grad(), torch.autocast(**amp):
        gen_model.generate(input_features=b["input_features"].to(dev), max_new_tokens=64)
    gen_s = (time.time() - t0) / min(4, len(ds))
    total_ex = n_train_rows * args.epochs if args.max_steps <= 0 else args.max_steps * args.batch * args.grad_accum
    hrs = total_ex * sec_per_ex / 3600
    print(f"\n  {sec_per_ex:.2f} s per training example  ->  {total_ex} examples "
          f"= about {hrs:.1f} hours ({hrs/24:.1f} days) of training")
    print(f"  {gen_s:.2f} s to transcribe one utterance (also roughly the eval cost per dev sample)")
    if gpu:
        print(f"  peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("  Rough guide: if that is more than your cluster allows, lower --train-hours in step 12,")
    print("  raise --train-encoder-from, or try --bf16 on (if the CPU supports it).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="sap_work")
    ap.add_argument("--train-csv", default=None, help="default: <work>/train.csv")
    ap.add_argument("--dev-csv", default=None, help="default: <work>/dev_small.csv")
    ap.add_argument("--base-model", default="openai/whisper-small",
                    help="folder or hub id; your ./whisper-torgo is a good start")
    ap.add_argument("--out", default=None, help="default: <work>/runs/whisper-sap")
    ap.add_argument("--method", choices=["lora", "partial", "full"], default="lora")
    ap.add_argument("--train-encoder-from", type=int, default=6,
                    help="encoder layers below this index are frozen (whisper-small has 12)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="default: 5e-4 lora, 1e-5 otherwise")
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--spec-augment", action="store_true", help="mask bits of the spectrogram (robustness)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = all cores)")
    ap.add_argument("--workers", type=int, default=4, help="data loading processes (use 0 on Windows)")
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="off",
                    help="bfloat16 on CPU: faster on CPUs with AVX512-BF16/AMX, try it in --benchmark")
    ap.add_argument("--eval-every", type=int, default=200, help="check dev WER every N steps")
    ap.add_argument("--save-every", type=int, default=200,
                    help="checkpoint every N steps (must be a multiple of --eval-every)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing: less GPU memory, ~30%% slower (for 8 GB cards)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps, print time estimate, exit")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: use only N training rows")
    args = ap.parse_args()

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)
    import jiwer

    work = Path(args.work)
    train_csv = Path(args.train_csv or work / "train.csv")
    dev_csv = Path(args.dev_csv or work / "dev_small.csv")
    out = Path(args.out or work / "runs" / "whisper-sap")
    lr = args.lr or (5e-4 if args.method == "lora" else 1e-5)
    if args.threads:
        torch.set_num_threads(args.threads)
    gpu = torch.cuda.is_available()

    processor = WhisperProcessor.from_pretrained(args.base_model)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_feature_prob = 0.05

    model = prepare_model(model, args.method, args.train_encoder_from, args.lora_r, args.lora_alpha)
    tr, tot = count_params(model)
    print(f"Method {args.method}: training {tr/1e6:.1f}M of {tot/1e6:.1f}M parameters")

    train_rows = read_rows(train_csv)
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    dev_rows = read_rows(dev_csv) if dev_csv.exists() else []
    print(f"Train: {len(train_rows)} utterances | Dev: {len(dev_rows)} | device: {'GPU' if gpu else 'CPU'}")
    train_ds = SpeechCSV(train_rows, work, processor)
    dev_ds = SpeechCSV(dev_rows, work, processor) if dev_rows else None
    collator = Collator(processor, model.config.decoder_start_token_id)

    if args.benchmark:
        benchmark(model, train_ds, collator, argparse.Namespace(**{**vars(args), "lr": lr}), len(train_rows))
        return

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        p = [norm_text(x) for x in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        r = [norm_text(x) for x in processor.batch_decode(label_ids, skip_special_tokens=True)]
        keep = [i for i, x in enumerate(r) if x]
        return {"wer": round(jiwer.wer([r[i] for i in keep], [p[i] for i in keep]), 4)}

    use_bf16 = (not gpu) and (args.bf16 == "on" or (args.bf16 == "auto" and cpu_supports_bf16()))
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        eval_strategy="steps" if dev_ds else "no",
        eval_steps=args.eval_every,
        save_strategy="steps",
        save_steps=args.save_every,
        save_total_limit=3,
        load_best_model_at_end=bool(dev_ds),
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=25,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.workers,
        fp16=gpu,
        bf16=use_bf16,
        gradient_checkpointing=args.grad_ckpt and args.method != "lora",
        use_cpu=not gpu,
        seed=42,
    )
    if dev_ds and args.save_every % args.eval_every:
        sys.exit("--save-every must be a multiple of --eval-every (needed to keep the best checkpoint)")

    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                             data_collator=collator, compute_metrics=compute_metrics,
                             processing_class=processor.feature_extractor)
    has_ckpt = any(out.glob("checkpoint-*")) if out.exists() else False
    trainer.train(resume_from_checkpoint=True if (args.resume and has_ckpt) else None)

    final = out / "final"
    m = trainer.model
    if args.method == "lora":
        m = m.merge_and_unload()
    m.config.use_cache = True
    m.save_pretrained(final)
    processor.save_pretrained(final)
    info = {"base_model": args.base_model, "method": args.method,
            "train_encoder_from": args.train_encoder_from, "train_utterances": len(train_rows),
            "epochs": args.epochs, "lr": lr,
            "best_dev_wer": trainer.state.best_metric,
            "data": "Speech Accessibility Project (University of Illinois), used under a Data Transfer "
                    "and Use Agreement. No audio or transcripts are included in this folder."}
    (final / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved merged model to {final}  (best dev WER: {trainer.state.best_metric})")


if __name__ == "__main__":
    main()

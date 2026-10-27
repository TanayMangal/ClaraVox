import re, sys, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
P = "Fix grammar: "
models = sys.argv[1:] or ["sap_work/runs/gec-v2/final"]
tests = ["want go park", "i want go park", "she like eat apple every day", "my name john i want drink water",
         "can help me find phone", "me hungry want pizza", "him go school yesterday", "dog is run fast",
         "I want to go to the park.", "What are my reminders?", "Is the pharmacy open?", "Answer the call."]
dev = "cuda" if torch.cuda.is_available() else "cpu"
for m in models:
    tok = AutoTokenizer.from_pretrained(m); mod = AutoModelForSeq2SeqLM.from_pretrained(m).to(dev).eval()
    print("==", m)
    for t in tests:
        enc = tok(P + t, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = tok.decode(mod.generate(**enc, max_length=64, num_beams=5)[0], skip_special_tokens=True)
        print(f"  {t!r:45} -> {out!r}")

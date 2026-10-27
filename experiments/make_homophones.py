import json
import re
from collections import defaultdict
from pathlib import Path

import cmudict
from transformers import WhisperTokenizer

tok = WhisperTokenizer.from_pretrained("openai/whisper-base.en")
d = cmudict.dict()


def common(w):
    return re.fullmatch(r"[a-z]+", w) and (len(w) > 1 or w in ("a", "i")) and \
        len(tok(" " + w, add_special_tokens=False).input_ids) == 1


groups = defaultdict(set)
for w, prons in d.items():
    if not common(w):
        continue
    for p in prons:
        groups[tuple(re.sub(r"\d", "", ph) for ph in p)].add(w)

homo = defaultdict(set)
for words in groups.values():
    if len(words) > 1:
        for w in words:
            homo[w] |= words - {w}

out = {w: sorted(v) for w, v in sorted(homo.items())}
path = Path(__file__).resolve().parent.parent / "backend" / "homophones.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"{len(out)} words with homophones -> {path}")
for w in ["know", "no", "there", "to", "just", "write", "eight", "four", "hear", "buy"]:
    print(f"  {w}: {out.get(w)}")

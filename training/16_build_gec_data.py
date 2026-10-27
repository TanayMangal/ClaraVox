import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARTICLES = {"a", "an", "the"}
BE = {"is", "are", "am", "was", "were", "'s", "'re", "'m"}
AUX = {"do", "does", "did", "will", "can", "would", "could", "should", "have", "has"}
SUBJ = {"i", "you", "he", "she", "it", "we", "they"}
PREPS = {"to", "for", "of", "in", "on", "at", "with"}
PRONOUN_PERSON = {
    "i": 1, "me": 1, "my": 1, "mine": 1, "myself": 1, "we": 1, "us": 1, "our": 1, "ours": 1,
    "you": 2, "your": 2, "yours": 2, "yourself": 2,
}
ANSWER_STARTS = {"yes", "no", "yeah", "okay", "ok", "right", "sure", "oh", "well", "uh", "um"}

PAST = {"went": "go", "saw": "see", "ate": "eat", "had": "have", "made": "make", "took": "take",
        "came": "come", "got": "get", "gave": "give", "did": "do", "said": "say", "told": "tell",
        "found": "find", "bought": "buy", "thought": "think", "felt": "feel", "left": "leave",
        "ran": "run", "wrote": "write", "read": "read", "played": "play", "watched": "watch",
        "liked": "like", "loved": "love", "wanted": "want", "lived": "live", "worked": "work"}


def words(s):
    return s.split()


def norm(w):
    return re.sub(r"[^a-z']", "", w.lower())


def drop_set(ws, vocab, p, rng, keep_first=False):
    out = [w for i, w in enumerate(ws) if not (norm(w) in vocab and rng.random() < p and not (keep_first and i == 0))]
    return out or ws


def corrupt(sentence, rng):
    ws = words(sentence)
    grammar_ops = [
        lambda w: drop_set(w, ARTICLES, 0.8, rng),
        lambda w: drop_set(w, BE, 0.8, rng),
        lambda w: drop_set(w, AUX, 0.6, rng, keep_first=True),
        lambda w: w[1:] if len(w) > 2 and norm(w[0]) in SUBJ else w,
        lambda w: drop_set(w, {"to"}, 0.7, rng),
        lambda w: drop_set(w, PREPS, 0.4, rng),
        lambda w: [re.sub(r"(?<=[a-z])s$", "", x) if norm(x).endswith("s") and len(norm(x)) > 3
                   and i > 0 and rng.random() < 0.5 else x for i, x in enumerate(w)],
        lambda w: [PAST.get(norm(x), x) if rng.random() < 0.8 else x for x in w],
    ]
    surface_ops = [
        lambda w: [x.lower() for x in w],
        lambda w: [re.sub(r"[.,!?]+$", "", x) for x in w],
    ]
    key = lambda w: [norm(x) for x in w if norm(x)]
    rng.shuffle(grammar_ops)
    out, made = ws, 0
    for op in grammar_ops:
        new = op(out)
        if key(new) != key(out):
            out, made = new, made + 1
            if made >= rng.randint(1, 3):
                break
    if not made:
        return None
    for op in surface_ops:
        if rng.random() < 0.4:
            out = op(out)
    return " ".join(out)


def person_signature(s):
    return Counter(PRONOUN_PERSON[w] for w in map(norm, words(s)) if w in PRONOUN_PERSON)


def safe_recast(src, tgt):
    tw = [norm(w) for w in words(tgt)]
    for w in {norm(w) for w in words(src)} - ARTICLES - BE - AUX - PREPS - {""}:
        if not any(t == w or (len(w) >= 3 and t.startswith(w[:3])) for t in tw):
            return False
    if tw and tw[0] in ANSWER_STARTS:
        return False
    ps, pt = person_signature(src), person_signature(tgt)
    if pt.get(2, 0) > ps.get(2, 0):
        return False
    if ps and ps != pt:
        return False
    return len(tw) <= len(words(src)) + 4


def bucket(text):
    return int(hashlib.sha1(text.lower().encode()).hexdigest()[:8], 16) / 0x100000000


def speaker_bucket(speaker):
    return int(hashlib.sha1(speaker.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="sap_work")
    ap.add_argument("--talkbank", default="gec_pairs.csv", help="output of 6_parse_talkbank.py (optional)")
    ap.add_argument("--identity-share", type=float, default=0.30)
    ap.add_argument("--max-sentences", type=int, default=60000)
    ap.add_argument("--jfleg", type=int, default=3000, help="max JFLEG pairs to add (0 = none)")
    ap.add_argument("--test-frac", type=float, default=0.16, help="must match step 12 (dev+test speakers)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    work = Path(args.work)
    out_dir = work / "gec_v2"
    out_dir.mkdir(parents=True, exist_ok=True)


    sents = set()
    with open(work / "utterances.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["drop_reason"] or speaker_bucket(r["speaker"]) < args.test_frac:
                continue
            t = r["text"].strip()
            if r["category"] == "Non-spontaneous Speech Prompts":
                continue
            if 3 <= len(words(t)) <= 30:
                sents.add(t)
    sents = sorted(sents)
    rng.shuffle(sents)
    sents = sents[: args.max_sentences]
    pairs = []
    for s in sents:
        if rng.random() < args.identity_share:
            pairs.append((s, s, "identity"))
        else:
            c = corrupt(s, rng)
            pairs.append((c, s, "synthetic") if c else (s, s, "identity"))
    print(f"SAP sentences (train speakers): {len(sents)}")


    kept = dropped = 0
    if Path(args.talkbank).exists():
        with open(args.talkbank, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("pair_type") != "echo_recast":
                    continue
                if safe_recast(r["source"], r["target"]):
                    pairs.append((r["source"], r["target"], "talkbank"))
                    kept += 1
                else:
                    dropped += 1
        print(f"TalkBank recasts kept {kept}, dropped {dropped} (unsafe: answers, pronoun changes, lost words)")


    if args.jfleg:
        try:
            from datasets import load_dataset
            jf = load_dataset("jhu-clsp/jfleg")
            n = 0
            for split in jf:
                for item in jf[split]:
                    for c in item["corrections"][:1]:
                        if c.strip() and c.strip() != item["sentence"].strip() and n < args.jfleg:
                            pairs.append((item["sentence"].strip(), c.strip(), "jfleg"))
                            n += 1
            print(f"JFLEG pairs: {n}")
        except Exception as e:  # noqa: BLE001
            print(f"[note] JFLEG not added ({e})")


    splits = {"train": [], "dev": [], "test": []}
    for src, tgt, kind in pairs:
        b = bucket(tgt)
        key = "test" if b < 0.05 else "dev" if b < 0.08 else "train"
        splits[key].append({"source": src, "target": tgt, "kind": kind})
    for k, rows in splits.items():
        rng.shuffle(rows)
        with open(out_dir / f"{k}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source", "target", "kind"])
            w.writeheader()
            w.writerows(rows)
        print(f"{k}: {len(rows)} pairs  {dict(Counter(r['kind'] for r in rows))}")
    print("\nexamples:")
    for r in splits["train"][:8]:
        print(f"  [{r['kind']}] {r['source']!r} -> {r['target']!r}")


if __name__ == "__main__":
    main()

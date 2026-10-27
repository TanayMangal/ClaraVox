import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = ["id", "speaker", "etiology", "category", "subcategory", "path", "duration", "text"]


CATEGORY_RANK = {
    "Spontaneous Speech Prompts": 0,
    "Novel Sentences": 1,
    "Non-spontaneous Speech Prompts": 2,
    "Digital Assistant Commands": 3,
}


def speaker_bucket(speaker: str) -> float:
    h = hashlib.sha1(speaker.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x100000000


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def hours(rows):
    return sum(float(r["duration"]) for r in rows) / 3600


def parse_weights(s):
    out = {}
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


def pick_round_robin(rows, budget_h, max_spk_h, wake_cap, rng):
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    for spk_rows in by_spk.values():
        rng.shuffle(spk_rows)
        spk_rows.sort(key=lambda r: CATEGORY_RANK.get(r["category"], 4))
    queues = {s: list(v) for s, v in by_spk.items()}
    used_s = defaultdict(float)
    wake_used = 0.0
    picked, total = [], 0.0
    budget_s, cap_s = budget_h * 3600, max_spk_h * 3600
    speakers = list(queues)
    rng.shuffle(speakers)
    while total < budget_s and speakers:
        nxt = []
        for s in speakers:
            q = queues[s]
            while q:
                r = q.pop(0)
                d = float(r["duration"])
                is_wake = r["subcategory"] == "Wake Phrases"
                if is_wake and wake_used + d > wake_cap * budget_s:
                    continue
                if used_s[s] + d > cap_s:
                    q.clear()
                    break
                picked.append(r)
                used_s[s] += d
                total += d
                wake_used += d if is_wake else 0
                break
            if q:
                nxt.append(s)
            if total >= budget_s:
                break
        speakers = nxt
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="sap_work")
    ap.add_argument("--train-hours", type=float, default=60.0)
    ap.add_argument("--weights", default="DS=3,CP=2,ALS=1.5,Stroke=1.5,PD=0.5",
                    help="relative weight per etiology (short names)")
    ap.add_argument("--max-hours-per-speaker", type=float, default=0.6)
    ap.add_argument("--wake-cap", type=float, default=0.02, help="max share of wake phrases")
    ap.add_argument("--dev-frac", type=float, default=0.08)
    ap.add_argument("--test-frac", type=float, default=0.08)
    ap.add_argument("--dev-small", type=int, default=300)
    ap.add_argument("--mix-torgo", default=None, help="torgo_pairs.csv from script 2 (optional)")
    ap.add_argument("--torgo-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    work = Path(args.work)
    rows = read_csv(work / "manifest_all.csv")
    if not rows:
        sys.exit("manifest_all.csv is empty; run step 11 first")
    rng = random.Random(args.seed)

    split = {"train": [], "dev": [], "test": []}
    for r in rows:
        b = speaker_bucket(r["speaker"])
        key = "test" if b < args.test_frac else "dev" if b < args.test_frac + args.dev_frac else "train"
        split[key].append(r)


    weights = parse_weights(args.weights)
    by_e = defaultdict(list)
    for r in split["train"]:
        by_e[r["etiology"]].append(r)
    avail = {e: hours(v) for e, v in by_e.items()}
    share = {e: weights.get(e, 1.0) * avail[e] for e in by_e}
    budget_total = args.train_hours * (1 - (args.torgo_share if args.mix_torgo else 0))
    budget = {e: 0.0 for e in by_e}
    remaining, open_e = budget_total, set(by_e)
    while remaining > 1e-6 and open_e:
        z = sum(share[e] for e in open_e)
        grant = {e: remaining * share[e] / z for e in open_e}
        remaining = 0.0
        for e in list(open_e):
            take = min(grant[e], avail[e] - budget[e])
            budget[e] += take
            remaining += grant[e] - take
            if avail[e] - budget[e] < 1e-6:
                open_e.discard(e)

    train = []
    for e, er in by_e.items():
        train.extend(pick_round_robin(er, budget[e], args.max_hours_per_speaker, args.wake_cap, rng))

    if args.mix_torgo:
        import soundfile as sf
        t_rows, missing = [], 0
        for r in read_csv(args.mix_torgo):
            if str(r.get("is_dysarthric", "True")).lower() not in ("true", "1"):
                continue
            try:
                dur = sf.info(r["audio_path"]).duration
            except Exception:  # noqa: BLE001  (file not on this machine)
                missing += 1
                continue
            if not (0.3 <= dur <= 30):
                continue
            if "[" in r["transcript"] or "]" in r["transcript"]:
                continue
            t_rows.append({"id": "torgo_" + r.get("speaker", "") + "_" + r.get("session", "") + "_"
                           + Path(r["audio_path"]).stem,
                           "speaker": "TORGO_" + r.get("speaker", ""), "etiology": "TORGO",
                           "category": "TORGO", "subcategory": "", "path": r["audio_path"],
                           "duration": f"{dur:.3f}", "text": r["transcript"]})
        if missing:
            print(f"[note] {missing} TORGO files listed in {args.mix_torgo} were not found on this machine")
        rng.shuffle(t_rows)
        n_torgo = int(len(train) * args.torgo_share / max(1e-6, 1 - args.torgo_share))
        train.extend(t_rows[:n_torgo])
        print(f"Added {min(n_torgo, len(t_rows))} TORGO utterances")

    rng.shuffle(train)

    dev_by_e = defaultdict(list)
    for r in split["dev"]:
        dev_by_e[r["etiology"]].append(r)
    dev_small = []
    per = max(1, args.dev_small // max(1, len(dev_by_e)))
    for e, v in dev_by_e.items():
        dev_small.extend(rng.sample(v, min(per, len(v))))

    write_csv(work / "train.csv", train)
    write_csv(work / "dev.csv", split["dev"])
    write_csv(work / "dev_small.csv", dev_small)
    write_csv(work / "test.csv", split["test"])

    lines = [f"{'split':<10}{'etiology':<9}{'speakers':>9}{'utts':>9}{'hours':>8}"]
    for name, rr in [("train", train), ("dev", split["dev"]), ("dev_small", dev_small), ("test", split["test"])]:
        g = defaultdict(list)
        for r in rr:
            g[r["etiology"]].append(r)
        for e, v in sorted(g.items()):
            lines.append(f"{name:<10}{e:<9}{len({r['speaker'] for r in v}):>9}{len(v):>9}{hours(v):>8.1f}")
        lines.append(f"{name:<10}{'ALL':<9}{len({r['speaker'] for r in rr}):>9}{len(rr):>9}{hours(rr):>8.1f}")
    summary = "\n".join(lines)
    (work / "split_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    small = [e for e, v in dev_by_e.items() if len({r['speaker'] for r in v}) < 2]
    if small:
        print(f"\n[note] dev has fewer than 2 speakers for {small}; numbers for those will be noisy "
              f"until you add another audio tar.")


if __name__ == "__main__":
    main()

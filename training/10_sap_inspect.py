import argparse
import csv
import io
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sap_common import (iter_speaker_json, records_from_speaker_json,  # noqa: E402
                        short_etiology)

UTT_FIELDS = ["id", "filename", "speaker", "etiology", "etiology_short", "category",
              "subcategory", "text_raw", "text", "drop_reason"]
BYTES_PER_HOUR = 16000 * 2 * 3600


def load_json_source(src: Path):
    if src.is_dir():
        for _, data in iter_speaker_json(src):
            yield data
        return
    with tarfile.open(src, "r:*") as tar:
        for m in tar:
            if m.isfile() and m.name.lower().endswith(".json"):
                f = tar.extractfile(m)
                if f is None:
                    continue
                try:
                    data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] bad json {m.name}: {e}")
                    continue
                if isinstance(data, dict) and isinstance(data.get("Files"), list):
                    yield data


def index_audio_tar(path: Path, spk_etiology: dict):
    rows = []
    uuid = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
    with tarfile.open(path, "r:") as tar:
        for m in tar:
            if not m.isfile():
                continue
            hit = uuid.search(m.name)
            spk = hit.group(0) if hit else ""
            rows.append({
                "tar": path.name,
                "member": m.name,
                "speaker": spk,
                "etiology": spk_etiology.get(spk, ""),
                "bytes": m.size,
                "est_hours": round(m.size / BYTES_PER_HOUR, 3),
            })

    if rows and all(r["member"].lower().endswith(".wav") for r in rows):
        agg = defaultdict(lambda: {"bytes": 0, "n": 0})
        for r in rows:
            agg[r["speaker"]]["bytes"] += r["bytes"]
            agg[r["speaker"]]["n"] += 1
        rows = [{"tar": path.name, "member": f"{v['n']} wav files", "speaker": s,
                 "etiology": spk_etiology.get(s, ""), "bytes": v["bytes"],
                 "est_hours": round(v["bytes"] / BYTES_PER_HOUR, 3)} for s, v in agg.items()]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="folder of speaker JSONs, or the *_Json.tar")
    ap.add_argument("--audio-tar", nargs="*", default=[], help="audio tar(s) to index (optional)")
    ap.add_argument("--work", default="sap_work", help="output folder (git-ignored)")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    utts, spk_rows = [], []
    for data in load_json_source(Path(args.json)):
        recs = records_from_speaker_json(data)
        if not recs:
            continue
        utts.extend(recs)
        cats = Counter(r["category"] for r in recs if not r["drop_reason"])
        spk_rows.append({
            "speaker": recs[0]["speaker"],
            "etiology": recs[0]["etiology"],
            "etiology_short": recs[0]["etiology_short"],
            "n_files": len(recs),
            "n_usable": sum(1 for r in recs if not r["drop_reason"]),
            "n_commands": cats.get("Digital Assistant Commands", 0),
            "n_novel": cats.get("Novel Sentences", 0),
            "n_spontaneous": cats.get("Spontaneous Speech Prompts", 0),
            "n_nonspontaneous": cats.get("Non-spontaneous Speech Prompts", 0),
        })

    if not utts:
        sys.exit(f"No speaker JSONs found in {args.json}")

    with open(work / "utterances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=UTT_FIELDS)
        w.writeheader()
        w.writerows(utts)
    with open(work / "speakers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(spk_rows[0].keys()))
        w.writeheader()
        w.writerows(spk_rows)


    print(f"\nSpeakers: {len(spk_rows)}   Utterances: {len(utts)}")
    by_e = defaultdict(lambda: [0, 0, 0])
    for s in spk_rows:
        e = by_e[s["etiology_short"]]
        e[0] += 1
        e[1] += s["n_files"]
        e[2] += s["n_usable"]
    print(f"\n{'Etiology':<10}{'speakers':>10}{'files':>10}{'usable':>10}")
    for k, (n, f_, u) in sorted(by_e.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:<10}{n:>10}{f_:>10}{u:>10}")
    drops = Counter(r["drop_reason"] for r in utts if r["drop_reason"])
    print("\nDropped for training:", dict(drops) or "none")
    print("Categories:", dict(Counter(r["category"] for r in utts)))


    if args.audio_tar:
        spk_et = {s["speaker"]: s["etiology_short"] for s in spk_rows}
        rows = []
        for t in args.audio_tar:
            print(f"\nIndexing {t} (reads headers only, may take a minute)...")
            rows.extend(index_audio_tar(Path(t), spk_et))
        with open(work / "tar_index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["tar", "member", "speaker", "etiology", "bytes", "est_hours"])
            w.writeheader()
            w.writerows(rows)
        per = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        for r in rows:
            c = per[r["tar"]][r["etiology"] or "?"]
            c[0] += 1
            c[1] += r["est_hours"]
        for tname, d in per.items():
            tot = sum(v[1] for v in d.values())
            print(f"\n{tname}: ~{tot:.0f} h of audio (estimate from file sizes, assumes 16 kHz 16-bit wav)")
            for e, (n, h) in sorted(d.items(), key=lambda kv: -kv[1][1]):
                print(f"   {e:<8} {n:>4} speakers  ~{h:6.1f} h")
        unknown = sum(1 for r in rows if not r["speaker"])
        if unknown:
            print(f"\n[note] {unknown} members had no speaker ID in their name; see tar_index.csv")

    print(f"\nWrote {work/'utterances.csv'}, {work/'speakers.csv'}"
          + (f", {work/'tar_index.csv'}" if args.audio_tar else ""))


if __name__ == "__main__":
    main()

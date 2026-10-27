import argparse
import csv
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--work", default="sap_work")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    work, ids = Path(args.work), set(args.ids)

    for pid in ids:
        d = work / "audio" / pid
        if d.exists():
            n = sum(1 for _ in d.iterdir())
            print(f"{'would delete' if args.dry_run else 'deleting'} {d} ({n} files)")
            if not args.dry_run:
                shutil.rmtree(d)

    for p in work.rglob("*.csv"):
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        keep = [rows[0]] + [r for r in rows[1:] if not any(pid in cell for cell in r for pid in ids)]
        removed = len(rows) - len(keep)
        if removed:
            print(f"{'would remove' if args.dry_run else 'removed'} {removed} rows from {p}")
            if not args.dry_run:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(keep)

    print("\nAlso delete these IDs from: the original SpeechAccessibility_*.tar files (or re-download "
          "without them), the *_Train_Only_Json folder (<id>.json), and any other machine or Kaggle copy.")
    print("The withdrawal clause covers Data; the agreement defines trained models as 'Results'. "
          "If you're unsure whether a model needs retraining, ask speechaccessibility@beckman.illinois.edu.")


if __name__ == "__main__":
    main()

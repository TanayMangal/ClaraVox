import argparse
import csv
import io
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sap_common import speaker_from_filename, utt_id  # noqa: E402

TARGET_SR = 16000
MIN_SEC, MAX_SEC = 0.5, 30.0


def convert_one(wav_bytes: bytes, out_path: str):
    import numpy as np
    import soundfile as sf
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = np.clip(audio, -1.0, 1.0)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path + ".part"
        sf.write(tmp, audio, TARGET_SR, format="FLAC", subtype="PCM_16")
        os.replace(tmp, out_path)
        return out_path, len(audio) / TARGET_SR, ""
    except Exception as e:  # noqa: BLE001
        return out_path, 0.0, f"{type(e).__name__}: {e}"


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def iter_wavs(tar_path: Path, wanted):
    with tarfile.open(tar_path, "r:") as outer:
        for m in outer:
            if not m.isfile():
                continue
            name = m.name.lower()
            if name.endswith(".tar"):
                spk_guess = speaker_from_filename(Path(m.name).stem)
                if wanted is not None and spk_guess not in wanted:
                    continue
                fobj = outer.extractfile(m)
                with tarfile.open(fileobj=fobj, mode="r|*") as inner:
                    for im in inner:
                        if im.isfile() and im.name.lower().endswith(".wav"):
                            spk = speaker_from_filename(im.name)
                            if wanted is not None and spk not in wanted:
                                continue
                            yield spk, Path(im.name).name, inner.extractfile(im).read()
            elif name.endswith(".wav"):
                spk = speaker_from_filename(m.name)
                if wanted is not None and spk not in wanted:
                    continue
                yield spk, Path(m.name).name, outer.extractfile(m).read()


def extract(tar_paths, work: Path, wanted, workers: int):
    audio_root = work / "audio"
    index_path = work / "audio_index.csv"
    errors_path = work / "prepare_errors.log"
    done = set()
    if index_path.exists():
        done = {r["id"] for r in load_csv(index_path)}
        print(f"Resuming: {len(done)} files already converted")

    new_index = not index_path.exists()
    idx_f = open(index_path, "a", newline="", encoding="utf-8")
    idx = csv.writer(idx_f)
    if new_index:
        idx.writerow(["id", "speaker", "path", "duration"])
    err_f = open(errors_path, "a", encoding="utf-8")

    t0, n_ok, n_err = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def drain(block_until):
            nonlocal n_ok, n_err
            while len(pending) > block_until:
                finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in finished:
                    uid, spk = pending.pop(fut)
                    out_path, dur, err = fut.result()
                    if err:
                        n_err += 1
                        err_f.write(f"{uid}\t{err}\n")
                    else:
                        n_ok += 1
                        idx.writerow([uid, spk, os.path.relpath(out_path, work), f"{dur:.3f}"])
                if n_ok and n_ok % 2000 == 0:
                    idx_f.flush()
                    rate = n_ok / (time.time() - t0)
                    print(f"  converted {n_ok} files ({rate:.0f}/s), errors {n_err}")

        for tp in tar_paths:
            print(f"\nReading {tp} ...")
            for spk, fname, data in iter_wavs(Path(tp), wanted):
                uid = utt_id(fname)
                if uid in done:
                    continue
                out = str(audio_root / spk / f"{uid}.flac")
                pending[pool.submit(convert_one, data, out)] = (uid, spk)
                drain(workers * 8)
        drain(0)

    idx_f.close()
    err_f.close()
    print(f"\nConverted {n_ok} new files, {n_err} errors (see {errors_path.name}) "
          f"in {(time.time()-t0)/60:.1f} min")


def build_manifest(work: Path):
    utts = {r["id"]: r for r in load_csv(work / "utterances.csv")}
    rows, missing_text, dropped = [], 0, 0
    seen = set()
    for a in load_csv(work / "audio_index.csv"):
        if a["id"] in seen:
            continue
        seen.add(a["id"])
        u = utts.get(a["id"])
        if u is None:
            missing_text += 1
            continue
        dur = float(a["duration"])
        if u["drop_reason"] or not u["text"] or not (MIN_SEC <= dur <= MAX_SEC):
            dropped += 1
            continue
        rows.append({
            "id": a["id"], "speaker": u["speaker"], "etiology": u["etiology_short"],
            "category": u["category"], "subcategory": u["subcategory"],
            "path": a["path"], "duration": f"{dur:.3f}", "text": u["text"],
        })
    out = work / "manifest_all.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "speaker", "etiology", "category", "subcategory",
                                          "path", "duration", "text"])
        w.writeheader()
        w.writerows(rows)
    hours = sum(float(r["duration"]) for r in rows) / 3600
    by_e = {}
    for r in rows:
        e = by_e.setdefault(r["etiology"], [set(), 0.0])
        e[0].add(r["speaker"])
        e[1] += float(r["duration"]) / 3600
    print(f"\nmanifest_all.csv: {len(rows)} usable utterances, {hours:.1f} h "
          f"({dropped} dropped by filters, {missing_text} audio files had no transcript)")
    for e, (spk, h) in sorted(by_e.items(), key=lambda kv: -kv[1][1]):
        print(f"   {e:<8} {len(spk):>4} speakers  {h:7.1f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-tar", nargs="*", default=[])
    ap.add_argument("--work", default="sap_work")
    ap.add_argument("--etiologies", nargs="*", default=None,
                    help="only keep these (short names: PD DS ALS CP Stroke). Default: all")
    ap.add_argument("--speakers-file", default=None, help="text file with one speaker ID per line")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    if not (work / "utterances.csv").exists():
        sys.exit("Run training/10_sap_inspect.py first (sap_work/utterances.csv is missing).")

    if not args.manifest_only:
        if not args.audio_tar:
            sys.exit("Pass --audio-tar (or --manifest-only).")
        wanted = None
        if args.etiologies or args.speakers_file:
            spk = load_csv(work / "speakers.csv")
            keep = set(s["speaker"] for s in spk)
            if args.etiologies:
                ets = {e.lower() for e in args.etiologies}
                keep = {s["speaker"] for s in spk if s["etiology_short"].lower() in ets}
            if args.speakers_file:
                listed = {l.strip() for l in open(args.speakers_file, encoding="utf-8") if l.strip()}
                keep &= listed
            wanted = keep
            print(f"Keeping {len(wanted)} speakers")
        extract(args.audio_tar, work, wanted, args.workers)

    build_manifest(work)


if __name__ == "__main__":
    main()

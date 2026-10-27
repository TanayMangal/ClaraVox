import json, os, sys, time
os.environ.setdefault("MODEL_REPO", os.path.abspath("models/SpeechCorrection"))
sys.path.insert(0, os.path.abspath("app/space"))
import app  # noqa: E402

t = time.time(); app._load_main(); print(f"load {time.time()-t:.1f}s on {app.device}")
for path in sys.argv[1:]:
    t = time.time(); r = app.process_audio(path); dt = time.time() - t
    print("\n==", os.path.basename(path), f"{dt:.2f}s")
    if r.get("error"):
        print("ERROR", r["error"]); continue
    print("raw n-best:", r["raw_nbest"])
    print("corrected:", r["corrected"], "| conf", r["confidence"], "| review", r["needs_review"])
    print("alternatives:", r["alternatives"])
    for i, w in enumerate(r["words"]):
        print(f"   [{i}] {w['text']:<14} conf={w['conf']:<5} alts={[a['text'] for a in w['alts']]}")
    weakest = min(range(len(r["words"])), key=lambda i: r["words"][i]["conf"]) if r["words"] else None
    if weakest is not None:
        t = time.time(); m = app.more_alternatives(path, r["raw"], r["corrected"], weakest)
        print(f"more_alternatives(word {weakest}) {time.time()-t:.2f}s:", [o["text"] for o in m["options"]], m.get("note"))
        target = app._norm_word(r["words"][weakest]["text"]) or "water"
        t = time.time(); p = app.practice_word(path, target)
        print(f"practice_word('{target}') {time.time()-t:.2f}s:", json.dumps(p))

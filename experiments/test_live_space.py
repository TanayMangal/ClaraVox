import json, sys, time
import requests

audio = sys.argv[1]
url = (sys.argv[2] if len(sys.argv) > 2 else "https://pineappleananas-speechcorrection.hf.space").rstrip("/")

def call(name, data, timeout=300):
    r = requests.post(f"{url}/gradio_api/call/{name}", json={"data": data}, timeout=60)
    r.raise_for_status()
    eid = r.json()["event_id"]
    text = requests.get(f"{url}/gradio_api/call/{name}/{eid}", timeout=timeout).text
    ev = None
    for line in text.splitlines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:") and ev == "complete":
            return json.loads(line[5:].strip())[0]
    raise RuntimeError(text[:300])

t = time.time()
with open(audio, "rb") as f:
    up = requests.post(f"{url}/gradio_api/upload", files={"files": (audio.split("/")[-1].split("\\")[-1], f)}, timeout=60)
up.raise_for_status()
fd = {"path": up.json()[0], "meta": {"_type": "gradio.FileData"}}
print(f"upload {time.time()-t:.1f}s")
t = time.time(); r = call("process_audio", [fd]); print(f"process_audio {time.time()-t:.1f}s")
print(json.dumps({k: r[k] for k in ("error", "raw", "corrected", "confidence", "needs_review", "alternatives")}, indent=1))
print("words:", [(w["text"], w["conf"], [a["text"] for a in w["alts"]]) for w in r["words"]])
if r.get("corrected"):
    i = min(range(len(r["words"])), key=lambda k: r["words"][k]["conf"])
    t = time.time(); m = call("more_alternatives", [fd, r["raw"], r["corrected"], i])
    print(f"more_alternatives {time.time()-t:.1f}s", [o["text"] for o in m["options"]])
    t = time.time(); p = call("practice_word", [fd, "water"]); print(f"practice_word {time.time()-t:.1f}s", p)

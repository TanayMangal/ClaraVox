import os, sys, threading, urllib.request

url, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
if os.path.exists(out) and os.path.getsize(out) == size:
    print("already complete", out); sys.exit(0)
parts = out + ".parts"
os.makedirs(parts, exist_ok=True)
chunk = 32 * 1024 * 1024
ranges = [(i, min(i + chunk, size) - 1) for i in range(0, size, chunk)]
lock = threading.Lock()
todo = list(range(len(ranges)))

def worker():
    while True:
        with lock:
            if not todo:
                return
            k = todo.pop(0)
        a, b = ranges[k]
        p = os.path.join(parts, f"{k:05d}")
        if os.path.exists(p) and os.path.getsize(p) == b - a + 1:
            continue
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(p + ".tmp", "wb") as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                if os.path.getsize(p + ".tmp") == b - a + 1:
                    os.replace(p + ".tmp", p)
                    break
            except Exception as e:  # noqa: BLE001
                print("retry", k, e, flush=True)
        print(f"chunk {k+1}/{len(ranges)}", flush=True)

ts = [threading.Thread(target=worker) for _ in range(n)]
[t.start() for t in ts]
[t.join() for t in ts]
missing = [k for k in range(len(ranges)) if not os.path.exists(os.path.join(parts, f"{k:05d}"))]
if missing:
    print("MISSING chunks", missing); sys.exit(1)
with open(out, "wb") as f:
    for k in range(len(ranges)):
        with open(os.path.join(parts, f"{k:05d}"), "rb") as c:
            f.write(c.read())
for k in range(len(ranges)):
    os.remove(os.path.join(parts, f"{k:05d}"))
os.rmdir(parts)
print("DONE", out, os.path.getsize(out))

import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
sys.path.insert(0, os.path.abspath("backend"))
import app
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
app._load_main()
app.process_audio(sys.argv[1])
configs = [(1, 0, 1), (2, 0, 2), (2, 2, 2), (3, 4, 3)]
for beams, samples, top in configs:
    app.N_ASR, app.N_SAMPLES, app.GEC_TOP = beams, samples, top
    ts, outs = [], []
    for p in sys.argv[1:]:
        app._enc_cache.clear()
        t = time.time(); r = app.process_audio(p); ts.append(time.time() - t)
        outs.append(r.get("corrected"))
    print(f"beams={beams} samples={samples} gec_top={top}: mean {sum(ts)/len(ts):.1f}s max {max(ts):.1f}s "
          f"(threads={torch.get_num_threads()})", flush=True)
    for o in outs:
        print("    ", o, flush=True)

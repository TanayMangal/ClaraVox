import cProfile, os, pstats, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
sys.path.insert(0, os.path.abspath("backend"))
import app
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
app._load_main()
app.process_audio(sys.argv[1])
t = time.time()
pr = cProfile.Profile(); pr.enable()
r = app.process_audio(sys.argv[1])
pr.disable()
print(f"process_audio {time.time()-t:.1f}s  N_ASR={app.N_ASR} N_SAMPLES={app.N_SAMPLES} GEC_TOP={app.GEC_TOP}")
print(r.get("corrected"))
st = pstats.Stats(pr); st.sort_stats("cumulative")
names = ("asr_nbest", "gec_nbest", "word_confidence", "word_network", "combine", "_load_audio", "_encode",
         "meaning_preserved", "generate", "safe_correction", "get_encoder")
for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
    if name in names:
        print(f"{name:20s} {os.path.basename(fn)}:{line:<5d} calls={nc:<4d} cum={ct:6.2f}s")
st.print_stats(25)

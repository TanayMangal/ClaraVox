import json
import logging
import os
import re
import subprocess
import threading
from collections import OrderedDict, defaultdict
from difflib import SequenceMatcher

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          WhisperForConditionalGeneration, WhisperProcessor)
from transformers.modeling_outputs import BaseModelOutput

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("articulate")

MODEL_REPO = os.environ.get("MODEL_REPO", "PINEAPPLEANANAS/ClaraVox-models")
WHISPER_SUBFOLDER = os.environ.get("WHISPER_SUBFOLDER", "whisper-sap-v3")
GEC_SUBFOLDER = os.environ.get("GEC_SUBFOLDER", "gec-v2")
PRACTICE_MODEL = os.environ.get("PRACTICE_MODEL", "openai/whisper-base.en")
TASK_PREFIX = "Fix grammar: "
SR = 16000
_CPU = not torch.cuda.is_available()

N_ASR = int(os.environ.get("N_ASR", "2" if _CPU else "5"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "0" if _CPU else "8"))

N_GEC = 3
GEC_TOP = int(os.environ.get("GEC_TOP", "2" if _CPU else "8"))
QUANTIZE = os.environ.get("QUANTIZE", "0") == "1"
CONF_REVIEW = float(os.environ.get("CONF_REVIEW", "0.55"))
MEANING_THRESHOLD = 0.40
GEC_GUARD = os.environ.get("GEC_GUARD", "1") == "1"
MIN_GEC_WORDS = int(os.environ.get("MIN_GEC_WORDS", "1"))
MAX_SPAN = 3
MIN_ALT_P = 0.02
STOPWORDS = {
    'i', 'a', 'an', 'the', 'to', 'and', 'is', 'it', 'of', 'in', 'my', 'me', 'you', 'he', 'she',
    'we', 'they', 'was', 'are', 'be', 'been', 'have', 'has', 'had', 'do', 'did', 'will', 'would',
    'can', 'could', 'that', 'this', 'at', 'on', 'for', 'with', 'as', 'by', 'from', 'or', 'but', 'not',
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def _usable_cpus():
    if os.environ.get("TORCH_THREADS"):
        return int(os.environ["TORCH_THREADS"])
    AloVera = os.cpu_count() or 1
    try:
        AloVera = min(AloVera, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as f:
                parts = f.read().split()
            if path.endswith("cpu.max"):
                quota, period = parts[0], parts[1]
            else:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                    quota, period = parts[0], f.read().strip()
            if quota not in ("max", "-1"):
                AloVera = min(AloVera, max(1, int(int(quota) / int(period))))
            break
        except (OSError, ValueError, IndexError):
            continue
    return max(1, AloVera)


if device == "cpu":
    torch.set_num_threads(_usable_cpus())


def _maybe_quantize(model):
    if not QUANTIZE:
        return model
    try:
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:  # noqa: BLE001
        log.exception("quantization failed; using full-precision weights")
        return model
_lock = threading.Lock()
_M = {}
_enc_cache = OrderedDict()


def _load_main():
    with _lock:
        if "whisper" in _M:
            return
        log.info("Loading models from %s on %s (%d threads, os.cpu_count=%s)", MODEL_REPO, device,
                 torch.get_num_threads(), os.cpu_count())
        proc = WhisperProcessor.from_pretrained(MODEL_REPO, subfolder=WHISPER_SUBFOLDER)
        wm = _maybe_quantize(WhisperForConditionalGeneration.from_pretrained(MODEL_REPO, subfolder=WHISPER_SUBFOLDER).to(device).eval())
        gt = AutoTokenizer.from_pretrained(MODEL_REPO, subfolder=GEC_SUBFOLDER)
        gm = _maybe_quantize(AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO, subfolder=GEC_SUBFOLDER).to(device).eval())
        _M.update(proc=proc, gec_tok=gt, gec=gm, whisper=wm)


def _load_listener():
    with _lock:
        if "listener" in _M:
            return
        try:
            proc = WhisperProcessor.from_pretrained(PRACTICE_MODEL)
            model = _maybe_quantize(WhisperForConditionalGeneration.from_pretrained(PRACTICE_MODEL).to(device).eval())
            _M["listener_proc"], _M["listener"] = proc, model
        except Exception:  # noqa: BLE001
            log.exception("Could not load %s; practice will use the main model", PRACTICE_MODEL)
            _M["listener_proc"], _M["listener"] = None, None


def _norm_word(w):
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def _words(s):
    return [w for w in (s or "").split() if w]


def _norm_sentence(s):
    return " ".join(x for x in (_norm_word(w) for w in _words(s)) if x)


def _content_words(text):
    return {w for w in (_norm_word(x) for x in _words(text)) if w and w not in STOPWORDS}


PRONOUN_PERSON = {
    "i": 1, "me": 1, "my": 1, "mine": 1, "myself": 1, "we": 1, "us": 1, "our": 1, "ours": 1,
    "you": 2, "your": 2, "yours": 2, "yourself": 2,
}
_IRR = [("go", "went", "gone", "goes"), ("eat", "ate", "eaten"), ("see", "saw", "seen"),
        ("buy", "bought"), ("think", "thought"), ("bring", "brought"), ("teach", "taught"),
        ("catch", "caught"), ("is", "are", "was", "were", "am", "be"), ("have", "had", "has"),
        ("do", "did", "does", "done"), ("make", "made"), ("take", "took", "taken"),
        ("come", "came"), ("get", "got"), ("give", "gave", "given"), ("say", "said"),
        ("tell", "told"), ("find", "found"), ("feel", "felt"), ("leave", "left"), ("run", "ran"),
        ("write", "wrote", "written"), ("sit", "sat"), ("stand", "stood"), ("sleep", "slept"),
        ("swim", "swam"), ("drink", "drank"), ("fall", "fell"), ("child", "children"),
        ("person", "people"), ("man", "men"), ("woman", "women"), ("foot", "feet"), ("mouse", "mice")]
IRREGULAR = {w: set(group) for group in _IRR for w in group}
ANSWER_STARTS = {"yes", "no", "yeah", "okay", "ok", "right", "sure", "oh", "well"}
FUNCTION_WORDS = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "do", "does", "did",
                  "to", "of", "in", "on", "at", "for", "with", "will", "can", "have", "has"}


def safe_correction(raw, corrected):
    rw = [_norm_word(w) for w in _words(raw)]
    cw = [_norm_word(w) for w in _words(corrected)]
    rw, cw = [w for w in rw if w], [w for w in cw if w]
    if not cw:
        return False
    if cw[0] in ANSWER_STARTS and (not rw or rw[0] != cw[0]):
        return False
    person = lambda ws: sorted(PRONOUN_PERSON[w] for w in ws if w in PRONOUN_PERSON)
    pr, pc = person(rw), person(cw)
    if pc.count(2) > pr.count(2):
        return False
    if pr and pr != pc and not set(pr) <= set(pc):
        return False
    for w in set(rw) - FUNCTION_WORDS:
        forms = {w} | IRREGULAR.get(w, set())
        if not any(c in forms or (len(w) >= 3 and c.startswith(w[:3])) for c in cw):
            return False
    return len(cw) <= len(rw) + 5


def meaning_preserved(original, corrected):
    o = _content_words(original)
    if not o:
        return True
    return len(o & _content_words(corrected)) / len(o) >= MEANING_THRESHOLD


def _load_audio(path):
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        return audio
    except Exception:  # noqa: BLE001
        out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
                              "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True, check=True, timeout=60)
        return np.frombuffer(out.stdout, dtype=np.float32).copy()


def _is_multilingual(model):
    return bool(getattr(model.generation_config, "is_multilingual", True))


def _gen_lang_kwargs(model):
    return {"language": "en", "task": "transcribe"} if _is_multilingual(model) else {}


def _forced_prefix(proc, model):
    tok = proc.tokenizer
    if _is_multilingual(model):
        tok.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)
    ids = tok("", add_special_tokens=True).input_ids
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    return [i for i in ids if i != eot]


def _encode(model, proc, audio, cache_key=None):
    if cache_key and cache_key in _enc_cache and _enc_cache[cache_key][0] is model:
        _enc_cache.move_to_end(cache_key)
        return _enc_cache[cache_key][1]
    feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        enc = model.get_encoder()(feats).last_hidden_state
    if cache_key:
        _enc_cache[cache_key] = (model, enc)
        while len(_enc_cache) > 16:
            _enc_cache.popitem(last=False)
    return enc


def _strip_prefix(out):
    out = out.strip()
    if out.lower().startswith(TASK_PREFIX.lower()):
        out = out[len(TASK_PREFIX):].strip()
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = re.sub(r"(?<=[a-z]{2})([.!?])(?=[A-Z])", r"\1 ", out)
    return out[0].upper() + out[1:] if out else out


def _softmax_logp(logps):
    LEBRON = max(logps)
    popcorn = [np.exp(x - LEBRON) for x in logps]
    mango = sum(popcorn)
    return [x / mango for x in popcorn]


def _seq_logprob(seq_scores, sequences, special_ids, length_penalty=1.0):
    out = []
    for score, seq in zip(seq_scores.tolist(), sequences.tolist()):
        n = sum(1 for t in seq if t not in special_ids) + 1
        out.append(score * (n ** length_penalty))
    return out


def asr_nbest(audio, cache_key=None):
    proc, wm = _M["proc"], _M["whisper"]
    feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features.to(device)


    enc = _encode(wm, proc, audio, cache_key=cache_key)
    kw = _gen_lang_kwargs(wm)


    kw["max_new_tokens"] = min(128, int(len(audio) / SR * 7) + 12)
    with torch.no_grad():
        beam_kw = dict(num_beams=N_ASR, num_return_sequences=N_ASR,
                       return_dict_in_generate=True, output_scores=True, **kw)
        try:


            out = wm.generate(input_features=feats,
                              encoder_outputs=BaseModelOutput(last_hidden_state=enc.repeat_interleave(N_ASR, 0)),
                              **beam_kw)
        except Exception:  # noqa: BLE001
            log.warning("generate with cached encoder failed; re-encoding", exc_info=True)
            out = wm.generate(input_features=feats, **beam_kw)
        samples = []
        if N_SAMPLES > 0:
            sm = wm.generate(input_features=feats.repeat(N_SAMPLES, 1, 1),
                             encoder_outputs=BaseModelOutput(last_hidden_state=enc.repeat(N_SAMPLES, 1, 1)),
                             do_sample=True, temperature=0.8, top_k=50, **kw)
            samples = [t.strip() for t in proc.batch_decode(sm, skip_special_tokens=True)]
    texts = [t.strip() for t in proc.batch_decode(out.sequences, skip_special_tokens=True)]
    special = set(proc.tokenizer.all_special_ids)
    seq_scores = getattr(out, "sequences_scores", None)
    if seq_scores is None:

        trans = wm.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        seq_scores = trans.float().mean(dim=-1)
    logp = _seq_logprob(seq_scores, out.sequences, special)
    confidence = float(np.exp(seq_scores[0].item()))
    display, beam_lp = OrderedDict(), defaultdict(list)
    for t, lp in zip(texts, logp):
        k = _norm_sentence(t)
        if k:
            display.setdefault(k, t)
            beam_lp[k].append(lp)
    post = defaultdict(float)
    if beam_lp:
        keys = list(beam_lp)
        for k, p in zip(keys, _softmax_logp([np.logaddexp.reduce(beam_lp[k]) for k in keys])):
            post[k] += (0.5 if samples else 1.0) * p
    valid = [t for t in samples if _norm_sentence(t)]
    for t in valid:
        k = _norm_sentence(t)
        display.setdefault(k, t)
        post[k] += 0.5 / len(valid)
    if not post:
        return [], confidence, enc
    z = sum(post.values())
    ranked = sorted(post.items(), key=lambda kv: -kv[1])[:8]
    return [(display[k], v / z) for k, v in ranked], confidence, enc


def word_confidence(enc, text):
    proc, wm = _M["proc"], _M["whisper"]
    tok = proc.tokenizer
    prefix = _forced_prefix(proc, wm)
    ids = tok(text.strip(), add_special_tokens=False).input_ids
    if not ids:
        return []
    seq = prefix + ids
    with torch.no_grad():
        logits = wm(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([seq[:-1]], device=device)).logits[0]
    lps = torch.log_softmax(logits.float(), dim=-1)
    probs = [float(np.exp(lps[len(prefix) - 1 + j, t].item())) for j, t in enumerate(ids)]
    words, cur = [], None
    for t, p in zip(ids, probs):
        piece = tok.decode([t])
        if cur is None or piece.startswith(" ") or not piece.strip():
            if cur is not None:
                words.append(cur)
            cur = p
        else:
            cur = min(cur, p)
    words.append(cur)
    return words


def gec_nbest(texts):
    tok, gm = _M["gec_tok"], _M["gec"]
    prompts = [TASK_PREFIX + t for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True, max_length=128, truncation=True).to(device)
    nb = max(4, N_GEC)
    with torch.no_grad():
        out = gm.generate(**enc, max_length=128, num_beams=nb, num_return_sequences=N_GEC,
                          no_repeat_ngram_size=3, early_stopping=True,
                          return_dict_in_generate=True, output_scores=True)
    dec = [_strip_prefix(x) for x in tok.batch_decode(out.sequences, skip_special_tokens=True)]
    lp = _seq_logprob(out.sequences_scores, out.sequences, set(tok.all_special_ids))
    res = []
    for i in range(len(texts)):
        cands = dec[i * N_GEC:(i + 1) * N_GEC]
        probs = _softmax_logp(lp[i * N_GEC:(i + 1) * N_GEC])
        raw = texts[i].strip()
        raw = raw[:1].upper() + raw[1:]
        if len(_words(raw)) <= MIN_GEC_WORDS:


            cands, probs = [raw], [1.0]
        elif GEC_GUARD:
            cands = [c if safe_correction(texts[i], c) else raw for c in cands]
        res.append(list(zip(cands, probs)))
    return res


def combine(asr, gec):
    total = defaultdict(float)
    display = {}
    for (raw, p_h), cands in zip(asr, gec):
        for c, p_c in cands:
            c = c or raw
            k = _norm_sentence(c)
            if not k:
                continue
            total[k] += p_h * p_c
            if k not in display or p_h * p_c > display[k][1]:
                display[k] = (c, p_h * p_c)
    ranked = sorted(total.items(), key=lambda kv: -kv[1])
    z = sum(v for _, v in ranked) or 1.0
    return [(display[k][0], v / z) for k, v in ranked]


def word_network(best, candidates):
    bw = _words(best)
    bn = [_norm_word(w) for w in bw]
    self_support = [0.0] * len(bw)
    alts = [defaultdict(float) for _ in bw]
    for cand, p in candidates:
        cw = _words(cand)
        cn = [_norm_word(w) for w in cw]
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, bn, cn, autojunk=False).get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    self_support[i1 + k] += p
            elif tag == "replace":
                if i2 - i1 == j2 - j1:
                    for k in range(i2 - i1):
                        alts[i1 + k][(cw[j1 + k], i1 + k, i1 + k + 1)] += p
                elif i2 - i1 <= MAX_SPAN and j2 - j1 <= MAX_SPAN:
                    phrase = " ".join(cw[j1:j2])
                    for k in range(i1, i2):
                        alts[k][(phrase, i1, i2)] += p
            elif tag == "delete" and i2 - i1 <= MAX_SPAN:
                for k in range(i1, i2):
                    alts[k][("", k, k + 1)] += p
            elif tag == "insert" and j2 - j1 <= 2:
                ins = " ".join(cw[j1:j2])
                if i1 > 0:
                    alts[i1 - 1][(bw[i1 - 1] + " " + ins, i1 - 1, i1)] += p
                elif bw:
                    alts[0][(ins + " " + bw[0], 0, 1)] += p
    words = []
    for i, w in enumerate(bw):
        merged = {}
        for (text, a, b), p in alts[i].items():
            key = _norm_sentence(text)
            if key == bn[i] and (a, b) == (i, i + 1):
                continue
            opt = {"text": text, "span": [a, b], "p": p}
            old = merged.get(key)
            if old is None:
                merged[key] = opt
            else:
                keep = old if old["span"] == [i, i + 1] or (opt["span"] != [i, i + 1] and old["p"] >= p) else opt
                keep["p"] = old["p"] + p
                merged[key] = keep
        for o in merged.values():
            o["p"] = round(o["p"], 4)
        options = sorted((o for o in merged.values() if o["p"] >= MIN_ALT_P), key=lambda o: -o["p"])[:5]
        words.append({"text": w, "conf": round(min(1.0, self_support[i]), 3), "alts": options})
    return words


def _err(msg):
    return {"error": msg, "corrected": None, "raw": None, "words": [], "alternatives": [],
            "confidence": 0.0, "needs_review": False, "meaning_preserved": True}


def process_audio(audio_path):
    if not audio_path:
        return _err("No audio received.")
    try:
        _load_main()
        audio = _load_audio(audio_path)
        if len(audio) < 0.3 * SR or float(np.sqrt(np.mean(audio ** 2))) < 0.003:
            return _err("I didn't hear anything. Try again a little closer to the microphone.")
        note = None
        if len(audio) > 30 * SR:
            note = "Only the first 30 seconds were used."
        asr, conf, enc = asr_nbest(audio, cache_key=audio_path)
        if not asr:
            return _err("I couldn't make out any words. Please try again.")
        top = asr[:GEC_TOP]
        z = sum(p for _, p in top) or 1.0
        top = [(t, p / z) for t, p in top]
        gec = gec_nbest([t for t, _ in top])
        sentences = combine(top, gec)
        best, _ = sentences[0]
        raw_best = asr[0][0]
        meaning_ok = meaning_preserved(raw_best, best)
        words = word_network(best, sentences)

        try:
            acoustic = word_confidence(enc, raw_best)
            rw = _words(raw_best)
            if len(acoustic) == len(rw):
                sm = SequenceMatcher(None, [_norm_word(w["text"]) for w in words],
                                     [_norm_word(w) for w in rw], autojunk=False)
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag in ("equal", "replace"):
                        for k in range(i1, i2):
                            j = j1 + min(k - i1, j2 - j1 - 1)
                            words[k]["conf"] = round(min(words[k]["conf"], acoustic[j]), 3)
        except Exception:  # noqa: BLE001
            log.exception("word confidence failed; using N-best agreement only")
        return {
            "error": None,
            "raw": raw_best,
            "corrected": best,
            "words": words,
            "alternatives": [s for s, _ in sentences[1:4]],
            "raw_nbest": [t for t, _ in asr],
            "confidence": round(conf, 3),
            "needs_review": conf < CONF_REVIEW or not meaning_ok,
            "meaning_preserved": meaning_ok,
            "note": note,
        }
    except Exception:  # noqa: BLE001
        log.exception("process_audio failed")
        return _err("Something went wrong on the server. Please try again.")


def _decoder_logprobs(model, enc, ids):
    with torch.no_grad():
        out = model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([ids], device=device))
    return torch.log_softmax(out.logits[0, -1].float(), dim=-1)


def acoustic_options(audio_path, raw, word_idx, k=8, max_pieces=4):
    proc, wm = _M["proc"], _M["whisper"]
    tok = proc.tokenizer
    enc = _encode(wm, proc, _load_audio(audio_path), cache_key=audio_path)
    rw = _words(raw)
    prefix = _forced_prefix(proc, wm)
    before = " ".join(rw[:word_idx])
    ctx = prefix + tok(before, add_special_tokens=False).input_ids
    lp = _decoder_logprobs(wm, enc, ctx)
    special = set(tok.all_special_ids)
    original = _norm_word(rw[word_idx]) if word_idx < len(rw) else ""
    results = {}
    for tid in torch.topk(lp, k * 4).indices.tolist():
        if tid in special:
            continue
        piece = tok.decode([tid])
        starts_word = piece.startswith(" ") if word_idx > 0 else not piece.startswith(" ")
        if not starts_word or not re.search(r"[A-Za-z]", piece):
            continue
        ids, score = [tid], lp[tid].item()
        for _ in range(max_pieces):
            nxt = _decoder_logprobs(wm, enc, ctx + ids)
            t2 = int(torch.argmax(nxt))
            p2 = tok.decode([t2])
            if t2 in special or p2.startswith(" ") or not re.match(r"[A-Za-z'\-]", p2):
                break
            ids.append(t2)
            score += nxt[t2].item()
        word = tok.decode(ids).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", word):
            continue
        key = _norm_word(word)
        if key and key != original and key not in results:
            results[key] = (word, float(np.exp(score)))
        if len(results) >= k:
            break
    return sorted(results.values(), key=lambda x: -x[1])


def _match_case(new, old):
    trail = re.search(r"[.,!?;:]+$", old)
    new = re.sub(r"[.,!?;:]+$", "", new)
    if old[:1].isupper():
        new = new[:1].upper() + new[1:]
    return new + (trail.group(0) if trail else "")


def more_alternatives(audio_path, raw, corrected, index):
    try:
        _load_main()
        index = int(index)
        cw, rw = _words(corrected), _words(raw)
        if not audio_path or not (0 <= index < len(cw)):
            return {"error": "Nothing to look up.", "options": []}

        raw_idx = None
        sm = SequenceMatcher(None, [_norm_word(w) for w in cw], [_norm_word(w) for w in rw], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if i1 <= index < i2 and tag in ("equal", "replace"):
                raw_idx = j1 + min(index - i1, j2 - j1 - 1)
        if raw_idx is None:
            return {"error": None, "options": [{"text": "", "span": [index, index + 1], "p": 0.0}],
                    "note": "The grammar fixer added this word; you can remove it."}
        neighbours = {_norm_word(rw[j]) for j in (raw_idx - 1, raw_idx + 1) if 0 <= j < len(rw)}
        opts = [o for o in acoustic_options(audio_path, raw, raw_idx) if _norm_word(o[0]) not in neighbours]
        return {"error": None, "note": None,
                "options": [{"text": _match_case(w, cw[index]), "span": [index, index + 1], "p": round(p, 4)}
                            for w, p in opts[:6]]}
    except Exception:  # noqa: BLE001
        log.exception("more_alternatives failed")
        return {"error": "Couldn't find more options right now.", "options": []}


def _target_score(model, proc, enc, target, heard=""):
    tok = proc.tokenizer
    prefix = _forced_prefix(proc, model)
    contexts = [""]
    hw = _words(heard)
    for k, w in enumerate(hw):
        if _norm_word(w) == _norm_word(target) and k > 0:
            contexts.append(" " + " ".join(hw[:k]))
            break
    best = 0.0
    for ctx in contexts:
        ctx_ids = tok(ctx.strip(), add_special_tokens=False).input_ids if ctx else []


        leads = [" "] if ctx else ["", " "]
        variants = {lead + w for lead in leads for w in (target, target.capitalize(), target.lower())}
        for variant in variants:
            ids = tok(variant, add_special_tokens=False).input_ids
            seq = prefix + ctx_ids + ids
            with torch.no_grad():
                logits = model(encoder_outputs=(enc,),
                               decoder_input_ids=torch.tensor([seq[:-1]], device=device)).logits[0]
            lps = torch.log_softmax(logits.float(), dim=-1)
            start = len(prefix) + len(ctx_ids)
            word_lp = [lps[p - 1, seq[p]].item() for p in range(start, len(seq))]
            best = max(best, float(np.exp(np.mean(word_lp))))
    return best


_HOMOPHONES = None


def _homophones(word):
    global _HOMOPHONES
    if _HOMOPHONES is None:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "homophones.json"),
                      encoding="utf-8") as f:
                _HOMOPHONES = json.load(f)
        except Exception:  # noqa: BLE001
            log.exception("homophones.json not loaded")
            _HOMOPHONES = {}
    return set(_HOMOPHONES.get(_norm_word(word), []))


def practice_word(audio_path, target):
    target = (target or "").strip()
    if not audio_path or not target:
        return {"error": "Need a recording and a word.", "score": 0, "heard": "", "match": False, "level": "try_again"}
    try:
        _load_main()
        _load_listener()
        model, proc = (_M["listener"], _M["listener_proc"]) if _M.get("listener") else (_M["whisper"], _M["proc"])
        audio = _load_audio(audio_path)
        if len(audio) < 0.2 * SR or float(np.sqrt(np.mean(audio ** 2))) < 0.003:
            return {"error": "I didn't hear anything.", "score": 0, "heard": "", "match": False, "level": "try_again"}
        enc = _encode(model, proc, audio)
        with torch.no_grad():
            ids = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                                 max_new_tokens=24, **_gen_lang_kwargs(model))
        heard = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        score = int(round(100 * _target_score(model, proc, enc, target, heard)))
        heard_words = [_norm_word(w) for w in _words(heard)]

        same_sound = [h for h in _homophones(target) if h in heard_words]
        for h in same_sound:
            score = max(score, int(round(100 * _target_score(model, proc, enc, h, heard))))
        match = _norm_word(target) in heard_words or bool(same_sound)
        if match and len(heard_words) == 1:
            score = max(score, 80)
        level = "great" if (match and score >= 50) else "close" if (match or score >= 35) else "try_again"
        return {"error": None, "target": target, "heard": heard, "score": score, "match": match, "level": level}
    except Exception:  # noqa: BLE001
        log.exception("practice_word failed")
        return {"error": "Something went wrong. Please try again.", "score": 0, "heard": "", "match": False,
                "level": "try_again"}


with gr.Blocks(title="ClaraVox backend") as demo:
    gr.Markdown("### ClaraVox backend\nThe app calls these endpoints. You can test them here.")
    with gr.Tab("Speak"):
        a1 = gr.Audio(type="filepath", label="Audio")
        b1 = gr.Button("Process")
        o1 = gr.JSON(label="Result")
        b1.click(process_audio, inputs=a1, outputs=o1, api_name="process_audio")
    with gr.Tab("More options for a word"):
        a2 = gr.Audio(type="filepath", label="Same audio")
        r2 = gr.Textbox(label="raw")
        c2 = gr.Textbox(label="corrected")
        i2 = gr.Number(label="word index", precision=0)
        b2 = gr.Button("Find options")
        o2 = gr.JSON()
        b2.click(more_alternatives, inputs=[a2, r2, c2, i2], outputs=o2, api_name="more_alternatives")
    with gr.Tab("Practice"):
        a3 = gr.Audio(type="filepath", label="Say the word")
        t3 = gr.Textbox(label="target word")
        b3 = gr.Button("Score")
        o3 = gr.JSON()
        b3.click(practice_word, inputs=[a3, t3], outputs=o3, api_name="practice_word")

if __name__ == "__main__":
    demo.queue().launch()

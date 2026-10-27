import json
import re
from pathlib import Path

ETIOLOGY_SHORT = {
    "parkinson's disease": "PD",
    "down syndrome": "DS",
    "als": "ALS",
    "cerebral palsy": "CP",
    "stroke": "Stroke",
}


def short_etiology(name: str) -> str:
    return ETIOLOGY_SHORT.get((name or "").strip().lower(), (name or "Unknown").strip() or "Unknown")


def speaker_from_filename(filename: str) -> str:
    return Path(filename).name.split("_")[0]


def utt_id(filename: str) -> str:
    return Path(filename).stem


_PAREN = re.compile(r"\(([^()]*)\)")
_CURLY = re.compile(r"\{([^{}]*)\}")
_SQUARE = re.compile(r"\[([^\[\]]*)\]")


def _labelled(content: str):
    m = re.match(r"\s*([A-Za-z]{1,3})\s*:\s*(.*)$", content, flags=re.S)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return None, content.strip()


def clean_transcript(text: str):
    if text is None:
        return "", "no_transcript"
    t = str(text)
    if t.lstrip().startswith("#ts"):

        t = " ".join(re.sub(r"^\s*[\d.]+\s+[\d.]+\s*", "", line).strip()
                     for line in t.lstrip()[3:].split("\n"))
    t = re.sub(r"^\s*#dis\s*", "", t)
    t = t.replace("\r", " ").replace("\n", " ")

    for m in _SQUARE.findall(t):
        if m.strip().lower() == "pii":
            return "", "pii"
    t = _SQUARE.sub(" ", t)

    t = re.sub(r"^[^\]]*\]", " ", t) if "]" in t and "[" not in t else t

    def curly(m):
        label, body = _labelled(m.group(1))
        if label in ("w", "u", "f") or not body:
            raise _Unknown()
        return " " + body + " "

    try:
        t = _CURLY.sub(curly, t)
    except _Unknown:
        return "", "unintelligible"

    if re.search(r"\(\s*cs\s*:", t, flags=re.I):
        return "", "caregiver_voice"

    def paren(m):
        label, body = _labelled(m.group(1))
        return (" " + body + " ") if label == "ss" else " "

    t = _PAREN.sub(paren, t)


    if re.search(r"[\[\]{}()]", t):
        return "", "bad_markup"

    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"^[,.;:]\s*", "", t)
    if not re.search(r"[A-Za-z0-9]", t):
        return "", "empty"
    return t, None


class _Unknown(Exception):
    pass


def iter_speaker_json(json_dir: Path):
    for p in sorted(Path(json_dir).rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not read {p}: {e}")
            continue
        if isinstance(data, dict) and isinstance(data.get("Files"), list):
            yield p, data


def records_from_speaker_json(data: dict):
    files = data.get("Files") or []
    spk_from_json = str(data.get("Contributor ID") or "").strip()
    etiology = str(data.get("Etiology") or "").strip()
    out, seen = [], set()
    for item in files:
        if not isinstance(item, dict):
            continue
        fname = str(item.get("Filename") or "").strip()
        if not fname or fname in seen:
            continue
        seen.add(fname)
        prompt = item.get("Prompt") or {}
        raw = prompt.get("Transcript")
        clean, drop = clean_transcript(raw)
        out.append({
            "id": utt_id(fname),
            "filename": Path(fname).name,
            "speaker": spk_from_json or speaker_from_filename(fname),
            "etiology": etiology,
            "etiology_short": short_etiology(etiology),
            "category": (prompt.get("Category Description") or "").strip(),
            "subcategory": (prompt.get("Sub Category Description") or "").strip(),
            "text_raw": raw if raw is not None else "",
            "text": clean,
            "drop_reason": drop or "",
        })
    return out

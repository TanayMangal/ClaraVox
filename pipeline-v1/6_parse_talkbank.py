import os
import re
import csv
import zipfile
import tempfile
import shutil
from pathlib import Path
from difflib import SequenceMatcher


ZIP_FILES = [
    "Flusberg.zip",
    "Rollins.zip",
    "Eigsti.zip",
    "AAC.zip",
    "QuigleyMcNally.zip",
    "NYU-Emerson.zip",
    "Nadig.zip",
    "AFG_datap1.zip",
]

OUTPUT_CSV = "gec_pairs.csv"
MIN_WORDS  = 2
MAX_WORDS  = 40


UNINTELLIGIBLE = re.compile(r'\b(xxx|yyy|www|0)\b')
CHAT_CLEANUP   = re.compile(r'[&+\[\]<>@%]|\(\.+\)|_|\+[/!?]')

CHAT_CODES     = re.compile(r'&-\w+|\[:\s*[^\]]+\]|\[!\]|\[//\]|\[/\]|\[=!\s*[^\]]+\]|\+\.\.\.')

def clean_chat_line(text: str) -> str:
    text = CHAT_CODES.sub('', text)
    text = CHAT_CLEANUP.sub(' ', text)

    text = re.sub(r'\s*[.!?]+\s*$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_usable(text: str) -> bool:
    if not text:
        return False
    if UNINTELLIGIBLE.search(text):
        return False
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False

    if all(w.lower() in {'uh', 'um', 'mm', 'hmm', 'oh', 'ah', 'yeah', 'no', 'yes', 'ok'} for w in words):
        return False
    return True

def parse_cha_file(filepath: str):
    utterances = []
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return utterances

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')


        if line.startswith('*'):
            match = re.match(r'^\*([A-Z]+):\t(.+)', line)
            if match:
                code = match.group(1)
                text = match.group(2)


                j = i + 1
                while j < len(lines) and lines[j].startswith('\t') and not lines[j].startswith('\t%'):
                    text += ' ' + lines[j].strip()
                    j += 1

                text = clean_chat_line(text)
                speaker = 'CHI' if code == 'CHI' else 'ADU'
                utterances.append((speaker, text, code))
        i += 1

    return utterances


def word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)

def is_echo_or_recast(chi_text: str, adu_text: str) -> bool:
    chi_words = chi_text.lower().split()
    adu_words = adu_text.lower().split()

    overlap = word_overlap(chi_text, adu_text)
    seq_ratio = SequenceMatcher(None, chi_text.lower(), adu_text.lower()).ratio()


    if overlap >= 0.55 and len(adu_words) >= len(chi_words) and chi_text.lower() != adu_text.lower():
        return True

    if seq_ratio >= 0.7 and len(adu_words) > len(chi_words):
        return True
    return False

def extract_echo_pairs(utterances):
    pairs = []
    for i, (speaker, text, code) in enumerate(utterances):
        if speaker != 'CHI':
            continue
        if not is_usable(text):
            continue


        for j in range(i + 1, min(i + 4, len(utterances))):
            nspeaker, ntext, ncode = utterances[j]
            if nspeaker == 'CHI':
                break
            if nspeaker == 'ADU' and is_usable(ntext):
                if is_echo_or_recast(text, ntext):
                    pairs.append({
                        'source': text,
                        'target': capitalize(ntext),
                        'pair_type': 'echo_recast',
                    })
                    break

    return pairs

def capitalize(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


import random
random.seed(42)

ARTICLES = {'a', 'an', 'the'}
AUX_VERBS = {'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
              'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
              'can', 'could', 'shall', 'should', 'may', 'might', 'must'}
PRONOUNS = {'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their'}

def drop_articles(text: str) -> str:
    words = text.split()
    result = []
    for w in words:
        if w.lower() in ARTICLES and random.random() < 0.6:
            continue
        result.append(w)
    return ' '.join(result) if result else text

def drop_subject(text: str) -> str:
    words = text.split()
    if words and words[0].lower() in PRONOUNS and len(words) > 2:
        return ' '.join(words[1:])
    return text

def drop_aux(text: str) -> str:
    words = text.split()
    result = []
    for i, w in enumerate(words):
        if w.lower() in AUX_VERBS and i > 0 and random.random() < 0.5:
            continue
        result.append(w)
    return ' '.join(result) if result else text

def make_telegraphic(text: str) -> str:
    funcs = [drop_articles, drop_subject, drop_aux]
    random.shuffle(funcs)
    result = text
    for fn in funcs[:random.randint(1, 2)]:
        result = fn(result)
    return result

def augment_chi_utterance(text: str):
    corrected = capitalize(text)
    broken = make_telegraphic(text)
    if broken == corrected.lower() or broken == text:
        return None
    return {
        'source': broken,
        'target': corrected,
        'pair_type': 'synthetic_augmented',
    }


def find_cha_files(directory: str):
    found = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.cha') or f.endswith('.cha.txt'):
                found.append(os.path.join(root, f))
    return found

def extract_zip_to_tmp(zip_path: str) -> str:
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp)
    return tmp


def process_corpus(zip_path: str, corpus_name: str):
    print(f"  Processing {corpus_name}...")
    tmp_dir = extract_zip_to_tmp(zip_path)
    cha_files = find_cha_files(tmp_dir)
    print(f"    Found {len(cha_files)} .cha files")

    all_pairs  = []
    chi_count  = 0

    for cha_file in cha_files:
        utterances = parse_cha_file(cha_file)


        echo_pairs = extract_echo_pairs(utterances)
        for p in echo_pairs:
            p['corpus'] = corpus_name
        all_pairs.extend(echo_pairs)


        for speaker, text, code in utterances:
            if speaker == 'CHI' and is_usable(text):
                chi_count += 1
                aug = augment_chi_utterance(text)
                if aug:
                    aug['corpus'] = corpus_name
                    all_pairs.append(aug)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    echo_count = sum(1 for p in all_pairs if p['pair_type'] == 'echo_recast')
    synth_count = sum(1 for p in all_pairs if p['pair_type'] == 'synthetic_augmented')
    print(f"    CHI utterances: {chi_count} | Echo pairs: {echo_count} | Synthetic pairs: {synth_count}")
    return all_pairs


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_pairs  = []

    print("=" * 60)
    print("TalkBank CHAT → GEC Pairs Parser")
    print("=" * 60)

    for zip_name in ZIP_FILES:
        zip_path = os.path.join(script_dir, zip_name)
        if not os.path.exists(zip_path):
            print(f"  WARNING: {zip_name} not found, skipping.")
            continue
        corpus_name = zip_name.replace('.zip', '')
        pairs = process_corpus(zip_path, corpus_name)
        all_pairs.extend(pairs)


    seen   = set()
    unique = []
    for p in all_pairs:
        key = (p['source'].lower().strip(), p['target'].lower().strip())
        if key not in seen and p['source'].lower() != p['target'].lower():
            seen.add(key)
            unique.append(p)

    print(f"\nTotal unique pairs: {len(unique)}")


    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'target', 'pair_type', 'corpus'])
        writer.writeheader()
        writer.writerows(unique)


    echo_total   = sum(1 for p in unique if p['pair_type'] == 'echo_recast')
    synth_total  = sum(1 for p in unique if p['pair_type'] == 'synthetic_augmented')
    corpora_seen = sorted(set(p['corpus'] for p in unique))

    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"  Echo/recast pairs:   {echo_total}")
    print(f"  Synthetic pairs:     {synth_total}")
    print(f"  Corpora used:        {corpora_seen}")
    print("\nSample pairs:")
    for p in unique[:5]:
        print(f"  [{p['pair_type']}]")
        print(f"    SRC: {p['source']}")
        print(f"    TGT: {p['target']}")

import os
import csv
import re

TORGO_DIR = "torgo_data"
OUTPUT_CSV = "torgo_pairs.csv"

def get_transcript(prompts_dir, utterance_id):


    prompt_file = os.path.join(prompts_dir, utterance_id + ".txt")
    if not os.path.exists(prompt_file):
        prompt_file = os.path.join(prompts_dir, utterance_id)
    if not os.path.exists(prompt_file):
        return None

    with open(prompt_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()


    if text == "xxx" or text.endswith(".jpg") or not text:
        return None

    return text

def parse_torgo(torgo_dir):
    pairs = []

    for speaker in sorted(os.listdir(torgo_dir)):
        speaker_dir = os.path.join(torgo_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue


        is_dysarthric = not ("C" in speaker[1:3])
        gender = "F" if speaker.startswith("F") else "M"

        for session in sorted(os.listdir(speaker_dir)):
            session_dir = os.path.join(speaker_dir, session)
            if not os.path.isdir(session_dir) or not session.startswith("Session"):
                continue


            wav_dir = os.path.join(session_dir, "wav_headMic")
            if not os.path.exists(wav_dir):
                wav_dir = os.path.join(session_dir, "wav_arrayMic")
            if not os.path.exists(wav_dir):
                continue

            prompts_dir = os.path.join(session_dir, "prompts")
            if not os.path.exists(prompts_dir):
                continue

            for wav_file in sorted(os.listdir(wav_dir)):
                if not wav_file.endswith(".wav"):
                    continue

                utterance_id = os.path.splitext(wav_file)[0]
                wav_path = os.path.abspath(os.path.join(wav_dir, wav_file))
                transcript = get_transcript(prompts_dir, utterance_id)

                if transcript is None:
                    continue

                pairs.append({
                    "audio_path": wav_path,
                    "transcript": transcript,
                    "speaker": speaker,
                    "session": session,
                    "gender": gender,
                    "is_dysarthric": is_dysarthric,
                })

    return pairs

print("Parsing TORGO dataset...")
pairs = parse_torgo(TORGO_DIR)


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "speaker", "session", "gender", "is_dysarthric"])
    writer.writeheader()
    writer.writerows(pairs)


dysarthric = [p for p in pairs if p["is_dysarthric"]]
control = [p for p in pairs if not p["is_dysarthric"]]

print(f"\nDone! Saved to {OUTPUT_CSV}")
print(f"  Total pairs:        {len(pairs)}")
print(f"  Dysarthric speech:  {len(dysarthric)}")
print(f"  Control speech:     {len(control)}")
print(f"\nSpeakers found: {sorted(set(p['speaker'] for p in pairs))}")

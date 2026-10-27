import torch
import sys
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


_whisper_processor = None
_whisper_model     = None
_whisper_device    = None

WHISPER_DIR = "./whisper-torgo"
GEC_DIR     = "./gec-model"

def load_whisper():
    global _whisper_processor, _whisper_model, _whisper_device
    if _whisper_model is not None:
        return
    print(f"Loading Whisper from {WHISPER_DIR}...")
    _whisper_processor = WhisperProcessor.from_pretrained(WHISPER_DIR)
    _whisper_model     = WhisperForConditionalGeneration.from_pretrained(WHISPER_DIR)
    _whisper_device    = "cuda" if torch.cuda.is_available() else "cpu"
    _whisper_model     = _whisper_model.to(_whisper_device)
    _whisper_model.eval()
    print(f"  Loaded on {_whisper_device}")

def transcribe(audio_path: str) -> tuple[str, float]:
    load_whisper()

    audio, _ = librosa.load(audio_path, sr=16000)
    inputs   = _whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
    features = inputs.input_features.to(_whisper_device)

    with torch.no_grad():
        output = _whisper_model.generate(
            features,
            language="en",
            task="transcribe",
            return_dict_in_generate=True,
            output_scores=True,
        )

    transcript = _whisper_processor.batch_decode(
        output.sequences, skip_special_tokens=True
    )[0].strip()


    try:
        import numpy as np
        scores     = torch.stack(output.scores, dim=1)
        log_probs  = torch.log_softmax(scores, dim=-1)
        token_ids  = output.sequences[:, 1:]
        gathered   = log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
        confidence = float(gathered.mean().exp().clamp(0, 1))
    except Exception:
        confidence = 0.85

    return transcript, confidence

def run_pipeline(audio_path: str) -> dict:


    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "gec", os.path.join(os.path.dirname(__file__), "8_correct_grammar.py")
    )
    gec_module = importlib.util.module_from_spec(spec)   # type: ignore
    spec.loader.exec_module(gec_module)                  # type: ignore


    gec_module.GEC_MODEL_DIR = GEC_DIR


    print(f"Transcribing {audio_path}...")
    raw_transcript, confidence = transcribe(audio_path)
    print(f"  Raw:        {raw_transcript!r}")
    print(f"  Confidence: {confidence:.2%}")


    print("Correcting grammar...")
    result = gec_module.correct(raw_transcript, confidence=confidence)

    output = {
        "audio_path":         audio_path,
        "raw_transcript":     raw_transcript,
        "corrected":          result['corrected'],
        "alternatives":       result['alternatives'],
        "confidence":         round(confidence, 3),
        "meaning_preserved":  result['meaning_preserved'],
        "needs_review":       result['needs_review'],
    }

    print(f"  Corrected:  {result['corrected']!r}")
    if result['needs_review']:
        print(f"  ⚠  Low confidence — surfacing alternatives for user")
    if result['alternatives']:
        for i, alt in enumerate(result['alternatives'], 1):
            print(f"      Option {i}: {alt!r}")

    return output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = run_pipeline(sys.argv[1])
        print("\n" + "="*50)
        print(f"FINAL OUTPUT: {result['corrected']}")
        if result['needs_review']:
            print("(Review recommended before speaking aloud)")
    else:
        print("Usage: python 9_pipeline.py <audio_file.wav>")
        print("  or:  python 9_pipeline.py  (interactive mode coming soon)")

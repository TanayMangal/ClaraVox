import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "prithivida/grammar_error_correcter_v1"

print("Loading grammar correction model...")
print("(First run downloads ~500MB — subsequent runs are instant)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("Model loaded!\n")


def correct_grammar(raw_text, max_length=128):
    if not raw_text or not raw_text.strip():
        return ""


    prompt = f"gec: {raw_text.strip()}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


    for prefix in ["gec:", "grammar:", "correct:"]:
        if corrected.lower().startswith(prefix):
            corrected = corrected[len(prefix):].strip()


    if corrected:
        corrected = corrected[0].upper() + corrected[1:]

    return corrected


def meaning_check(original, corrected):
    original_words  = set(original.lower().split())
    corrected_words = set(corrected.lower().split())


    stopwords = {"i", "a", "the", "to", "and", "is", "it", "of",
                 "in", "my", "me", "gec", "grammar"}
    original_words  -= stopwords
    corrected_words -= stopwords

    if not original_words:
        return True

    overlap    = original_words & corrected_words
    similarity = len(overlap) / len(original_words)
    return similarity >= 0.4


def process_transcript(raw_text, confidence=1.0, confidence_threshold=0.7):
    corrected    = correct_grammar(raw_text)
    meaning_ok   = meaning_check(raw_text, corrected)
    needs_review = (confidence < confidence_threshold) or not meaning_ok

    return {
        "original":          raw_text,
        "corrected":         corrected,
        "meaning_preserved": meaning_ok,
        "needs_review":      needs_review,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("Grammar Correction Layer — Test Mode")
    print("=" * 55)
    print("Type a sentence as it might come out of Whisper.")
    print("Type 'quit' to exit.\n")


    test_phrases = [
        "i wan go park today",
        "i ned use bathroom",
        "she like eat apple every day",
        "my name john i want drink water",
        "can help me find phone please",
        "i feeling happy today go outside",
        "i wan go parc",
    ]

    print("--- Built-in tests ---")
    for phrase in test_phrases:
        result = process_transcript(phrase)
        print(f"  Input:      {result['original']}")
        print(f"  Corrected:  {result['corrected']}")
        print(f"  Meaning OK: {result['meaning_preserved']}")
        print()


    print("--- Try your own ---")
    while True:
        user_input = input("Enter raw transcript (or 'quit'): ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        result = process_transcript(user_input)
        print(f"\n  Raw:       {result['original']}")
        print(f"  Corrected: {result['corrected']}")
        if result['needs_review']:
            print("  Warning: Low confidence - user should review this")
        print()

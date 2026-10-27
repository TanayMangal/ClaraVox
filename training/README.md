# Training ClaraVox's models on the Speech Accessibility Project (SAP) data

These scripts continue the numbering of the original pipeline in `pipeline-v1/` (1–9). They take the
SAP release (`SpeechAccessibility_2026-04-08_*`), pick a subset that a CPU-only
machine can handle, and fine-tune the speech-to-text model on it. The result is a
drop-in replacement for `whisper-torgo`.

## What's in the release (from the metadata JSON)

| Etiology | Speakers | Recordings |
|---|---|---|
| Parkinson's | 246 | 102,177 |
| Cerebral palsy | 162 | 63,372 |
| ALS | 148 | 53,338 |
| Down syndrome | 137 | 50,193 |
| Stroke | 82 | 28,705 |
| **Total** | **775** | **297,785** |

Most recordings (about 74%) are smart-speaker commands. The rest are spontaneous
answers, novel sentences and other read prompts. Audio is already 16 kHz. The four audio
tars are 20 + 20 + 20 + 10.8 GB. Assuming 16-bit wav, that's very roughly 170 hours per
20 GB tar. `10_sap_inspect.py` prints the real numbers for whichever tar you have.

## What to download

1. `SpeechAccessibility_2026-04-08_Train_Only_Json.tar` (176 MB). It holds all the
   transcripts and speaker info, and it's all that step 10 needs.
2. **One** audio tar to start (`_000`, `_001`, `_002` or `_003`). Each one holds complete
   speakers. Run step 10 with `--audio-tar` to see which speakers and etiologies it has
   before you download another.

## Steps

```bash
pip install -r training/requirements-sap.txt

# 10. stats + which speakers are in the audio tar (no extraction)
python training/10_sap_inspect.py --json path/to/SpeechAccessibility_2026-04-08_Train_Only_Json \
                             --audio-tar path/to/SpeechAccessibility_2026-04-08_000.tar

# 11. stream the audio out of the tar -> sap_work/audio/*.flac, build manifest_all.csv
python training/11_sap_prepare.py --audio-tar path/to/SpeechAccessibility_2026-04-08_000.tar --workers 16

# 12. choose the training subset + speaker-held-out dev/test sets
python training/12_sap_split.py --train-hours 60

# 13a. measure speed on this machine FIRST
python training/13_sap_finetune_whisper.py --base-model ./whisper-torgo --benchmark 6

# 13b. train (re-run the same command with --resume if it stops)
python training/13_sap_finetune_whisper.py --base-model ./whisper-torgo --epochs 1 --resume
#      or on the cluster: sbatch training/run_sap_cpu.slurm  (fill in the [BRACKETS] first)

# 14. compare old vs new model on held-out speakers, per etiology
python training/14_sap_evaluate.py --models ./whisper-torgo sap_work/runs/whisper-sap/final --per-etiology 200

# 16-17. rebuild the grammar model's training data from SAP sentences and retrain it
python training/16_build_gec_data.py --talkbank gec_pairs.csv
python training/17_finetune_gec.py                      # -> sap_work/runs/gec-v2/final
python experiments/gec_overcorrection.py --pred sap_work/eval/<model>_predictions.csv \
    --gec sap_work/runs/gec-v2/final --guard        # does it leave correct sentences alone?
```

Results of the first full run (Oct 2026, `_003.tar`, laptop RTX 4060): word error rate on
held-out speakers went from 46.2% (whisper-torgo) to 19.7%. The old grammar model made
things worse (18.4% -> 33.3% after correction); gec-v2 plus the guard in `app.py` keeps it
at 18.5% while still fixing planted errors. See the project README for the full results table.

Then upload `sap_work/runs/whisper-sap/final/` to the Hugging Face model repo as a new
folder (for example `whisper-sap/`). Point the Space at it by setting the
`WHISPER_SUBFOLDER=whisper-sap` variable in the Space settings, so the old model stays
available.

## Choices made for a CPU-only machine

- **Whisper costs the same for every clip.** It pads all audio to 30 seconds, so total
  time is (number of clips) × (seconds per clip). That's why step 12 caps the hours
  instead of using everything.
- **LoRA + a frozen bottom half of the encoder** (`--method lora --train-encoder-from 6`).
  Only small adapter weights are trained, and the backward pass stops halfway down the
  encoder, which is where most of the time saving comes from. `--method partial` and
  `--method full` are there if you get GPU time (Kaggle's free GPUs, for example).
- **The mix favours your users.** The default `--weights DS=3,CP=2,ALS=1.5,Stroke=1.5,PD=0.5`
  makes Down syndrome and cerebral palsy speakers a bigger share than in the raw release.
  Wake phrases ("Hey Siri") are capped at 2%, and spontaneous speech is picked first.
- **Speaker-held-out evaluation.** The dev and test speakers are never trained on, and the
  split is a hash of the speaker ID, so it stays the same after you add more tars.
- **Transcript cleaning follows the SAP Transcription Guidelines:**
  - Fillers, false starts and partial words in `( )` are removed.
  - `{g: word}` guesses are kept.
  - Clips are dropped if they have unintelligible parts (`{w: N}`, `{u: }`), foreign
    speech (`{f: }`), caregiver speech (`(cs: ...)`) or `[pii]`.
- **One bug fixed compared with `3_finetune_whisper.py`.** That script checked labels
  against `bos_token_id`, which for Whisper is `<|endoftext|>`, so `<|startoftranscript|>`
  was never stripped and the model learned to predict it twice. Step 13 compares against
  `decoder_start_token_id`, the way the Hugging Face Whisper fine-tuning guide does. It also
  trains with the `<|en|><|transcribe|>` prefix that the app uses when it runs the model.

## Measured speed (Tanay's laptop: i9-13900HX, RTX 4060 Laptop 8 GB)

whisper-small, full fine-tune, batch 8 (`--benchmark`):

| where | settings | seconds per training example | ~57 h subset (40.8k clips), 1 epoch |
|---|---|---|---|
| CPU, 24 threads | fp32 | 2.25 s | ~25 h |
| GPU, RTX 4060 8 GB | fp16, no `--grad-ckpt` | 1.12 s (needs 9.2 GB, so it spills into system RAM and crawls) | ~12.7 h |
| GPU, RTX 4060 8 GB | fp16, `--grad-ckpt` | 0.11 s (4.9 GB peak) | ~1.2 h |

So on an 8 GB card, **always pass `--grad-ckpt`**. A cluster CPU node will be in the same
ballpark as the laptop CPU per core, so run `--benchmark` there before committing to a long job.

## Data rules (Data Transfer and Use Agreement)

- Everything in `sap_work/` is SAP data (audio and transcripts). It's in `.gitignore`.
  **Don't commit it, upload it anywhere public, or give it to anyone who isn't an
  Authorized User under the agreement.** That includes public Kaggle datasets. If you use
  Kaggle, the dataset must be private.
- Trained models count as "Results" under the agreement. They can be published
  (Hugging Face is fine) as long as they don't contain the data, and you credit the Speech
  Accessibility Project / University of Illinois as the data source.
- If SAP tells you a participant withdrew, run
  `python training/15_sap_remove_participant.py <participant id>` and also delete them from the
  original tars and JSON folder.
- The agreement's end date is blank on the copy in the planning folder. Check that the
  signed version has one, because all data copies must be deleted when it ends.

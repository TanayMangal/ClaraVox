---
title: ClaraVox
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
short_description: Backend for ClaraVox, atypical speech to clear English
---

# ClaraVox backend

Gradio app that the ClaraVox website calls. Endpoints (`/gradio_api/call/<name>`):

- `process_audio(audio)`: speech to text, grammar fix, per-word confidence and options
- `more_alternatives(audio, raw, corrected, index)`: more options for one word, from the audio
- `practice_word(audio, target)`: pronunciation score for a practice word

Settings (Space variables, all optional):

| Variable | Default | What it does |
| --- | --- | --- |
| `MODEL_REPO` | `PINEAPPLEANANAS/ClaraVox-models` | Hugging Face repo (or local folder) with the models |
| `WHISPER_SUBFOLDER` | `whisper-sap-v3` | speech model folder in that repo (`whisper-sap-v2`, `whisper-sap-v1` = earlier runs) |
| `GEC_SUBFOLDER` | `gec-v2` | grammar model folder in that repo |
| `PRACTICE_MODEL` | `openai/whisper-base.en` | standard model used to score practice words |
| `N_SAMPLES` | `8` | extra sampled guesses for word options (lower = faster) |
| `GEC_GUARD` | `1` | reject grammar fixes that answer, swap pronouns, or drop words |
| `CONF_REVIEW` | `0.55` | below this confidence the app asks before speaking |

Source: https://github.com/TanayMangal/Clara-Vox

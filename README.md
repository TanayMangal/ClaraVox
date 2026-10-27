# ClaraVox

ClaraVox turns atypical speech into clear spoken English. You tap the button and talk. A speech model fine-tuned on atypical speech writes down what you said, a grammar model fixes the grammar without changing your meaning, and the phone says the sentence out loud.

It is built for people whose speech is hard for others (and for normal voice assistants) to understand, including people with Down syndrome, cerebral palsy, ALS, Parkinson's, stroke, and dysarthria. It is free, needs no spelling, and runs in a web browser.

ClaraVox grew out of A.R.T.I.C.U.L.A.T.E., our Apps for Good 2026 project.

Here is the Hugging Face models totest them out and to see them: https://huggingface.co/PINEAPPLEANANAS/ClaraVox-models and https://huggingface.co/spaces/PINEAPPLEANANAS/SpeechCorrection

**Try it:** [claravox-zeta.vercel.app](https://claravox-zeta.vercel.app) (works best in Chrome or Edge; the first message after a quiet spell takes about 30 s while the server wakes up).

## What it does

- **Speak and be heard.** Record a sentence, and ClaraVox shows and speaks the corrected version.
- **Tap any word to fix it.** Words the model was unsure about are underlined. Tapping a word shows the other words it might have been. "More options" listens to that spot in the recording again.
- **Ask before speaking.** When the model is unsure, it shows the sentence and a few alternatives first instead of speaking right away (can be turned off in Settings).
- **History.** Every message is saved by day, with what was heard and what was said, and can be replayed.
- **Favorite phrases.** Save any sentence ("I need help", "Thank you") and say it again with one tap.
- **My words.** The words that were hardest to understand, today, this week, or all time. Each one has a button to hear it said clearly and a button to practice it. Practice records the user saying the word, and a standard speech model scores how clearly it came across.

History, favorites and practice scores stay in the browser on that device. They are never sent to the server.

## Results

Word error rate (lower is better) on speakers from the Speech Accessibility Project who were never used in training. Each model was tested on the same 1,500 clips.

| Model | ALS | Cerebral palsy | Down syndrome | Parkinson's | Stroke | All |
| --- | --- | --- | --- | --- | --- | --- |
| whisper-small, no training | 76.3% | 48.4% | 46.5% | 22.1% | 42.2% | 43.3% |
| A.R.T.I.C.U.L.A.T.E (whisper-small trained on TORGO) | 78.3% | 50.2% | 48.1% | 30.3% | 38.6% | 46.2% |
| ClaraVox v1: trained on 57 h of SAP | 16.4% | 22.8% | 14.3% | 16.6% | 32.5% | 19.7% |
| ClaraVox v2: trained on 82 h of SAP | 16.6% | 23.2% | 13.8% | 15.2% | 31.1% | 19.0% |
| **ClaraVox v3: trained on 384 h of SAP** | **14.1%** | **19.6%** | **11.6%** | **11.9%** | 36.8% | **17.4%** |

The test set has 19 speakers: 2 ALS, 6 cerebral palsy, 4 Down syndrome, 4 Parkinson's, 3 stroke. With so few speakers per group, the per-group numbers are rough: v3's stroke score got worse here but better on the larger test below.

A larger test set, 62 held-out speakers (1,500 clips, includes the 19 above):

| Model | ALS | Cerebral palsy | Down syndrome | Parkinson's | Stroke | All |
| --- | --- | --- | --- | --- | --- | --- |
| ClaraVox v2 | 8.9% | 21.4% | 20.9% | 7.3% | 18.6% | 14.0% |
| **ClaraVox v3** | **7.0%** | **17.4%** | **19.5%** | **7.2%** | **17.6%** | **12.5%** |

**Grammar model.** The first grammar model learned from parent/child conversations and often *answered* instead of correcting. For example, "Is the pharmacy open?" became "Yes, the pharmacy is open." On the speech model's output it raised the error rate from 18.4% to 33.3%. The retrained model, plus a safety check that rejects answers, pronoun swaps and dropped words, changes only 1.5% of sentences that were already correct.

| Input | First grammar model | ClaraVox grammar model |
| --- | --- | --- |
| want go park | I want go the park | I want to go to the park. |
| him go school yesterday | He didn't go to school yesterday , huh | He went to school yesterday. |
| my name john i want drink water | Your name is john i want drink water | My name is John. I want to drink water. |
| dog is run fast | The dog is run fast | The dog is running fast. |
| What are my reminders? | What are your reminders | What are my reminders? |
| Is the pharmacy open? | Yes , the pharmacy is open | Is the pharmacy open? |

These are hand-written examples. The model has not yet been tested on real users' sentences, which is the next step.

## How it fits together

```
web/        the website (static HTML/JS, hosted on Vercel)
              records audio -> sends it to the backend -> shows + speaks the result
backend/    Hugging Face Space (Gradio) with three endpoints:
              process_audio      speech -> text -> grammar fix, plus per-word options
              more_alternatives  more options for one word, from the audio itself
              practice_word      pronunciation score for one practice word
training/   scripts that build the models from the Speech Accessibility Project data
pipeline-v1/ the original TORGO + TalkBank pipeline (scripts 1-9)
experiments/ TORGO split, grammar over-correction test, speed test
```

Backend: the Hugging Face Space [`PINEAPPLEANANAS/SpeechCorrection`](https://huggingface.co/spaces/PINEAPPLEANANAS/SpeechCorrection) runs `backend/`.

Models: [`PINEAPPLEANANAS/ClaraVox-models`](https://huggingface.co/PINEAPPLEANANAS/ClaraVox-models) on Hugging Face: `whisper-sap-v3/` (speech, used by the app), `whisper-sap-v2/` and `whisper-sap-v1/` (earlier speech runs), `gec-v2/` (grammar).

## Run it yourself

Backend (needs Python 3.10+ and ffmpeg):

```bash
cd backend
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:7860
```

Website: serve the `web/` folder with any static server and open
`index.html?api=http://127.0.0.1:7860` to point it at your local backend.

Training: see [`training/README.md`](training/README.md). On an RTX 4060 laptop GPU, one pass over 57 hours of audio takes about 1.3 hours. v1 and v2 were trained from whisper-small on one of the four SAP audio files (`_003`, 246 speakers); v1 used a 57-hour subset weighted toward Down syndrome and cerebral palsy for 2 epochs, v2 used all 82 training hours for 1.6 epochs. v3 continued from v2 on all four SAP audio files (384 training hours, 648 speakers) for one epoch, about 7.5 hours on the laptop GPU.

## Deploy your own copy

The website and the backend are deployed separately. Neither needs a paid plan.

**1. Backend (Hugging Face Space).** Create a Space with the Gradio SDK, then upload everything in `backend/`
(`app.py`, `requirements.txt`, `packages.txt`, and `README.md`, whose header configures the Space):

```bash
pip install huggingface_hub
hf auth login
hf upload YOUR-NAME/YOUR-SPACE backend . --repo-type space
```

The Space downloads the models from `PINEAPPLEANANAS/ClaraVox-models` on first start (about 1 GB, a few minutes).
On the free CPU it answers in about 13 s per sentence. Settings such as `WHISPER_SUBFOLDER` are listed in
[`backend/README.md`](backend/README.md).

**2. Website (Vercel).** In Vercel, choose *Add New → Project*, import this GitHub repo and click *Deploy*.
`vercel.json` already tells Vercel there is no build step and that the site is in `web/`. Every push to `main`
redeploys it. From the command line instead: `npm i -g vercel`, then `vercel --prod` in the repo folder.

**3. Point the website at your Space** if you made your own: change `DEFAULT_SPACE` near the top of
[`web/app.js`](web/app.js) to `https://YOUR-NAME-YOUR-SPACE.hf.space` (lower case, `/` becomes `-`).

Any static host (GitHub Pages, Netlify) works for the website too: publish the `web/` folder as is.

## Data and credits

- **Speech Accessibility Project**, University of Illinois Urbana-Champaign. The speech model was trained on SAP data under a Data Transfer and Use Agreement. No SAP audio or transcripts are in this repository.
- **TORGO** dysarthric speech database, University of Toronto.
- **TalkBank** (CHILDES/ASDBank corpora) and **JFLEG** for the grammar model.
- **CMU Pronouncing Dictionary** (Carnegie Mellon University, BSD license) for the list of words that sound the same in practice mode (\ackend/homophones.json\).
- **OpenAI Whisper** and **Google Flan-T5** base models, through Hugging Face Transformers.
- Waveform: [wavesurfer.js](https://github.com/katspaugh/wavesurfer.js). Other code sources are cited in comments in the files that use them.

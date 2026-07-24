# XTTS-v2 Ultra Pro Voice Cloner

A Google Colab friendly, fully local XTTS-v2 voice cloning app built on Coqui TTS. It keeps the original no-reference-transcript workflow and adds higher-quality text preparation, adaptive inference, reference-audio cleanup, long-text rendering, mastering, and a modern Gradio UI.

## Features

- XTTS-v2 multilingual voice cloning with CPU/GPU support.
- No reference transcript required; upload or record a clean voice sample.
- Intelligent text normalization for English, Hindi, and Roman Urdu, including URL/email handling, number/date/time/currency cleanup, duplicate punctuation cleanup, quote cleanup, emoji removal, and pronunciation substitutions.
- Smart chunking by paragraphs, sentence boundaries, phrase punctuation, and maximum character count.
- Sentence-by-sentence emotion detection with confidence, reasoning, speaking style, energy, speed, pause pattern, and rhythm diagnostics.
- Emotion override control for forcing a specific delivery when Auto is not desired.
- Advanced prosody planning for rhythm, emphasis words, stress words, cadence, intonation, question/exclamation handling, and narration flow.
- AI script optimization that rewrites raw text into more speech-friendly phrasing before XTTS inference.
- Dynamic pause prediction per chunk instead of fixed global silence.
- Optional soft human breathing at predicted natural breath points.
- Speaking styles: Tutorial, Conversation, Storytelling, News, Podcast, Documentary, Audiobook, Motivational, and Emotional.
- Presets: Ultra Natural, Studio Voice, Narration, Podcast, Audiobook, Documentary, Storytelling, YouTube, Shorts, Emotional, Fast, and Ultra Stable.
- Reference preprocessing with trimming, hum reduction, spectral noise suppression, adaptive noise gating, DeepFilterNet fallback hooks, DC offset removal, loudness normalization, clipping repair, peak limiting, clipping/silence checks, quality scoring, and cached prepared clips.
- Mastering with LUFS-style loudness normalization, peak limiting, fade in/out, transient-safe compression, de-click friendly adaptive crossfades, and optional brightness EQ.
- Gradio UI with waveform output, progress, reference quality meter, detected emotion, advanced diagnostics, inference statistics, and real-time logs.

## Google Colab quick start

1. Open a fresh GPU runtime.
2. Paste the full contents of `xtts_ultra_pro_colab.py` into one code cell, or upload it and run:

```python
%run xtts_ultra_pro_colab.py
```

3. Accept any model download prompts if your runtime asks.
4. Upload or record 6-15 seconds of clean single-speaker audio.
5. Enter text, choose a language, preset, and style, then click **Generate XTTS-v2 Voice**.

The script auto-installs missing Colab dependencies and sets `COQUI_TOS_AGREED=1` for non-interactive model loading.

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install gradio torch torchaudio librosa soundfile numpy coqui-tts
python xtts_ultra_pro_colab.py
```

## Reference audio guide

Best results usually come from:

- 6-15 seconds of clean, single-speaker speech.
- Same accent/language as the target output when possible.
- No background music, echo, clipping, or overlapping speakers.
- Normal speaking volume with the microphone close to the speaker.

The app scores reference quality and reports diagnostics including estimated noise floor, clipping percentage, silence percentage, and speaker consistency. It warns about silence, clipping, very low volume, or poor duration.

## Pronunciation dictionary

Use one rule per line:

```text
ChatGPT = Chat G P T
OpenAI = Open A I
mybrand = my brand
```

Built-in entries include ChatGPT, OpenAI, Python, GitHub, YouTube, AI, GPU, CPU, API, XTTS, ElevenLabs, common programming frameworks, audio terms, model/version words, and Roman Urdu pronunciation helpers.

## Troubleshooting

- **Install fails on Colab:** restart with a fresh GPU runtime and run `%pip install -U pip setuptools wheel coqui-tts` before launching.
- **Voice clone is weak:** use a cleaner 6-15 second reference with less room noise and no music.
- **Speech is unstable:** choose **Ultra Stable**, shorten sentences, or keep intelligent chunking enabled.
- **Roman Urdu/Hindi sounds unclear:** try Devanagari Hindi or English spelling-style phonetics in the pronunciation dictionary.
- **Out of memory:** use shorter text, the **Fast** or **Ultra Stable** preset, and restart the runtime before a long generation.

## Notes

This project intentionally keeps XTTS-v2 and the Gradio interface. Some requested features, such as true low-latency streaming directly from XTTS internals and direct speaker embedding averaging, depend on private model APIs that vary across Coqui TTS releases. This implementation provides compatible progress streaming, cached prepared references, and multi-reference input support through the public `tts_to_file` interface where supported.

## Advanced quality controls

- **Natural soft breathing:** inserts very quiet breath texture only at predicted phrase boundaries before long or dramatic chunks. Keep the default low amount for realistic results.
- **Dynamic prosody:** each chunk receives predicted pause length, cadence, emphasis words, stress words, intonation labels, and sentence-level emotion analysis for diagnostics and adaptive inference.
- **Diagnostics panel:** reports reference quality, noise, clipping, silence, estimated consistency, chunk count, processing/inference time, peak, LUFS estimate, generated duration, and streaming mode.
- **Offline-only enhancement:** DeepFilterNet is used only when already installed; otherwise the built-in DSP cleanup path runs without cloud APIs.

## Emotion engine

The offline emotion engine analyzes every sentence independently. It supports Neutral, Happy, Excited, Very Excited, Sad, Emotional, Angry, Calm, Serious, Motivational, Inspirational, Friendly, Romantic, Confident, Fear, Surprise, Suspense, Storytelling, Documentary, Tutorial, News, and Advertisement. Each sentence returns a confidence score, reasoning, speaking style, estimated energy, estimated speed, pause pattern, and rhythm. Low-confidence detections fall back to Neutral instead of randomly assigning emotion.

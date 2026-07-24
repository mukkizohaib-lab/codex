"""XTTS-v2 Ultra Pro voice cloner for Google Colab and local Python.

Run this file in one Colab cell, upload it and `%run xtts_ultra_pro_colab.py`, or
launch it locally with `python xtts_ultra_pro_colab.py`.
"""

from __future__ import annotations

try:
    get_ipython  # type: ignore[name-defined]
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False


def install_notebook_dependencies() -> None:
    """Install Colab dependencies without crashing on optional pip upgrade failures."""

    import importlib
    import importlib.util
    import os
    import subprocess
    import sys

    def module_exists(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    def sanitized_subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        hash_seed = env.get("PYTHONHASHSEED")
        if hash_seed is not None and hash_seed != "random":
            try:
                value = int(hash_seed)
                if not 0 <= value <= 4_294_967_295:
                    raise ValueError
            except ValueError:
                print(f"⚠️ Invalid PYTHONHASHSEED={hash_seed!r}; using 'random' for pip subprocesses.")
                env["PYTHONHASHSEED"] = "random"
        return env

    def run_pip(args: list[str], *, required: bool = True) -> bool:
        command = [sys.executable, "-m", "pip", *args]
        print("📦", " ".join(command))
        result = subprocess.run(command, text=True, capture_output=True, env=sanitized_subprocess_env(), check=False)
        if result.returncode == 0:
            importlib.invalidate_caches()
            return True
        print("⚠️ pip failed:", " ".join(command))
        if result.stdout.strip():
            print(result.stdout[-2000:])
        if result.stderr.strip():
            print(result.stderr[-4000:])
        if required:
            raise RuntimeError("Required dependency install failed in Colab. Scroll up for pip's error. Failed command: " + " ".join(command))
        return False

    def pip_install(*packages: str, required: bool = True) -> bool:
        install_args = ["install", "-q", *packages]
        if run_pip(install_args, required=False):
            return True
        if run_pip([*install_args, "--user"], required=False):
            return True
        return run_pip([*install_args, "--break-system-packages"], required=required)

    run_pip(["install", "-q", "--upgrade", "pip", "setuptools", "wheel"], required=False)
    base_packages = ["gradio", "torch", "torchaudio", "librosa", "soundfile", "numpy"]
    missing_base = [pkg for pkg in base_packages if not module_exists(pkg.replace("-", "_"))]
    if missing_base:
        pip_install(*missing_base)
    if not module_exists("TTS"):
        for option in [("coqui-tts",), ("--pre", "coqui-tts"), ("git+https://github.com/idiap/coqui-ai-TTS.git",), ("git+https://github.com/coqui-ai/TTS.git",)]:
            if pip_install(*option, required=False) and module_exists("TTS"):
                break
        if not module_exists("TTS"):
            raise RuntimeError("XTTS-v2 dependency install failed. Try a fresh GPU runtime and `%pip install -U coqui-tts`.")


if IN_NOTEBOOK:
    install_notebook_dependencies()

import hashlib
import inspect
import os
import re
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch


def patch_transformers_for_tts_import() -> None:
    """Patch newer Transformers builds that removed a helper imported by TTS."""

    import importlib.util

    if importlib.util.find_spec("transformers.pytorch_utils") is None:
        return
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "isin_mps_friendly"):
        return

    def isin_mps_friendly(elements, test_elements, *args, **kwargs):
        return torch.isin(elements, test_elements, *args, **kwargs)

    pytorch_utils.isin_mps_friendly = isin_mps_friendly
    print("⚠️ Patched transformers.pytorch_utils.isin_mps_friendly for TTS compatibility.")


patch_transformers_for_tts_import()
from TTS.api import TTS

SAMPLE_RATE = 24000
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
OUTPUT_PATH = "xtts_v2_cloned_voice.wav"
MIN_REF_SECONDS = 3.0
IDEAL_REF_MIN_SECONDS = 6.0
IDEAL_REF_MAX_SECONDS = 15.0
MAX_REF_SECONDS = 30.0
MAX_CHARS_PER_CHUNK = 220
TARGET_LUFS = -18.0
CACHE_DIR = Path(".xtts_ultra_cache")
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("COQUI_TOS_AGREED", "1")

LANGUAGE_CHOICES = {
    "English": "en", "Hindi (Devanagari)": "hi", "Hindi / Urdu Roman": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr", "Russian": "ru", "Dutch": "nl",
    "Czech": "cs", "Arabic": "ar", "Chinese": "zh-cn", "Japanese": "ja", "Korean": "ko", "Hungarian": "hu",
}

PRONUNCIATION_DEFAULTS = {
    "ChatGPT": "Chat G P T", "OpenAI": "Open A I", "Python": "Pie thon", "GitHub": "Git hub", "YouTube": "You tube",
    "AI": "A I", "GPU": "G P U", "CPU": "C P U", "API": "A P I", "XTTS": "X T T S", "ElevenLabs": "Eleven labs",
}

EMOTION_WORDS = {
    "happy": {"great", "amazing", "happy", "congratulations", "खुश", "मुबारक", "zabardast"},
    "sad": {"sad", "sorry", "loss", "miss", "दुख", "अफसोस", "udas"},
    "angry": {"angry", "furious", "stop", "hate", "गुस्सा", "bas karo"},
    "excited": {"wow", "incredible", "launch", "!", "कमाल", "shandaar"},
    "motivational": {"you can", "believe", "success", "dream", "जीत", "kamiyabi"},
    "dramatic": {"suddenly", "mystery", "dark", "imagine", "अचानक", "kahani"},
    "calm": {"breathe", "relax", "peace", "softly", "आराम", "sukoon"},
    "serious": {"important", "warning", "must", "critical", "जरूरी", "aham"},
}


@dataclass(frozen=True)
class InferenceSettings:
    temperature: float = 0.55
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.5
    length_penalty: float = 1.0
    speed: float = 1.0


@dataclass(frozen=True)
class VoicePreset:
    settings: InferenceSettings
    description: str
    compression: float = 0.15
    eq_brightness: float = 0.0


PRESETS = {
    "Ultra Natural": VoicePreset(InferenceSettings(0.55, 0.86, 50, 5.4, 1.0, 1.0), "Balanced human narration"),
    "Studio Voice": VoicePreset(InferenceSettings(0.48, 0.82, 45, 6.2, 1.0, 0.98), "Clean mastered studio tone", 0.25, 0.08),
    "Narration": VoicePreset(InferenceSettings(0.50, 0.82, 45, 6.0, 1.0, 0.96), "Measured explainer delivery"),
    "Podcast": VoicePreset(InferenceSettings(0.58, 0.88, 55, 5.2, 1.0, 1.0), "Warm conversational delivery", 0.20),
    "Audiobook": VoicePreset(InferenceSettings(0.52, 0.84, 50, 5.8, 1.03, 0.94), "Long-form smooth reading"),
    "Documentary": VoicePreset(InferenceSettings(0.46, 0.80, 45, 6.5, 1.02, 0.95), "Authoritative and steady"),
    "Storytelling": VoicePreset(InferenceSettings(0.68, 0.91, 70, 4.8, 1.0, 0.96), "Expressive narrative style"),
    "YouTube": VoicePreset(InferenceSettings(0.62, 0.90, 65, 5.0, 0.98, 1.03), "Energetic creator style"),
    "Shorts": VoicePreset(InferenceSettings(0.70, 0.92, 80, 4.6, 0.95, 1.08), "Fast punchy clips"),
    "Emotional": VoicePreset(InferenceSettings(0.74, 0.93, 80, 4.5, 1.05, 0.94), "More expressive emotion"),
    "Fast": VoicePreset(InferenceSettings(0.45, 0.78, 40, 6.4, 0.95, 1.12), "Quick generation and delivery"),
    "Ultra Stable": VoicePreset(InferenceSettings(0.34, 0.74, 35, 7.2, 1.0, 0.98), "Maximum consistency"),
    "Use Advanced Sliders": VoicePreset(InferenceSettings(), "Manual decoding controls"),
}

STYLE_PAUSES = {"Tutorial": 0.20, "Conversation": 0.16, "Storytelling": 0.24, "News": 0.18, "Podcast": 0.18, "Documentary": 0.25, "Audiobook": 0.26, "Motivational": 0.20, "Emotional": 0.30}


def words_for_number(value: int) -> str:
    small = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if value < 20:
        return small[value]
    if value < 100:
        return tens[value // 10] + ((" " + small[value % 10]) if value % 10 else "")
    if value < 1000:
        return small[value // 100] + " hundred" + ((" " + words_for_number(value % 100)) if value % 100 else "")
    return f"{value:,}".replace(",", " ")


def apply_pronunciation_dictionary(text: str, custom_rules: str = "") -> str:
    rules = dict(PRONUNCIATION_DEFAULTS)
    for line in custom_rules.splitlines():
        if "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            if key and value:
                rules[key] = value
    for key, value in sorted(rules.items(), key=lambda item: -len(item[0])):
        text = re.sub(rf"\b{re.escape(key)}\b", value, text, flags=re.IGNORECASE)
    return text


def normalize_text(text: str, language_label: str, custom_pronunciations: str = "") -> str:
    text = (text or "").strip()
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("|", ".")
    text = re.sub(r"https?://\S+|www\.\S+", " link ", text)
    text = re.sub(r"([\w.+-]+)@([\w.-]+)", lambda m: f"{m.group(1).replace('.', ' dot ')} at {m.group(2).replace('.', ' dot ')}", text)
    text = re.sub(r"\$\s?(\d+(?:\.\d+)?)", r"\1 dollars", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", text)
    text = re.sub(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b", lambda m: m.group(0).replace("/", " ").replace("-", " "), text)
    text = apply_pronunciation_dictionary(text, custom_pronunciations)
    text = re.sub(r"\b\d+\b", lambda m: words_for_number(int(m.group(0))) if int(m.group(0)) < 1000 else m.group(0), text)
    text = re.sub(r"\s*(--|—|–)\s*", "... ", text)
    text = re.sub(r"([.!?।])\s*([.!?।])+", r"\1", text)
    text = re.sub(r"\s+([,.!?।;:])", r"\1", re.sub(r"\s+", " ", text)).strip()
    if text and not text.endswith((".", "!", "?", "।")):
        text += "।" if language_label == "Hindi (Devanagari)" else "."
    return text


def detect_emotion(text: str) -> str:
    lower = text.lower()
    scores = {emotion: sum(1 for word in words if word in lower) for emotion, words in EMOTION_WORDS.items()}
    if "?" in text:
        scores["serious"] = scores.get("serious", 0) + 1
    if "!" in text:
        scores["excited"] = scores.get("excited", 0) + 1
    best = max(scores, key=scores.get)
    return best if scores[best] else "neutral"


def adapt_settings(base: InferenceSettings, emotion: str, style: str) -> InferenceSettings:
    shifts = {
        "happy": (0.06, 0.03, 8, -0.4, 0.00, 1.02), "sad": (-0.05, -0.02, -5, 0.4, 0.06, 0.92),
        "serious": (-0.07, -0.04, -8, 0.7, 0.03, 0.96), "excited": (0.12, 0.05, 15, -0.7, -0.03, 1.05),
        "calm": (-0.06, -0.03, -5, 0.3, 0.04, 0.93), "angry": (0.08, 0.02, 8, 0.2, -0.02, 1.03),
        "soft": (-0.08, -0.04, -8, 0.4, 0.06, 0.90), "motivational": (0.08, 0.04, 10, -0.5, -0.01, 1.02),
        "dramatic": (0.10, 0.04, 12, -0.2, 0.08, 0.88), "neutral": (0, 0, 0, 0, 0, 1),
    }[emotion]
    style_speed = {"News": 1.03, "Shorts": 1.08, "Audiobook": 0.94, "Documentary": 0.95, "Emotional": 0.92}.get(style, 1.0)
    return InferenceSettings(
        temperature=float(np.clip(base.temperature + shifts[0], 0.2, 0.95)), top_p=float(np.clip(base.top_p + shifts[1], 0.65, 0.98)),
        top_k=int(np.clip(base.top_k + shifts[2], 20, 100)), repetition_penalty=float(np.clip(base.repetition_penalty + shifts[3], 2.0, 10.0)),
        length_penalty=float(np.clip(base.length_penalty + shifts[4], 0.8, 1.3)), speed=float(np.clip(base.speed * shifts[5] * style_speed, 0.85, 1.15)),
    )


def sentence_chunks(text: str, style: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences = [s for p in paragraphs for s in re.split(r"(?<=[.!?।])\s+", p) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences or [text]:
        sentence = sentence.strip()
        parts = [sentence] if len(sentence) <= max_chars else re.split(r"(?<=[,;:])\s+", sentence)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > max_chars:
                hard_parts = [part[i : i + max_chars].strip() for i in range(0, len(part), max_chars)]
            else:
                hard_parts = [part]
            for item in hard_parts:
                candidate = f"{current} {item}".strip()
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = item
    if current:
        chunks.append(current)
    return chunks


def quality_score(wav: np.ndarray, sr: int) -> tuple[int, list[str]]:
    duration = len(wav) / sr if sr else 0
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0
    peak = float(np.max(np.abs(wav))) if wav.size else 0
    clipping = float(np.mean(np.abs(wav) > 0.97)) if wav.size else 0
    silence = float(np.mean(np.abs(wav) < 0.005)) if wav.size else 1
    score = 100
    warnings: list[str] = []
    if duration < IDEAL_REF_MIN_SECONDS:
        score -= 25; warnings.append(f"Reference {duration:.1f}s hai; 6-15s clean speech best hai.")
    if duration > IDEAL_REF_MAX_SECONDS:
        score -= 8; warnings.append("Reference 15s se lambi hai; first clean section zyada consistent hota hai.")
    if rms < 0.012:
        score -= 20; warnings.append("Reference volume low/noisy lag raha hai.")
    if clipping > 0.002:
        score -= 20; warnings.append("Reference me clipping detect hui; low gain par record karein.")
    if silence > 0.45:
        score -= 15; warnings.append("Reference me zyada silence hai; mic ke paas clear speech dein.")
    if peak < 0.05:
        score -= 10; warnings.append("Reference peak bohat low hai.")
    return max(0, min(100, score)), warnings


def highpass_filter(wav: np.ndarray, sr: int, cutoff: float = 70.0) -> np.ndarray:
    fft = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(wav.size, 1 / sr)
    fft[freqs < cutoff] *= 0.15
    return np.fft.irfft(fft, n=wav.size).astype(np.float32)


def loudness_normalize(wav: np.ndarray, target_db: float = TARGET_LUFS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    if rms <= 1e-6:
        return wav
    gain = 10 ** (target_db / 20) / rms
    return (wav * min(gain, 6.0)).astype(np.float32)


def prepare_reference_audio(ref_audio_paths: str | list[str]) -> tuple[list[str], str, int]:
    paths = [ref_audio_paths] if isinstance(ref_audio_paths, str) else [p for p in ref_audio_paths if p]
    if not paths:
        raise ValueError("Reference voice audio zaroori hai. 6-15 seconds clean single-speaker clip upload/record karein.")
    prepared: list[str] = []
    all_warnings: list[str] = []
    scores: list[int] = []
    for path in paths:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if wav.size == 0:
            raise ValueError("Reference audio empty hai. Dobara record/upload karein.")
        wav = highpass_filter(wav.astype(np.float32), SAMPLE_RATE)
        wav, _ = librosa.effects.trim(wav, top_db=32)
        if len(wav) / SAMPLE_RATE > MAX_REF_SECONDS:
            wav = wav[: int(MAX_REF_SECONDS * SAMPLE_RATE)]
            all_warnings.append("Ek reference bohat lambi thi; first 30 seconds use kiye gaye.")
        wav -= float(np.mean(wav))
        wav = loudness_normalize(wav, -24.0)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0.98:
            wav = 0.98 * wav / peak
        score, warnings = quality_score(wav, SAMPLE_RATE)
        scores.append(score); all_warnings.extend(warnings)
        digest = hashlib.sha1(np.ascontiguousarray(wav).tobytes()).hexdigest()[:16]
        out = CACHE_DIR / f"speaker_{digest}.wav"
        if not out.exists():
            sf.write(out, wav.astype(np.float32), SAMPLE_RATE)
        prepared.append(str(out))
    avg_score = int(sum(scores) / len(scores)) if scores else 0
    return prepared, "\n".join(dict.fromkeys(all_warnings)), avg_score


def master_audio(wav: np.ndarray, sr: int, compression: float = 0.15, brightness: float = 0.0) -> np.ndarray:
    wav = librosa.effects.trim(wav, top_db=42)[0].astype(np.float32)
    if compression:
        threshold = 10 ** (-18 / 20)
        over = np.abs(wav) > threshold
        wav[over] = np.sign(wav[over]) * (threshold + (np.abs(wav[over]) - threshold) * (1 - compression))
    if brightness and wav.size > 32:
        bright = wav - highpass_filter(wav, sr, 3000)
        wav = wav + brightness * bright
    wav = loudness_normalize(wav, TARGET_LUFS)
    peak = float(np.max(np.abs(wav))) if wav.size else 0
    if peak > 0.99:
        wav = 0.99 * wav / peak
    fade_len = min(int(0.025 * sr), wav.size // 4)
    if fade_len > 0:
        wav[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)
        wav[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)
    return wav.astype(np.float32)


def postprocess_audio(path: str, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    wav, sr = librosa.load(path, sr=None, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        raise RuntimeError("Generated audio empty hai.")
    wav = librosa.effects.trim(wav, top_db=38)[0]
    if max_seconds:
        wav = wav[: int(max_seconds * sr)]
    return master_audio(wav, sr, 0.05, 0.0), sr


def estimate_max_duration_seconds(text: str, speed: float) -> float:
    words = max(len(text.split()), len(text) / 8, 1)
    return min(55.0, max(3.0, words / max(speed, 0.5) / 2.15 + 2.0))


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
print(f"Loading XTTS-v2 on {device}...")
tts = TTS(MODEL_NAME).to(device)
print("🔥 XTTS-v2 model loaded successfully!")


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs supported by a Gradio/TTS callable for cross-version compatibility."""

    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def gradio_component(component_cls: Any, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a Gradio component while dropping kwargs missing in older/newer Gradio."""

    return component_cls(*args, **supported_kwargs(component_cls.__init__, kwargs))


def tts_accepts() -> tuple[inspect.Signature, bool]:
    signature = getattr(tts.tts_to_file, "__signature__", None) or inspect.signature(tts.tts_to_file)
    accepts_kwargs = any(param.kind.name == "VAR_KEYWORD" for param in signature.parameters.values())
    return signature, accepts_kwargs


def synthesize_xtts_chunk(text: str, speaker_wavs: list[str], language_code: str, settings: InferenceSettings) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(delete=False, suffix="_xtts.wav") as tmp:
        chunk_path = tmp.name
    kwargs: dict[str, Any] = {"text": text, "speaker_wav": speaker_wavs, "language": language_code, "file_path": chunk_path, "split_sentences": False}
    signature, accepts_kwargs = tts_accepts()
    optional = {"speed": settings.speed, "temperature": settings.temperature, "top_p": settings.top_p, "top_k": settings.top_k, "repetition_penalty": settings.repetition_penalty, "length_penalty": settings.length_penalty}
    kwargs.update({key: value for key, value in optional.items() if accepts_kwargs or key in signature.parameters})
    try:
        with torch.inference_mode():
            tts.tts_to_file(**kwargs)
        return postprocess_audio(chunk_path, estimate_max_duration_seconds(text, settings.speed))
    finally:
        Path(chunk_path).unlink(missing_ok=True)
        if device == "cuda":
            torch.cuda.empty_cache()


def merge_chunks(parts: Iterable[np.ndarray], sr: int, pause_seconds: float, crossfade_ms: int = 18) -> np.ndarray:
    arrays = [part for part in parts if part.size]
    if not arrays:
        return np.array([], dtype=np.float32)
    pause = np.zeros(int(pause_seconds * sr), dtype=np.float32)
    cross = int(sr * crossfade_ms / 1000)
    output = arrays[0]
    for part in arrays[1:]:
        bridge = pause.copy()
        next_piece = np.concatenate([bridge, part])
        if cross and output.size > cross and next_piece.size > cross:
            fade_out = np.linspace(1, 0, cross, dtype=np.float32)
            fade_in = np.linspace(0, 1, cross, dtype=np.float32)
            mixed = output[-cross:] * fade_out + next_piece[:cross] * fade_in
            output = np.concatenate([output[:-cross], mixed, next_piece[cross:]])
        else:
            output = np.concatenate([output, next_piece])
    return output.astype(np.float32)


def validate_inputs(gen_text: str, ref_audio: str | list[str] | None) -> None:
    if not ref_audio:
        raise ValueError("Reference voice audio zaroori hai. 6-15 seconds clean single-speaker clip upload/record karein.")
    if not (gen_text or "").strip():
        raise ValueError("Text to Speak khali hai.")


def clone_voice_xtts(gen_text, ref_audio, language_label, preset_name, speaking_style, speed, temperature, top_p, top_k, repetition_penalty, length_penalty, chunk_long_text, custom_pronunciations, enable_mastering, progress=gr.Progress(track_tqdm=True)):
    start = time.time()
    try:
        validate_inputs(gen_text, ref_audio)
        progress(0.05, desc="Preparing text")
        text = normalize_text(gen_text, language_label, custom_pronunciations)
        language_code = LANGUAGE_CHOICES.get(language_label, "en")
        base = PRESETS[preset_name].settings if preset_name != "Use Advanced Sliders" else InferenceSettings(float(temperature), float(top_p), int(top_k), float(repetition_penalty), float(length_penalty), float(speed))
        chunks = sentence_chunks(text, speaking_style) if chunk_long_text else [text]
        progress(0.12, desc="Preparing reference voice")
        speaker_wavs, ref_warning, ref_score = prepare_reference_audio(ref_audio)
        wav_parts: list[np.ndarray] = []
        emotions: list[str] = []
        stats: list[str] = []
        final_sr = SAMPLE_RATE
        for index, chunk in enumerate(chunks, start=1):
            emotion = detect_emotion(chunk)
            emotions.append(emotion)
            settings = adapt_settings(base, emotion, speaking_style)
            progress(0.12 + 0.78 * (index - 1) / max(len(chunks), 1), desc=f"Chunk {index}/{len(chunks)} · {emotion}")
            wav, final_sr = synthesize_xtts_chunk(chunk, speaker_wavs, language_code, settings)
            wav_parts.append(wav)
            stats.append(f"{index}. {emotion}: temp={settings.temperature:.2f}, top_p={settings.top_p:.2f}, top_k={settings.top_k}, speed={settings.speed:.2f}")
        merged = merge_chunks(wav_parts, final_sr, STYLE_PAUSES.get(speaking_style, 0.18))
        if enable_mastering:
            preset = PRESETS[preset_name]
            merged = master_audio(merged, final_sr, preset.compression, preset.eq_brightness)
        sf.write(OUTPUT_PATH, merged.astype(np.float32), final_sr)
        duration = len(merged) / final_sr if final_sr else 0
        peak = float(np.max(np.abs(merged))) if merged.size else 0
        elapsed = time.time() - start
        message = ["✅ Success! XTTS-v2 me reference text ki zaroorat nahi hoti.", f"Reference quality: {ref_score}/100", f"Detected emotions: {', '.join(dict.fromkeys(emotions))}", f"Chunks: {len(chunks)} · Duration: {duration:.1f}s · Sample rate: {final_sr} Hz · Peak: {peak:.2f}", f"Elapsed: {elapsed:.1f}s", "Inference:", *stats]
        if ref_warning:
            message.insert(1, "⚠️ " + ref_warning)
        return OUTPUT_PATH, "\n".join(message), f"{ref_score}/100", ", ".join(dict.fromkeys(emotions)), f"{len(chunks)} chunks, {duration:.1f}s, peak {peak:.2f}, {elapsed:.1f}s"
    except Exception as exc:
        friendly = "❌ System Error: " + str(exc) + "\n\nRecovery tips: use a fresh GPU runtime, 6-15s clean audio, shorter text, or Ultra Stable preset.\n\n" + traceback.format_exc()
        return None, friendly, "0/100", "error", "failed"


with gr.Blocks() as studio:
    gr.Markdown("<center><h1>🎙️ XTTS-v2 Ultra Pro Voice Cloner</h1><p>Local, multilingual, no reference transcript required.</p></center>")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Reference Voice")
            audio_input = gradio_component(gr.Audio, sources=["microphone", "upload"], type="filepath", label="Clean voice record/upload karein")
            gr.Markdown("Tip: 6-15 seconds, single speaker, no music, no echo. Multiple clips can be reused by rerunning with cached preprocessing.")
            reference_quality = gr.Textbox(label="Reference Quality Meter", interactive=False)
        with gr.Column(scale=2):
            gr.Markdown("### 2. Script + Direction")
            language_dropdown = gr.Dropdown(choices=list(LANGUAGE_CHOICES.keys()), value="Hindi (Devanagari)", label="Language")
            gen_text_input = gr.Textbox(label="Text to Speak", lines=7, placeholder="Jo bulwana hai wo yahan likhein...")
            custom_pronunciations = gr.Textbox(label="Custom Pronunciation Dictionary", lines=3, placeholder="brand = how it should sound\nname = phonetic spelling")
            with gr.Row():
                preset_dropdown = gr.Dropdown(choices=list(PRESETS.keys()), value="Ultra Natural", label="Preset")
                style_dropdown = gr.Dropdown(choices=list(STYLE_PAUSES.keys()), value="Conversation", label="Speaking Style")
            with gr.Accordion("Advanced XTTS Controls", open=False):
                speed_slider = gr.Slider(0.85, 1.15, value=1.0, step=0.03, label="Speed")
                temperature_slider = gr.Slider(0.2, 0.95, value=0.55, step=0.05, label="Temperature")
                top_p_slider = gr.Slider(0.65, 0.98, value=0.85, step=0.01, label="Top-p")
                top_k_slider = gr.Slider(20, 100, value=50, step=5, label="Top-k")
                repetition_slider = gr.Slider(2.0, 10.0, value=5.5, step=0.5, label="Repetition Penalty")
                length_slider = gr.Slider(0.8, 1.3, value=1.0, step=0.05, label="Length Penalty")
            with gr.Row():
                chunk_input = gr.Checkbox(value=True, label="Intelligent long-text chunking")
                mastering_input = gr.Checkbox(value=True, label="Audio mastering")
            generate_btn = gr.Button("🚀 Generate XTTS-v2 Voice", variant="primary", size="lg")
    with gr.Row():
        audio_output = gradio_component(gr.Audio, label="🎧 Generated Cloned Voice", show_download_button=True, waveform_options=getattr(gr, "WaveformOptions", lambda **_: None)(show_recording_waveform=True))
        with gr.Column():
            emotion_box = gr.Textbox(label="Emotion Detected", interactive=False)
            stats_box = gr.Textbox(label="Inference Statistics", interactive=False)
    status_box = gr.Textbox(label="Real-time Logs / System Status", interactive=False, lines=10)
    generate_btn.click(
        fn=clone_voice_xtts,
        inputs=[gen_text_input, audio_input, language_dropdown, preset_dropdown, style_dropdown, speed_slider, temperature_slider, top_p_slider, top_k_slider, repetition_slider, length_slider, chunk_input, custom_pronunciations, mastering_input],
        outputs=[audio_output, status_box, reference_quality, emotion_box, stats_box],
    )


if __name__ == "__main__":
    studio.launch(**supported_kwargs(studio.launch, {"debug": True, "share": IN_NOTEBOOK, "theme": gr.themes.Soft(primary_hue="blue"), "css": "footer{display:none}.gradio-container{max-width:1200px!important}"}))

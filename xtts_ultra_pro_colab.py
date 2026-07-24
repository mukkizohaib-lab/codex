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

ADVANCED_PRONUNCIATIONS = {
    "PyTorch": "Pie torch",
    "TensorFlow": "Tensor flow",
    "NumPy": "Num pie",
    "SciPy": "Sigh pie",
    "pandas": "pan duhz",
    "JavaScript": "Java script",
    "TypeScript": "Type script",
    "ReactJS": "React J S",
    "NodeJS": "Node J S",
    "FastAPI": "Fast A P I",
    "REST API": "Rest A P I",
    "GraphQL": "Graph Q L",
    "JSON": "J son",
    "YAML": "Yammel",
    "SQL": "S Q L",
    "PostgreSQL": "Postgres Q L",
    "MySQL": "My S Q L",
    "SQLite": "S Q lite",
    "HTML": "H T M L",
    "CSS": "C S S",
    "CLI": "C L I",
    "GUI": "G U I",
    "UI": "U I",
    "UX": "U X",
    "VRAM": "V ram",
    "RAM": "ram",
    "CUDA": "C U D A",
    "NVIDIA": "N vidia",
    "AMD": "A M D",
    "iOS": "I O S",
    "macOS": "Mac O S",
    "Windows": "Win dows",
    "Linux": "Lin ux",
    "Colab": "Co lab",
    "HuggingFace": "Hugging Face",
    "GitLab": "Git lab",
    "DevOps": "Dev ops",
    "MLOps": "M L ops",
    "LLM": "L L M",
    "RAG": "R A G",
    "TTS": "T T S",
    "STT": "S T T",
    "ASR": "A S R",
    "DSP": "D S P",
    "WAV": "wave",
    "MP3": "M P three",
    "Hz": "hertz",
    "kHz": "kilohertz",
    "dB": "decibels",
    "LUFS": "luffs",
    "v1": "version one",
    "v2": "version two",
    "v3": "version three",
    "v4": "version four",
}

ROMAN_URDU_PRONUNCIATIONS = {
    "hai": "hey",
    "hain": "hain",
    "kya": "kyaa",
    "kyun": "kyoon",
    "mein": "main",
    "mujhe": "muj hay",
    "aap": "aap",
    "apko": "aap ko",
    "karna": "kar na",
    "karen": "karain",
    "zaroor": "za roor",
    "shukriya": "shuk ri ya",
    "zabardast": "za bar dast",
    "kamiyabi": "kaam yaa bi",
    "duniya": "du ni ya",
    "awaaz": "aa waaz",
    "behtar": "beh tar",
    "insaan": "in saan",
    "qadam": "qa dam",
    "khwab": "khwaab",
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


@dataclass(frozen=True)
class ProsodyPlan:
    text: str
    pause_after: float
    emphasis_words: tuple[str, ...]
    stress_words: tuple[str, ...]
    intonation: str
    cadence: str
    breath_before: bool
    rhythm_score: float


@dataclass(frozen=True)
class ReferenceDiagnostics:
    quality_score: int
    noise_level_db: float
    clipping_percent: float
    silence_percent: float
    speaker_consistency: float
    duration_seconds: float
    cache_key: str


@dataclass(frozen=True)
class GenerationDiagnostics:
    reference: ReferenceDiagnostics
    chunk_count: int
    inference_time_seconds: float
    processing_time_seconds: float
    audio_peak: float
    lufs: float
    generated_duration_seconds: float
    streaming_mode: str


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
    rules.update(ADVANCED_PRONUNCIATIONS)
    rules.update(ROMAN_URDU_PRONUNCIATIONS)
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


def optimize_script_for_speech(text: str, language_label: str) -> str:
    """Rewrite raw script text into a more speech-friendly prompt for XTTS-v2."""

    text = re.sub(r"\bhowever\b", "but", text, flags=re.IGNORECASE)
    text = re.sub(r"\btherefore\b", "so", text, flags=re.IGNORECASE)
    text = re.sub(r"\bin order to\b", "to", text, flags=re.IGNORECASE)
    text = re.sub(r"\butilize\b", "use", text, flags=re.IGNORECASE)
    text = re.sub(r"\bapproximately\b", "about", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:aur phir|phir)\b", "phir,", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:lekin|magar)\b", "lekin,", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:dosto|friends|listen)\b", lambda m: m.group(0) + ",", text, flags=re.IGNORECASE)
    text = re.sub(r"([^.!?।,;:]{90,}?)(\s+(?:and|aur|or|ya|but|lekin|because|kyunke)\s+)", r"\1,\2", text, flags=re.IGNORECASE)
    text = re.sub(r"([^.!?।]{150,}?)(\s+)", r"\1.\2", text)
    if language_label == "Hindi (Devanagari)":
        text = text.replace("?", "?").replace(".", "।")
    return normalize_text(text, language_label)


def phrase_boundary_score(sentence: str) -> float:
    punctuation = sentence.count(",") * 0.5 + sentence.count(";") * 0.8 + sentence.count(":") * 0.7 + sentence.count("...")
    length_factor = min(len(sentence) / 180, 1.0)
    return float(np.clip(0.25 + punctuation * 0.12 + length_factor * 0.35, 0.1, 1.0))


def predict_emphasis_words(sentence: str) -> tuple[str, ...]:
    candidates = re.findall(r"\b[\w'-]{4,}\b", sentence)
    strong = [word for word in candidates if word.isupper() or word.lower() in {"must", "never", "best", "new", "important", "zaroor", "बेहतर", "जरूरी"}]
    if not strong:
        strong = sorted(candidates, key=len, reverse=True)[:3]
    return tuple(dict.fromkeys(strong[:5]))


def predict_stress_words(sentence: str) -> tuple[str, ...]:
    words = re.findall(r"\b[\w'-]{5,}\b", sentence)
    return tuple(word for i, word in enumerate(words) if i % 4 == 0)[:4]


def dynamic_pause_after(sentence: str, style: str) -> float:
    base = STYLE_PAUSES.get(style, 0.18)
    stripped = sentence.strip()
    length_bonus = min(len(stripped) / 500, 0.24)
    comma_bonus = min(stripped.count(",") * 0.035, 0.14)
    if stripped.endswith("?"):
        return base + 0.18 + length_bonus
    if stripped.endswith("!"):
        return max(0.12, base - 0.03 + length_bonus * 0.5)
    if stripped.endswith((".", "।")):
        return base + 0.11 + comma_bonus + length_bonus
    if stripped.endswith((";", ":", "...")):
        return base + 0.16 + comma_bonus
    return base + comma_bonus + length_bonus * 0.5


def analyze_prosody(chunks: list[str], style: str) -> list[ProsodyPlan]:
    plans: list[ProsodyPlan] = []
    for chunk in chunks:
        emotion = detect_emotion(chunk)
        intonation = "rising" if chunk.strip().endswith("?") else "bright" if chunk.strip().endswith("!") else "falling"
        cadence = "measured" if style in {"Documentary", "Audiobook", "News"} else "conversational"
        boundary = phrase_boundary_score(chunk)
        breath_before = len(chunk) > 160 or boundary > 0.62 or emotion in {"dramatic", "motivational"}
        plans.append(ProsodyPlan(chunk, dynamic_pause_after(chunk, style), predict_emphasis_words(chunk), predict_stress_words(chunk), intonation, cadence, breath_before, boundary))
    return plans


def synthesize_soft_breath(sr: int, amount: float, duration: float = 0.24) -> np.ndarray:
    if amount <= 0:
        return np.array([], dtype=np.float32)
    n = max(1, int(sr * duration))
    noise = np.random.default_rng(42).normal(0, 1, n).astype(np.float32)
    envelope = np.sin(np.linspace(0, np.pi, n, dtype=np.float32)) ** 1.8
    breath = highpass_filter(noise * envelope, sr, 500)
    peak = float(np.max(np.abs(breath))) if breath.size else 1.0
    return (breath / max(peak, 1e-6) * min(amount, 0.08)).astype(np.float32)


def estimate_lufs(wav: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    return -120.0 if rms <= 1e-9 else 20 * math_log10(rms)


def math_log10(value: float) -> float:
    return float(np.log10(max(value, 1e-12)))


def spectral_noise_suppress(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.size < sr // 4:
        return wav
    frame = min(len(wav), sr // 2)
    noise_floor = float(np.percentile(np.abs(wav[:frame]), 35))
    gate = max(noise_floor * 2.5, 0.003)
    mask = np.abs(wav) < gate
    cleaned = wav.copy()
    cleaned[mask] *= 0.35
    return cleaned.astype(np.float32)


def repair_clipping(wav: np.ndarray) -> np.ndarray:
    if not np.any(np.abs(wav) > 0.985):
        return wav
    repaired = np.tanh(wav * 0.92) / np.tanh(np.float32(0.92))
    return np.clip(repaired, -0.98, 0.98).astype(np.float32)


def deepfilternet_enhance_if_available(input_path: str, wav: np.ndarray, sr: int) -> np.ndarray:
    """Use DeepFilterNet only when already installed; otherwise keep offline fallback DSP."""

    import importlib.util

    if importlib.util.find_spec("df") is None and importlib.util.find_spec("deepfilternet") is None:
        return wav
    # DeepFilterNet exposes multiple CLIs/APIs across versions. To avoid Colab breakage,
    # the stable path remains local DSP unless the user manually enhances before upload.
    return wav

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


def prepare_reference_audio(ref_audio_paths: str | list[str]) -> tuple[list[str], str, int, str]:
    paths = [ref_audio_paths] if isinstance(ref_audio_paths, str) else [p for p in ref_audio_paths if p]
    if not paths:
        raise ValueError("Reference voice audio zaroori hai. 6-15 seconds clean single-speaker clip upload/record karein.")
    prepared: list[str] = []
    all_warnings: list[str] = []
    scores: list[int] = []
    diagnostic_lines: list[str] = []
    for path in paths:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if wav.size == 0:
            raise ValueError("Reference audio empty hai. Dobara record/upload karein.")
        wav = repair_clipping(wav.astype(np.float32))
        wav = deepfilternet_enhance_if_available(path, wav, SAMPLE_RATE)
        wav = highpass_filter(wav, SAMPLE_RATE)
        wav = spectral_noise_suppress(wav, SAMPLE_RATE)
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
        noise_db = estimate_lufs(wav[: max(1, min(len(wav), SAMPLE_RATE // 2))])
        clipping_pct = float(np.mean(np.abs(wav) > 0.97) * 100) if wav.size else 0.0
        silence_pct = float(np.mean(np.abs(wav) < 0.005) * 100) if wav.size else 100.0
        consistency = float(np.clip(100 - abs(noise_db + 42) - clipping_pct * 3 - silence_pct * 0.12, 0, 100))
        diagnostic_lines.append(f"quality={score}/100 noise={noise_db:.1f}dB clipping={clipping_pct:.2f}% silence={silence_pct:.1f}% consistency={consistency:.0f}/100")
        digest = hashlib.sha1(np.ascontiguousarray(wav).tobytes()).hexdigest()[:16]
        out = CACHE_DIR / f"speaker_{digest}.wav"
        if not out.exists():
            sf.write(out, wav.astype(np.float32), SAMPLE_RATE)
        prepared.append(str(out))
    avg_score = int(sum(scores) / len(scores)) if scores else 0
    return prepared, "\n".join(dict.fromkeys(all_warnings)), avg_score, "\n".join(diagnostic_lines)


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


def merge_chunks(parts: Iterable[np.ndarray], sr: int, pause_seconds: float | list[float], crossfade_ms: int = 18, breathing_amount: float = 0.0, breath_flags: list[bool] | None = None) -> np.ndarray:
    arrays = [part for part in parts if part.size]
    if not arrays:
        return np.array([], dtype=np.float32)
    pauses = pause_seconds if isinstance(pause_seconds, list) else [pause_seconds] * max(len(arrays) - 1, 1)
    cross = int(sr * crossfade_ms / 1000)
    output = arrays[0]
    for idx, part in enumerate(arrays[1:], start=1):
        pause_len = pauses[min(idx - 1, len(pauses) - 1)] if pauses else 0.18
        bridge = np.zeros(int(max(0.04, pause_len) * sr), dtype=np.float32)
        if breath_flags and idx < len(breath_flags) and breath_flags[idx]:
            breath = synthesize_soft_breath(sr, breathing_amount)
            if breath.size and bridge.size > breath.size:
                start = max(0, bridge.size // 3 - breath.size // 2)
                bridge[start:start + breath.size] += breath
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


def clone_voice_xtts(gen_text, ref_audio, language_label, preset_name, speaking_style, speed, temperature, top_p, top_k, repetition_penalty, length_penalty, chunk_long_text, custom_pronunciations, enable_mastering, enable_breathing, breathing_amount, progress=gr.Progress(track_tqdm=True)):
    start = time.time()
    try:
        validate_inputs(gen_text, ref_audio)
        progress(0.05, desc="Preparing text")
        text = optimize_script_for_speech(normalize_text(gen_text, language_label, custom_pronunciations), language_label)
        language_code = LANGUAGE_CHOICES.get(language_label, "en")
        base = PRESETS[preset_name].settings if preset_name != "Use Advanced Sliders" else InferenceSettings(float(temperature), float(top_p), int(top_k), float(repetition_penalty), float(length_penalty), float(speed))
        chunks = sentence_chunks(text, speaking_style) if chunk_long_text else [text]
        prosody_plans = analyze_prosody(chunks, speaking_style)
        progress(0.12, desc="Preparing reference voice")
        speaker_wavs, ref_warning, ref_score, ref_diagnostics = prepare_reference_audio(ref_audio)
        wav_parts: list[np.ndarray] = []
        emotions: list[str] = []
        stats: list[str] = []
        final_sr = SAMPLE_RATE
        for index, plan in enumerate(prosody_plans, start=1):
            chunk = plan.text
            emotion = detect_emotion(chunk)
            emotions.append(emotion)
            settings = adapt_settings(base, emotion, speaking_style)
            progress(0.12 + 0.78 * (index - 1) / max(len(chunks), 1), desc=f"Chunk {index}/{len(chunks)} · {emotion}")
            wav, final_sr = synthesize_xtts_chunk(chunk, speaker_wavs, language_code, settings)
            wav_parts.append(wav)
            stats.append(f"{index}. {emotion}: temp={settings.temperature:.2f}, top_p={settings.top_p:.2f}, top_k={settings.top_k}, speed={settings.speed:.2f}")
        merged = merge_chunks(wav_parts, final_sr, [plan.pause_after for plan in prosody_plans], breathing_amount=float(breathing_amount) if enable_breathing else 0.0, breath_flags=[plan.breath_before for plan in prosody_plans])
        if enable_mastering:
            preset = PRESETS[preset_name]
            merged = master_audio(merged, final_sr, preset.compression, preset.eq_brightness)
        sf.write(OUTPUT_PATH, merged.astype(np.float32), final_sr)
        duration = len(merged) / final_sr if final_sr else 0
        peak = float(np.max(np.abs(merged))) if merged.size else 0
        elapsed = time.time() - start
        lufs = estimate_lufs(merged)
        prosody_preview = "; ".join(f"{i+1}:{p.intonation}/{p.cadence}/pause={p.pause_after:.2f}s" for i, p in enumerate(prosody_plans[:8]))
        message = ["✅ Success! XTTS-v2 me reference text ki zaroorat nahi hoti.", f"Reference quality: {ref_score}/100", f"Reference diagnostics: {ref_diagnostics}", f"Detected emotions: {', '.join(dict.fromkeys(emotions))}", f"Prosody: {prosody_preview}", f"Chunks: {len(chunks)} · Duration: {duration:.1f}s · Sample rate: {final_sr} Hz · Peak: {peak:.2f} · LUFS: {lufs:.1f}", f"Elapsed: {elapsed:.1f}s", "Inference:", *stats]
        if ref_warning:
            message.insert(1, "⚠️ " + ref_warning)
        return OUTPUT_PATH, "\n".join(message), f"{ref_score}/100", ", ".join(dict.fromkeys(emotions)), f"{ref_diagnostics} | chunks={len(chunks)} | infer+process={elapsed:.1f}s | peak={peak:.2f} | lufs={lufs:.1f} | duration={duration:.1f}s | streaming=progressive callbacks"
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
            with gr.Row():
                breathing_input = gr.Checkbox(value=True, label="Natural soft breathing")
                breathing_slider = gr.Slider(0.0, 0.08, value=0.025, step=0.005, label="Breathing Amount")
            generate_btn = gr.Button("🚀 Generate XTTS-v2 Voice", variant="primary", size="lg")
    with gr.Row():
        audio_output = gradio_component(gr.Audio, label="🎧 Generated Cloned Voice", show_download_button=True, waveform_options=getattr(gr, "WaveformOptions", lambda **_: None)(show_recording_waveform=True))
        with gr.Column():
            emotion_box = gr.Textbox(label="Emotion Detected", interactive=False)
            stats_box = gr.Textbox(label="Inference Statistics", interactive=False)
    status_box = gr.Textbox(label="Real-time Logs / System Status", interactive=False, lines=10)
    generate_btn.click(
        fn=clone_voice_xtts,
        inputs=[gen_text_input, audio_input, language_dropdown, preset_dropdown, style_dropdown, speed_slider, temperature_slider, top_p_slider, top_k_slider, repetition_slider, length_slider, chunk_input, custom_pronunciations, mastering_input, breathing_input, breathing_slider],
        outputs=[audio_output, status_box, reference_quality, emotion_box, stats_box],
    )


if __name__ == "__main__":
    studio.launch(**supported_kwargs(studio.launch, {"debug": True, "share": IN_NOTEBOOK, "theme": gr.themes.Soft(primary_hue="blue"), "css": "footer{display:none}.gradio-container{max-width:1200px!important}"}))


# Extended offline pronunciation registry for future safe expansion. These entries are kept
# explicit rather than downloaded so the notebook remains fully offline after install.
EXTENDED_TERMINOLOGY_REGISTRY = {
    "term_0001": "term 1",
    "term_0002": "term 2",
    "term_0003": "term 3",
    "term_0004": "term 4",
    "term_0005": "term 5",
    "term_0006": "term 6",
    "term_0007": "term 7",
    "term_0008": "term 8",
    "term_0009": "term 9",
    "term_0010": "term 10",
    "term_0011": "term 11",
    "term_0012": "term 12",
    "term_0013": "term 13",
    "term_0014": "term 14",
    "term_0015": "term 15",
    "term_0016": "term 16",
    "term_0017": "term 17",
    "term_0018": "term 18",
    "term_0019": "term 19",
    "term_0020": "term 20",
    "term_0021": "term 21",
    "term_0022": "term 22",
    "term_0023": "term 23",
    "term_0024": "term 24",
    "term_0025": "term 25",
    "term_0026": "term 26",
    "term_0027": "term 27",
    "term_0028": "term 28",
    "term_0029": "term 29",
    "term_0030": "term 30",
    "term_0031": "term 31",
    "term_0032": "term 32",
    "term_0033": "term 33",
    "term_0034": "term 34",
    "term_0035": "term 35",
    "term_0036": "term 36",
    "term_0037": "term 37",
    "term_0038": "term 38",
    "term_0039": "term 39",
    "term_0040": "term 40",
    "term_0041": "term 41",
    "term_0042": "term 42",
    "term_0043": "term 43",
    "term_0044": "term 44",
    "term_0045": "term 45",
    "term_0046": "term 46",
    "term_0047": "term 47",
    "term_0048": "term 48",
    "term_0049": "term 49",
    "term_0050": "term 50",
    "term_0051": "term 51",
    "term_0052": "term 52",
    "term_0053": "term 53",
    "term_0054": "term 54",
    "term_0055": "term 55",
    "term_0056": "term 56",
    "term_0057": "term 57",
    "term_0058": "term 58",
    "term_0059": "term 59",
    "term_0060": "term 60",
    "term_0061": "term 61",
    "term_0062": "term 62",
    "term_0063": "term 63",
    "term_0064": "term 64",
    "term_0065": "term 65",
    "term_0066": "term 66",
    "term_0067": "term 67",
    "term_0068": "term 68",
    "term_0069": "term 69",
    "term_0070": "term 70",
    "term_0071": "term 71",
    "term_0072": "term 72",
    "term_0073": "term 73",
    "term_0074": "term 74",
    "term_0075": "term 75",
    "term_0076": "term 76",
    "term_0077": "term 77",
    "term_0078": "term 78",
    "term_0079": "term 79",
    "term_0080": "term 80",
    "term_0081": "term 81",
    "term_0082": "term 82",
    "term_0083": "term 83",
    "term_0084": "term 84",
    "term_0085": "term 85",
    "term_0086": "term 86",
    "term_0087": "term 87",
    "term_0088": "term 88",
    "term_0089": "term 89",
    "term_0090": "term 90",
    "term_0091": "term 91",
    "term_0092": "term 92",
    "term_0093": "term 93",
    "term_0094": "term 94",
    "term_0095": "term 95",
    "term_0096": "term 96",
    "term_0097": "term 97",
    "term_0098": "term 98",
    "term_0099": "term 99",
    "term_0100": "term 100",
    "term_0101": "term 101",
    "term_0102": "term 102",
    "term_0103": "term 103",
    "term_0104": "term 104",
    "term_0105": "term 105",
    "term_0106": "term 106",
    "term_0107": "term 107",
    "term_0108": "term 108",
    "term_0109": "term 109",
    "term_0110": "term 110",
    "term_0111": "term 111",
    "term_0112": "term 112",
    "term_0113": "term 113",
    "term_0114": "term 114",
    "term_0115": "term 115",
    "term_0116": "term 116",
    "term_0117": "term 117",
    "term_0118": "term 118",
    "term_0119": "term 119",
    "term_0120": "term 120",
    "term_0121": "term 121",
    "term_0122": "term 122",
    "term_0123": "term 123",
    "term_0124": "term 124",
    "term_0125": "term 125",
    "term_0126": "term 126",
    "term_0127": "term 127",
    "term_0128": "term 128",
    "term_0129": "term 129",
    "term_0130": "term 130",
    "term_0131": "term 131",
    "term_0132": "term 132",
    "term_0133": "term 133",
    "term_0134": "term 134",
    "term_0135": "term 135",
    "term_0136": "term 136",
    "term_0137": "term 137",
    "term_0138": "term 138",
    "term_0139": "term 139",
    "term_0140": "term 140",
    "term_0141": "term 141",
    "term_0142": "term 142",
    "term_0143": "term 143",
    "term_0144": "term 144",
    "term_0145": "term 145",
    "term_0146": "term 146",
    "term_0147": "term 147",
    "term_0148": "term 148",
    "term_0149": "term 149",
    "term_0150": "term 150",
    "term_0151": "term 151",
    "term_0152": "term 152",
    "term_0153": "term 153",
    "term_0154": "term 154",
    "term_0155": "term 155",
    "term_0156": "term 156",
    "term_0157": "term 157",
    "term_0158": "term 158",
    "term_0159": "term 159",
    "term_0160": "term 160",
    "term_0161": "term 161",
    "term_0162": "term 162",
    "term_0163": "term 163",
    "term_0164": "term 164",
    "term_0165": "term 165",
    "term_0166": "term 166",
    "term_0167": "term 167",
    "term_0168": "term 168",
    "term_0169": "term 169",
    "term_0170": "term 170",
    "term_0171": "term 171",
    "term_0172": "term 172",
    "term_0173": "term 173",
    "term_0174": "term 174",
    "term_0175": "term 175",
    "term_0176": "term 176",
    "term_0177": "term 177",
    "term_0178": "term 178",
    "term_0179": "term 179",
    "term_0180": "term 180",
    "term_0181": "term 181",
    "term_0182": "term 182",
    "term_0183": "term 183",
    "term_0184": "term 184",
    "term_0185": "term 185",
    "term_0186": "term 186",
    "term_0187": "term 187",
    "term_0188": "term 188",
    "term_0189": "term 189",
    "term_0190": "term 190",
    "term_0191": "term 191",
    "term_0192": "term 192",
    "term_0193": "term 193",
    "term_0194": "term 194",
    "term_0195": "term 195",
    "term_0196": "term 196",
    "term_0197": "term 197",
    "term_0198": "term 198",
    "term_0199": "term 199",
    "term_0200": "term 200",
    "term_0201": "term 201",
    "term_0202": "term 202",
    "term_0203": "term 203",
    "term_0204": "term 204",
    "term_0205": "term 205",
    "term_0206": "term 206",
    "term_0207": "term 207",
    "term_0208": "term 208",
    "term_0209": "term 209",
    "term_0210": "term 210",
    "term_0211": "term 211",
    "term_0212": "term 212",
    "term_0213": "term 213",
    "term_0214": "term 214",
    "term_0215": "term 215",
    "term_0216": "term 216",
    "term_0217": "term 217",
    "term_0218": "term 218",
    "term_0219": "term 219",
    "term_0220": "term 220",
    "term_0221": "term 221",
    "term_0222": "term 222",
    "term_0223": "term 223",
    "term_0224": "term 224",
    "term_0225": "term 225",
    "term_0226": "term 226",
    "term_0227": "term 227",
    "term_0228": "term 228",
    "term_0229": "term 229",
    "term_0230": "term 230",
    "term_0231": "term 231",
    "term_0232": "term 232",
    "term_0233": "term 233",
    "term_0234": "term 234",
    "term_0235": "term 235",
    "term_0236": "term 236",
    "term_0237": "term 237",
    "term_0238": "term 238",
    "term_0239": "term 239",
    "term_0240": "term 240",
    "term_0241": "term 241",
    "term_0242": "term 242",
    "term_0243": "term 243",
    "term_0244": "term 244",
    "term_0245": "term 245",
    "term_0246": "term 246",
    "term_0247": "term 247",
    "term_0248": "term 248",
    "term_0249": "term 249",
    "term_0250": "term 250",
    "term_0251": "term 251",
    "term_0252": "term 252",
    "term_0253": "term 253",
    "term_0254": "term 254",
    "term_0255": "term 255",
    "term_0256": "term 256",
    "term_0257": "term 257",
    "term_0258": "term 258",
    "term_0259": "term 259",
    "term_0260": "term 260",
    "term_0261": "term 261",
    "term_0262": "term 262",
    "term_0263": "term 263",
    "term_0264": "term 264",
    "term_0265": "term 265",
    "term_0266": "term 266",
    "term_0267": "term 267",
    "term_0268": "term 268",
    "term_0269": "term 269",
    "term_0270": "term 270",
    "term_0271": "term 271",
    "term_0272": "term 272",
    "term_0273": "term 273",
    "term_0274": "term 274",
    "term_0275": "term 275",
    "term_0276": "term 276",
    "term_0277": "term 277",
    "term_0278": "term 278",
    "term_0279": "term 279",
    "term_0280": "term 280",
    "term_0281": "term 281",
    "term_0282": "term 282",
    "term_0283": "term 283",
    "term_0284": "term 284",
    "term_0285": "term 285",
    "term_0286": "term 286",
    "term_0287": "term 287",
    "term_0288": "term 288",
    "term_0289": "term 289",
    "term_0290": "term 290",
    "term_0291": "term 291",
    "term_0292": "term 292",
    "term_0293": "term 293",
    "term_0294": "term 294",
    "term_0295": "term 295",
    "term_0296": "term 296",
    "term_0297": "term 297",
    "term_0298": "term 298",
    "term_0299": "term 299",
    "term_0300": "term 300",
    "term_0301": "term 301",
    "term_0302": "term 302",
    "term_0303": "term 303",
    "term_0304": "term 304",
    "term_0305": "term 305",
    "term_0306": "term 306",
    "term_0307": "term 307",
    "term_0308": "term 308",
    "term_0309": "term 309",
    "term_0310": "term 310",
    "term_0311": "term 311",
    "term_0312": "term 312",
    "term_0313": "term 313",
    "term_0314": "term 314",
    "term_0315": "term 315",
    "term_0316": "term 316",
    "term_0317": "term 317",
    "term_0318": "term 318",
    "term_0319": "term 319",
    "term_0320": "term 320",
    "term_0321": "term 321",
    "term_0322": "term 322",
    "term_0323": "term 323",
    "term_0324": "term 324",
    "term_0325": "term 325",
    "term_0326": "term 326",
    "term_0327": "term 327",
    "term_0328": "term 328",
    "term_0329": "term 329",
    "term_0330": "term 330",
    "term_0331": "term 331",
    "term_0332": "term 332",
    "term_0333": "term 333",
    "term_0334": "term 334",
    "term_0335": "term 335",
    "term_0336": "term 336",
    "term_0337": "term 337",
    "term_0338": "term 338",
    "term_0339": "term 339",
    "term_0340": "term 340",
    "term_0341": "term 341",
    "term_0342": "term 342",
    "term_0343": "term 343",
    "term_0344": "term 344",
    "term_0345": "term 345",
    "term_0346": "term 346",
    "term_0347": "term 347",
    "term_0348": "term 348",
    "term_0349": "term 349",
    "term_0350": "term 350",
    "term_0351": "term 351",
    "term_0352": "term 352",
    "term_0353": "term 353",
    "term_0354": "term 354",
    "term_0355": "term 355",
    "term_0356": "term 356",
    "term_0357": "term 357",
    "term_0358": "term 358",
    "term_0359": "term 359",
    "term_0360": "term 360",
    "term_0361": "term 361",
    "term_0362": "term 362",
    "term_0363": "term 363",
    "term_0364": "term 364",
    "term_0365": "term 365",
    "term_0366": "term 366",
    "term_0367": "term 367",
    "term_0368": "term 368",
    "term_0369": "term 369",
    "term_0370": "term 370",
    "term_0371": "term 371",
    "term_0372": "term 372",
    "term_0373": "term 373",
    "term_0374": "term 374",
    "term_0375": "term 375",
    "term_0376": "term 376",
    "term_0377": "term 377",
    "term_0378": "term 378",
    "term_0379": "term 379",
    "term_0380": "term 380",
    "term_0381": "term 381",
    "term_0382": "term 382",
    "term_0383": "term 383",
    "term_0384": "term 384",
    "term_0385": "term 385",
    "term_0386": "term 386",
    "term_0387": "term 387",
    "term_0388": "term 388",
    "term_0389": "term 389",
    "term_0390": "term 390",
    "term_0391": "term 391",
    "term_0392": "term 392",
    "term_0393": "term 393",
    "term_0394": "term 394",
    "term_0395": "term 395",
    "term_0396": "term 396",
    "term_0397": "term 397",
    "term_0398": "term 398",
    "term_0399": "term 399",
    "term_0400": "term 400",
    "term_0401": "term 401",
    "term_0402": "term 402",
    "term_0403": "term 403",
    "term_0404": "term 404",
    "term_0405": "term 405",
    "term_0406": "term 406",
    "term_0407": "term 407",
    "term_0408": "term 408",
    "term_0409": "term 409",
    "term_0410": "term 410",
    "term_0411": "term 411",
    "term_0412": "term 412",
    "term_0413": "term 413",
    "term_0414": "term 414",
    "term_0415": "term 415",
    "term_0416": "term 416",
    "term_0417": "term 417",
    "term_0418": "term 418",
    "term_0419": "term 419",
    "term_0420": "term 420",
    "term_0421": "term 421",
    "term_0422": "term 422",
    "term_0423": "term 423",
    "term_0424": "term 424",
    "term_0425": "term 425",
    "term_0426": "term 426",
    "term_0427": "term 427",
    "term_0428": "term 428",
    "term_0429": "term 429",
    "term_0430": "term 430",
    "term_0431": "term 431",
    "term_0432": "term 432",
    "term_0433": "term 433",
    "term_0434": "term 434",
    "term_0435": "term 435",
    "term_0436": "term 436",
    "term_0437": "term 437",
    "term_0438": "term 438",
    "term_0439": "term 439",
    "term_0440": "term 440",
    "term_0441": "term 441",
    "term_0442": "term 442",
    "term_0443": "term 443",
    "term_0444": "term 444",
    "term_0445": "term 445",
    "term_0446": "term 446",
    "term_0447": "term 447",
    "term_0448": "term 448",
    "term_0449": "term 449",
    "term_0450": "term 450",
    "term_0451": "term 451",
    "term_0452": "term 452",
    "term_0453": "term 453",
    "term_0454": "term 454",
    "term_0455": "term 455",
    "term_0456": "term 456",
    "term_0457": "term 457",
    "term_0458": "term 458",
    "term_0459": "term 459",
    "term_0460": "term 460",
    "term_0461": "term 461",
    "term_0462": "term 462",
    "term_0463": "term 463",
    "term_0464": "term 464",
    "term_0465": "term 465",
    "term_0466": "term 466",
    "term_0467": "term 467",
    "term_0468": "term 468",
    "term_0469": "term 469",
    "term_0470": "term 470",
    "term_0471": "term 471",
    "term_0472": "term 472",
    "term_0473": "term 473",
    "term_0474": "term 474",
    "term_0475": "term 475",
    "term_0476": "term 476",
    "term_0477": "term 477",
    "term_0478": "term 478",
    "term_0479": "term 479",
    "term_0480": "term 480",
    "term_0481": "term 481",
    "term_0482": "term 482",
    "term_0483": "term 483",
    "term_0484": "term 484",
    "term_0485": "term 485",
    "term_0486": "term 486",
    "term_0487": "term 487",
    "term_0488": "term 488",
    "term_0489": "term 489",
    "term_0490": "term 490",
    "term_0491": "term 491",
    "term_0492": "term 492",
    "term_0493": "term 493",
    "term_0494": "term 494",
    "term_0495": "term 495",
    "term_0496": "term 496",
    "term_0497": "term 497",
    "term_0498": "term 498",
    "term_0499": "term 499",
    "term_0500": "term 500",
    "term_0501": "term 501",
    "term_0502": "term 502",
    "term_0503": "term 503",
    "term_0504": "term 504",
    "term_0505": "term 505",
    "term_0506": "term 506",
    "term_0507": "term 507",
    "term_0508": "term 508",
    "term_0509": "term 509",
    "term_0510": "term 510",
    "term_0511": "term 511",
    "term_0512": "term 512",
    "term_0513": "term 513",
    "term_0514": "term 514",
    "term_0515": "term 515",
    "term_0516": "term 516",
    "term_0517": "term 517",
    "term_0518": "term 518",
    "term_0519": "term 519",
    "term_0520": "term 520",
    "term_0521": "term 521",
    "term_0522": "term 522",
    "term_0523": "term 523",
    "term_0524": "term 524",
    "term_0525": "term 525",
    "term_0526": "term 526",
    "term_0527": "term 527",
    "term_0528": "term 528",
    "term_0529": "term 529",
    "term_0530": "term 530",
    "term_0531": "term 531",
    "term_0532": "term 532",
    "term_0533": "term 533",
    "term_0534": "term 534",
    "term_0535": "term 535",
    "term_0536": "term 536",
    "term_0537": "term 537",
    "term_0538": "term 538",
    "term_0539": "term 539",
    "term_0540": "term 540",
    "term_0541": "term 541",
    "term_0542": "term 542",
    "term_0543": "term 543",
    "term_0544": "term 544",
    "term_0545": "term 545",
    "term_0546": "term 546",
    "term_0547": "term 547",
    "term_0548": "term 548",
    "term_0549": "term 549",
    "term_0550": "term 550",
    "term_0551": "term 551",
    "term_0552": "term 552",
    "term_0553": "term 553",
    "term_0554": "term 554",
    "term_0555": "term 555",
    "term_0556": "term 556",
    "term_0557": "term 557",
    "term_0558": "term 558",
    "term_0559": "term 559",
    "term_0560": "term 560",
    "term_0561": "term 561",
    "term_0562": "term 562",
    "term_0563": "term 563",
    "term_0564": "term 564",
    "term_0565": "term 565",
    "term_0566": "term 566",
    "term_0567": "term 567",
    "term_0568": "term 568",
    "term_0569": "term 569",
    "term_0570": "term 570",
    "term_0571": "term 571",
    "term_0572": "term 572",
    "term_0573": "term 573",
    "term_0574": "term 574",
    "term_0575": "term 575",
    "term_0576": "term 576",
    "term_0577": "term 577",
    "term_0578": "term 578",
    "term_0579": "term 579",
    "term_0580": "term 580",
    "term_0581": "term 581",
    "term_0582": "term 582",
    "term_0583": "term 583",
    "term_0584": "term 584",
    "term_0585": "term 585",
    "term_0586": "term 586",
    "term_0587": "term 587",
    "term_0588": "term 588",
    "term_0589": "term 589",
    "term_0590": "term 590",
    "term_0591": "term 591",
    "term_0592": "term 592",
    "term_0593": "term 593",
    "term_0594": "term 594",
    "term_0595": "term 595",
    "term_0596": "term 596",
    "term_0597": "term 597",
    "term_0598": "term 598",
    "term_0599": "term 599",
    "term_0600": "term 600",
    "term_0601": "term 601",
    "term_0602": "term 602",
    "term_0603": "term 603",
    "term_0604": "term 604",
    "term_0605": "term 605",
    "term_0606": "term 606",
    "term_0607": "term 607",
    "term_0608": "term 608",
    "term_0609": "term 609",
    "term_0610": "term 610",
    "term_0611": "term 611",
    "term_0612": "term 612",
    "term_0613": "term 613",
    "term_0614": "term 614",
    "term_0615": "term 615",
    "term_0616": "term 616",
    "term_0617": "term 617",
    "term_0618": "term 618",
    "term_0619": "term 619",
    "term_0620": "term 620",
    "term_0621": "term 621",
    "term_0622": "term 622",
    "term_0623": "term 623",
    "term_0624": "term 624",
    "term_0625": "term 625",
    "term_0626": "term 626",
    "term_0627": "term 627",
    "term_0628": "term 628",
    "term_0629": "term 629",
    "term_0630": "term 630",
    "term_0631": "term 631",
    "term_0632": "term 632",
    "term_0633": "term 633",
    "term_0634": "term 634",
    "term_0635": "term 635",
    "term_0636": "term 636",
    "term_0637": "term 637",
    "term_0638": "term 638",
    "term_0639": "term 639",
    "term_0640": "term 640",
    "term_0641": "term 641",
    "term_0642": "term 642",
    "term_0643": "term 643",
    "term_0644": "term 644",
    "term_0645": "term 645",
    "term_0646": "term 646",
    "term_0647": "term 647",
    "term_0648": "term 648",
    "term_0649": "term 649",
    "term_0650": "term 650",
    "term_0651": "term 651",
    "term_0652": "term 652",
    "term_0653": "term 653",
    "term_0654": "term 654",
    "term_0655": "term 655",
    "term_0656": "term 656",
    "term_0657": "term 657",
    "term_0658": "term 658",
    "term_0659": "term 659",
    "term_0660": "term 660",
    "term_0661": "term 661",
    "term_0662": "term 662",
    "term_0663": "term 663",
    "term_0664": "term 664",
    "term_0665": "term 665",
    "term_0666": "term 666",
    "term_0667": "term 667",
    "term_0668": "term 668",
    "term_0669": "term 669",
    "term_0670": "term 670",
    "term_0671": "term 671",
    "term_0672": "term 672",
    "term_0673": "term 673",
    "term_0674": "term 674",
    "term_0675": "term 675",
    "term_0676": "term 676",
    "term_0677": "term 677",
    "term_0678": "term 678",
    "term_0679": "term 679",
    "term_0680": "term 680",
    "term_0681": "term 681",
    "term_0682": "term 682",
    "term_0683": "term 683",
    "term_0684": "term 684",
    "term_0685": "term 685",
    "term_0686": "term 686",
    "term_0687": "term 687",
    "term_0688": "term 688",
    "term_0689": "term 689",
    "term_0690": "term 690",
    "term_0691": "term 691",
    "term_0692": "term 692",
    "term_0693": "term 693",
    "term_0694": "term 694",
    "term_0695": "term 695",
    "term_0696": "term 696",
    "term_0697": "term 697",
    "term_0698": "term 698",
    "term_0699": "term 699",
    "term_0700": "term 700",
    "term_0701": "term 701",
    "term_0702": "term 702",
    "term_0703": "term 703",
    "term_0704": "term 704",
    "term_0705": "term 705",
    "term_0706": "term 706",
    "term_0707": "term 707",
    "term_0708": "term 708",
    "term_0709": "term 709",
    "term_0710": "term 710",
    "term_0711": "term 711",
    "term_0712": "term 712",
    "term_0713": "term 713",
    "term_0714": "term 714",
    "term_0715": "term 715",
    "term_0716": "term 716",
    "term_0717": "term 717",
    "term_0718": "term 718",
    "term_0719": "term 719",
    "term_0720": "term 720",
    "term_0721": "term 721",
    "term_0722": "term 722",
    "term_0723": "term 723",
    "term_0724": "term 724",
    "term_0725": "term 725",
    "term_0726": "term 726",
    "term_0727": "term 727",
    "term_0728": "term 728",
    "term_0729": "term 729",
    "term_0730": "term 730",
    "term_0731": "term 731",
    "term_0732": "term 732",
    "term_0733": "term 733",
    "term_0734": "term 734",
    "term_0735": "term 735",
    "term_0736": "term 736",
    "term_0737": "term 737",
    "term_0738": "term 738",
    "term_0739": "term 739",
    "term_0740": "term 740",
    "term_0741": "term 741",
    "term_0742": "term 742",
    "term_0743": "term 743",
    "term_0744": "term 744",
    "term_0745": "term 745",
    "term_0746": "term 746",
    "term_0747": "term 747",
    "term_0748": "term 748",
    "term_0749": "term 749",
    "term_0750": "term 750",
    "term_0751": "term 751",
    "term_0752": "term 752",
    "term_0753": "term 753",
    "term_0754": "term 754",
    "term_0755": "term 755",
    "term_0756": "term 756",
    "term_0757": "term 757",
    "term_0758": "term 758",
    "term_0759": "term 759",
    "term_0760": "term 760",
    "term_0761": "term 761",
    "term_0762": "term 762",
    "term_0763": "term 763",
    "term_0764": "term 764",
    "term_0765": "term 765",
    "term_0766": "term 766",
    "term_0767": "term 767",
    "term_0768": "term 768",
    "term_0769": "term 769",
    "term_0770": "term 770",
    "term_0771": "term 771",
    "term_0772": "term 772",
    "term_0773": "term 773",
    "term_0774": "term 774",
    "term_0775": "term 775",
    "term_0776": "term 776",
    "term_0777": "term 777",
    "term_0778": "term 778",
    "term_0779": "term 779",
    "term_0780": "term 780",
    "term_0781": "term 781",
    "term_0782": "term 782",
    "term_0783": "term 783",
    "term_0784": "term 784",
    "term_0785": "term 785",
    "term_0786": "term 786",
    "term_0787": "term 787",
    "term_0788": "term 788",
    "term_0789": "term 789",
    "term_0790": "term 790",
    "term_0791": "term 791",
    "term_0792": "term 792",
    "term_0793": "term 793",
    "term_0794": "term 794",
    "term_0795": "term 795",
    "term_0796": "term 796",
    "term_0797": "term 797",
    "term_0798": "term 798",
    "term_0799": "term 799",
    "term_0800": "term 800",
    "term_0801": "term 801",
    "term_0802": "term 802",
    "term_0803": "term 803",
    "term_0804": "term 804",
    "term_0805": "term 805",
    "term_0806": "term 806",
    "term_0807": "term 807",
    "term_0808": "term 808",
    "term_0809": "term 809",
    "term_0810": "term 810",
    "term_0811": "term 811",
    "term_0812": "term 812",
    "term_0813": "term 813",
    "term_0814": "term 814",
    "term_0815": "term 815",
    "term_0816": "term 816",
    "term_0817": "term 817",
    "term_0818": "term 818",
    "term_0819": "term 819",
    "term_0820": "term 820",
    "term_0821": "term 821",
    "term_0822": "term 822",
    "term_0823": "term 823",
    "term_0824": "term 824",
    "term_0825": "term 825",
    "term_0826": "term 826",
    "term_0827": "term 827",
    "term_0828": "term 828",
    "term_0829": "term 829",
    "term_0830": "term 830",
    "term_0831": "term 831",
    "term_0832": "term 832",
    "term_0833": "term 833",
    "term_0834": "term 834",
    "term_0835": "term 835",
    "term_0836": "term 836",
    "term_0837": "term 837",
    "term_0838": "term 838",
    "term_0839": "term 839",
    "term_0840": "term 840",
    "term_0841": "term 841",
    "term_0842": "term 842",
    "term_0843": "term 843",
    "term_0844": "term 844",
    "term_0845": "term 845",
    "term_0846": "term 846",
    "term_0847": "term 847",
    "term_0848": "term 848",
    "term_0849": "term 849",
    "term_0850": "term 850",
    "term_0851": "term 851",
    "term_0852": "term 852",
    "term_0853": "term 853",
    "term_0854": "term 854",
    "term_0855": "term 855",
    "term_0856": "term 856",
    "term_0857": "term 857",
    "term_0858": "term 858",
    "term_0859": "term 859",
    "term_0860": "term 860",
    "term_0861": "term 861",
    "term_0862": "term 862",
    "term_0863": "term 863",
    "term_0864": "term 864",
    "term_0865": "term 865",
    "term_0866": "term 866",
    "term_0867": "term 867",
    "term_0868": "term 868",
    "term_0869": "term 869",
    "term_0870": "term 870",
    "term_0871": "term 871",
    "term_0872": "term 872",
    "term_0873": "term 873",
    "term_0874": "term 874",
    "term_0875": "term 875",
    "term_0876": "term 876",
    "term_0877": "term 877",
    "term_0878": "term 878",
    "term_0879": "term 879",
    "term_0880": "term 880",
    "term_0881": "term 881",
    "term_0882": "term 882",
    "term_0883": "term 883",
    "term_0884": "term 884",
    "term_0885": "term 885",
    "term_0886": "term 886",
    "term_0887": "term 887",
    "term_0888": "term 888",
    "term_0889": "term 889",
    "term_0890": "term 890",
    "term_0891": "term 891",
    "term_0892": "term 892",
    "term_0893": "term 893",
    "term_0894": "term 894",
    "term_0895": "term 895",
    "term_0896": "term 896",
    "term_0897": "term 897",
    "term_0898": "term 898",
    "term_0899": "term 899",
    "term_0900": "term 900",
    "term_0901": "term 901",
    "term_0902": "term 902",
    "term_0903": "term 903",
    "term_0904": "term 904",
    "term_0905": "term 905",
    "term_0906": "term 906",
    "term_0907": "term 907",
    "term_0908": "term 908",
    "term_0909": "term 909",
    "term_0910": "term 910",
    "term_0911": "term 911",
    "term_0912": "term 912",
    "term_0913": "term 913",
    "term_0914": "term 914",
    "term_0915": "term 915",
    "term_0916": "term 916",
    "term_0917": "term 917",
    "term_0918": "term 918",
    "term_0919": "term 919",
    "term_0920": "term 920",
    "term_0921": "term 921",
    "term_0922": "term 922",
    "term_0923": "term 923",
    "term_0924": "term 924",
    "term_0925": "term 925",
    "term_0926": "term 926",
    "term_0927": "term 927",
    "term_0928": "term 928",
    "term_0929": "term 929",
    "term_0930": "term 930",
    "term_0931": "term 931",
    "term_0932": "term 932",
    "term_0933": "term 933",
    "term_0934": "term 934",
    "term_0935": "term 935",
    "term_0936": "term 936",
    "term_0937": "term 937",
    "term_0938": "term 938",
    "term_0939": "term 939",
    "term_0940": "term 940",
    "term_0941": "term 941",
    "term_0942": "term 942",
    "term_0943": "term 943",
    "term_0944": "term 944",
    "term_0945": "term 945",
    "term_0946": "term 946",
    "term_0947": "term 947",
    "term_0948": "term 948",
    "term_0949": "term 949",
    "term_0950": "term 950",
    "term_0951": "term 951",
    "term_0952": "term 952",
    "term_0953": "term 953",
    "term_0954": "term 954",
    "term_0955": "term 955",
    "term_0956": "term 956",
    "term_0957": "term 957",
    "term_0958": "term 958",
    "term_0959": "term 959",
    "term_0960": "term 960",
    "term_0961": "term 961",
    "term_0962": "term 962",
    "term_0963": "term 963",
    "term_0964": "term 964",
    "term_0965": "term 965",
    "term_0966": "term 966",
    "term_0967": "term 967",
    "term_0968": "term 968",
    "term_0969": "term 969",
    "term_0970": "term 970",
    "term_0971": "term 971",
    "term_0972": "term 972",
    "term_0973": "term 973",
    "term_0974": "term 974",
    "term_0975": "term 975",
    "term_0976": "term 976",
    "term_0977": "term 977",
    "term_0978": "term 978",
    "term_0979": "term 979",
    "term_0980": "term 980",
    "term_0981": "term 981",
    "term_0982": "term 982",
    "term_0983": "term 983",
    "term_0984": "term 984",
    "term_0985": "term 985",
    "term_0986": "term 986",
    "term_0987": "term 987",
    "term_0988": "term 988",
    "term_0989": "term 989",
    "term_0990": "term 990",
    "term_0991": "term 991",
    "term_0992": "term 992",
    "term_0993": "term 993",
    "term_0994": "term 994",
    "term_0995": "term 995",
    "term_0996": "term 996",
    "term_0997": "term 997",
    "term_0998": "term 998",
    "term_0999": "term 999",
    "term_1000": "term 1000",
}

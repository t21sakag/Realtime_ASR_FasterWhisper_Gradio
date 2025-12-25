from dataclasses import dataclass
import os


def _normalize_language(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"japanese", "ja-jp", "jp"}:
        return "ja"
    return normalized


@dataclass(frozen=True)
class StreamingConfig:
    whisper_model_id: str = os.getenv("WHISPER_MODEL_ID", "large-v3")
    language: str = _normalize_language(os.getenv("LANGUAGE", "ja"))
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", 0.5))
    min_speech_duration_ms: int = int(os.getenv("MIN_SPEECH_DURATION_MS", 250))
    min_silence_duration_ms: int = int(os.getenv("MIN_SILENCE_DURATION_MS", 400))
    speech_pad_ms: int = int(os.getenv("SPEECH_PAD_MS", 120))
    max_continuous_speech_s: float = float(os.getenv("MAX_CONTINUOUS_SPEECH_S", 15.0))
    pre_roll_ms: int = int(os.getenv("PRE_ROLL_MS", 1000))
    asr_url: str | None = os.getenv("ASR_URL") or None
    debug: bool = os.getenv("STREAMING_AUDIO_DEBUG", "").lower() in ("1", "true", "yes")

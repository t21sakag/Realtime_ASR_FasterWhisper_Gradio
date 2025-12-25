from __future__ import annotations

import logging
import time
from io import BytesIO
from typing import Tuple

import numpy as np
import requests
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available

from core.config import StreamingConfig
from core.session_store import StreamingState
from utils.audio import int16_to_float32, resample_to_16k, to_mono

logger = logging.getLogger(__name__)


class StreamingAudioProcessor:
    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self.streaming_pipeline = None
        self.vad_model = None
        self.get_speech_timestamps = None
        self.np_dtype = np.float32
        if self.config.debug:
            logger.setLevel(logging.INFO)
            logger.info(
                "Streaming debug enabled: vad_threshold=%s min_speech_ms=%s min_silence_ms=%s",
                self.config.vad_threshold,
                self.config.min_speech_duration_ms,
                self.config.min_silence_duration_ms,
            )
        self._load_models()

    def _load_models(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.np_dtype = np.float16 if torch.cuda.is_available() else np.float32
        attention = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        if self.config.asr_url:
            logger.info("Remote ASR enabled: %s", self.config.asr_url)
        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.whisper_model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attention,
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(self.config.whisper_model_id)
            self.streaming_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )

        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.vad_model = vad_model
        self.get_speech_timestamps = vad_utils[0]
        self.vad_model.eval()

    def _transcribe(self, audio_16k: np.ndarray) -> str:
        try:
            if self.config.asr_url:
                buffer = BytesIO()
                sf.write(buffer, audio_16k.astype(np.float32), 16000, format="WAV")
                buffer.seek(0)
                resp = requests.post(
                    f"{self.config.asr_url.rstrip('/')}/transcribe",
                    files={"file": ("audio.wav", buffer.read(), "audio/wav")},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json().get("text", "").strip()

            if self.streaming_pipeline is None:
                return ""

            outputs = self.streaming_pipeline(
                {"sampling_rate": 16000, "raw": audio_16k},
                chunk_length_s=5,
                batch_size=1,
                generate_kwargs={
                    "task": "transcribe",
                    "language": self.config.language,
                },
            )
            return outputs.get("text", "").strip()
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return ""

    def transcribe_audio(self, audio_16k: np.ndarray) -> str:
        return self._transcribe(audio_16k)

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio_array = to_mono(audio)
        if audio_array.dtype == np.int16:
            audio_float32 = int16_to_float32(audio_array)
        else:
            audio_float32 = audio_array.astype(np.float32)
        audio_float32, _ = resample_to_16k(audio_float32, sample_rate)
        return audio_float32.astype(np.float32)

    def detect_speech(self, audio_16k: np.ndarray) -> list[dict[str, int]]:
        try:
            speech_timestamps = self.get_speech_timestamps(
                torch.from_numpy(audio_16k),
                self.vad_model,
                threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms,
                return_seconds=False,
            )
        except Exception as exc:
            logger.error("VAD model prediction failed: %s", exc)
            speech_timestamps = []
        return speech_timestamps

    def _append_transcript(self, state: StreamingState, result_text: str) -> None:
        if not result_text:
            return
        if state.accumulated_transcript:
            state.accumulated_transcript = f"{state.accumulated_transcript} {result_text}"
        else:
            state.accumulated_transcript = result_text

    def _consume_buffer(self, state: StreamingState) -> str:
        if not state.speech_buffer:
            return ""
        full_speech = np.concatenate(state.speech_buffer)
        max_val = np.abs(full_speech).max()
        if max_val > 0:
            full_speech = full_speech / max_val
        full_speech = full_speech.astype(self.np_dtype)
        result_text = self._transcribe(full_speech)
        state.speech_buffer = []
        return result_text

    def process_chunk(
        self,
        state: StreamingState,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[StreamingState, str, bool]:
        if audio is None:
            raise TypeError("Audio input cannot be None")
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Expected np.ndarray for audio, got {type(audio).__name__}")

        audio_float32 = self.preprocess_audio(audio, sample_rate)
        current_time = time.time()

        pre_roll_samples = int((self.config.pre_roll_ms / 1000.0) * 16000)
        pre_buf = state.pre_roll_buffer
        pre_buf = np.concatenate([pre_buf, audio_float32])
        if len(pre_buf) > pre_roll_samples:
            pre_buf = pre_buf[-pre_roll_samples:]
        state.pre_roll_buffer = pre_buf

        speech_timestamps = self.detect_speech(audio_float32)

        has_speech = len(speech_timestamps) > 0
        if self.config.debug:
            rms = float(np.sqrt(np.mean(audio_float32**2))) if audio_float32.size else 0.0
            max_abs = float(np.max(np.abs(audio_float32))) if audio_float32.size else 0.0
            logger.info(
                "VAD chunk: samples=%d rms=%.4f max=%.4f speech=%s segs=%d",
                len(audio_float32),
                rms,
                max_abs,
                has_speech,
                len(speech_timestamps),
            )

        if has_speech:
            if state.new_recording_session:
                state.accumulated_transcript = ""
                state.new_recording_session = False

            state.last_speech_time = current_time
            state.is_speaking = True

            if len(state.pre_roll_buffer) > 0:
                state.speech_buffer.append(state.pre_roll_buffer.copy())
                state.pre_roll_buffer = np.zeros(0, dtype=np.float32)

            for seg in speech_timestamps:
                state.speech_buffer.append(audio_float32[seg["start"] : seg["end"]])

        buffer_size = sum(len(chunk) for chunk in state.speech_buffer)
        buffer_duration_s = buffer_size / 16000 if buffer_size > 0 else 0.0
        silence_duration_ms = (current_time - state.last_speech_time) * 1000

        stale_transcript = (
            not state.is_speaking
            and not has_speech
            and len(state.speech_buffer) == 0
            and state.accumulated_transcript.strip()
            and silence_duration_ms >= (self.config.min_silence_duration_ms * 2)
        )
        if stale_transcript:
            state.accumulated_transcript = ""
            state.new_recording_session = True

        should_transcribe_on_silence = (
            state.is_speaking
            and not has_speech
            and silence_duration_ms >= self.config.min_silence_duration_ms
            and len(state.speech_buffer) > 0
        )
        should_transcribe_on_duration = (
            buffer_duration_s >= self.config.max_continuous_speech_s
            and len(state.speech_buffer) > 0
        )
        should_transcribe = should_transcribe_on_silence or should_transcribe_on_duration

        result_text = ""
        is_final_transcript = False

        if should_transcribe:
            result_text = self._consume_buffer(state)
            self._append_transcript(state, result_text)

            if should_transcribe_on_silence:
                state.is_speaking = False
                state.new_recording_session = True
                is_final_transcript = True

        return state, state.accumulated_transcript, is_final_transcript

    def flush(self, state: StreamingState) -> Tuple[StreamingState, str, bool]:
        result_text = self._consume_buffer(state)
        self._append_transcript(state, result_text)
        state.is_speaking = False
        state.new_recording_session = True
        state.pre_roll_buffer = np.zeros(0, dtype=np.float32)
        return state, state.accumulated_transcript, True

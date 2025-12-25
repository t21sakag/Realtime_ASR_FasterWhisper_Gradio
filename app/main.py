import os
from io import BytesIO
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import StreamingConfig
from core.processor import StreamingAudioProcessor
from core.session_store import SessionStore

app = FastAPI(title="Realtime ASR FasterWhisper Service", version="0.1.0")

config = StreamingConfig()
processor = StreamingAudioProcessor(config)
session_store = SessionStore(ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")))


class ResetRequest(BaseModel):
    session_id: str


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/stream")
async def stream_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    end_of_stream: Optional[str] = Form(None),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio, samplerate = sf.read(BytesIO(data))
        audio = np.asarray(audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}")

    session_id, state = session_store.get_or_create(session_id)
    try:
        _, text, is_final = processor.process_chunk(state, audio, int(samplerate))
        if _parse_bool(end_of_stream):
            _, text, is_final = processor.flush(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Streaming processing failed: {exc}")

    return {"session_id": session_id, "text": text or "", "is_final": is_final}


@app.post("/vad")
async def vad(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio, samplerate = sf.read(BytesIO(data))
        audio = np.asarray(audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}")

    audio_16k = processor.preprocess_audio(audio, int(samplerate))
    segments = processor.detect_speech(audio_16k)
    segments_out = [
        {
            "start": int(seg["start"]),
            "end": int(seg["end"]),
            "start_sec": float(seg["start"]) / 16000,
            "end_sec": float(seg["end"]) / 16000,
        }
        for seg in segments
    ]
    return {
        "has_speech": len(segments_out) > 0,
        "segments": segments_out,
        "sample_rate": 16000,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    use_vad: Optional[str] = Form(None),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio, samplerate = sf.read(BytesIO(data))
        audio = np.asarray(audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}")

    audio_16k = processor.preprocess_audio(audio, int(samplerate))
    use_vad_flag = _parse_bool(use_vad)
    segments_out = []
    if use_vad_flag:
        segments = processor.detect_speech(audio_16k)
        segments_out = [
            {
                "start": int(seg["start"]),
                "end": int(seg["end"]),
                "start_sec": float(seg["start"]) / 16000,
                "end_sec": float(seg["end"]) / 16000,
            }
            for seg in segments
        ]
        if segments:
            audio_16k = np.concatenate([audio_16k[seg["start"] : seg["end"]] for seg in segments])
        else:
            audio_16k = np.zeros(0, dtype=np.float32)

    text = processor.transcribe_audio(audio_16k) if len(audio_16k) > 0 else ""
    return {"text": text or "", "use_vad": use_vad_flag, "segments": segments_out}


@app.post("/session/reset")
async def reset_session(req: ResetRequest):
    if not session_store.reset(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9100)

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Optional, Tuple

import numpy as np


@dataclass
class StreamingState:
    speech_buffer: list[np.ndarray] = field(default_factory=list)
    last_speech_time: float = field(default_factory=time.time)
    is_speaking: bool = False
    accumulated_transcript: str = ""
    new_recording_session: bool = True
    pre_roll_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass
class _SessionItem:
    state: StreamingState
    last_access: float


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionItem] = {}

    def _cleanup(self, now: float) -> None:
        expired = [
            session_id
            for session_id, item in self._sessions.items()
            if (now - item.last_access) > self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def get_or_create(self, session_id: Optional[str]) -> Tuple[str, StreamingState]:
        now = time.time()
        with self._lock:
            self._cleanup(now)
            if session_id and session_id in self._sessions:
                item = self._sessions[session_id]
                item.last_access = now
                return session_id, item.state

            new_id = str(uuid.uuid4())
            state = StreamingState()
            self._sessions[new_id] = _SessionItem(state=state, last_access=now)
            return new_id, state

    def reset(self, session_id: str) -> bool:
        now = time.time()
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None:
                return False
            item.state = StreamingState()
            item.last_access = now
            self._sessions[session_id] = item
            return True

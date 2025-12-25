import inspect
import os
import time
from io import BytesIO
from typing import Any

import gradio as gr
import numpy as np
import requests
import soundfile as sf


if "GRADIO_DEBUG" in os.environ:
    debug_raw = os.environ["GRADIO_DEBUG"].strip().lower()
    if debug_raw in {"1", "true", "yes", "on"}:
        os.environ["GRADIO_DEBUG"] = "1"
    else:
        os.environ["GRADIO_DEBUG"] = "0"

STREAMING_AUDIO_URL = os.getenv("STREAMING_AUDIO_URL", "http://localhost:19100")
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7862"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "").lower() in {"1", "true", "yes", "on"}
GRADIO_DEBUG = os.getenv("GRADIO_DEBUG", "").lower() in {"1", "true", "yes", "on"}
GRADIO_SHARE_WAIT = float(os.getenv("GRADIO_SHARE_WAIT", "30"))
WAIT_FOR_AUDIO = os.getenv("WAIT_FOR_AUDIO", "").lower() in {"1", "true", "yes", "on"}
WAIT_FOR_AUDIO_TIMEOUT = float(os.getenv("WAIT_FOR_AUDIO_TIMEOUT", "180"))
WAIT_FOR_AUDIO_INTERVAL = float(os.getenv("WAIT_FOR_AUDIO_INTERVAL", "2.0"))
CHATBOT_FORMAT = os.getenv("CHATBOT_FORMAT", "tuples").strip().lower()
STREAM_EVERY = float(os.getenv("STREAM_EVERY", "0.3"))

DEFAULT_CHATBOT_USE_MESSAGES = False
try:
    CHATBOT_SUPPORTS_TYPE = "type" in inspect.signature(gr.Chatbot.__init__).parameters
except (TypeError, ValueError):
    CHATBOT_SUPPORTS_TYPE = False

try:
    LAUNCH_SUPPORTS_PREVENT_THREAD_LOCK = (
        "prevent_thread_lock" in inspect.signature(gr.Blocks.launch).parameters
    )
except (TypeError, ValueError):
    LAUNCH_SUPPORTS_PREVENT_THREAD_LOCK = False


def _init_state(chatbot_use_messages: bool | None = None) -> dict[str, Any]:
    if chatbot_use_messages is None:
        chatbot_use_messages = DEFAULT_CHATBOT_USE_MESSAGES
    return {
        "session_id": None,
        "speaking": False,
        "last_text": "",
        "chatbot_use_messages": bool(chatbot_use_messages),
        "history": [],
        "last_final_text": "",
    }


def _normalize_audio(audio: Any) -> tuple[int, np.ndarray] | None:
    if audio is None:
        return None
    if isinstance(audio, list) and audio and isinstance(audio[0], tuple):
        audio = audio[0]
    if not isinstance(audio, tuple) or len(audio) != 2:
        raise TypeError("audio must be (sample_rate, audio_array)")
    sample_rate, audio_array = audio
    audio_array = np.asarray(audio_array)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    return int(sample_rate), audio_array


def _post_stream(session_id: str | None, audio: tuple[int, np.ndarray]) -> dict[str, Any]:
    sample_rate, audio_array = audio
    buffer = BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    data = {}
    if session_id:
        data["session_id"] = session_id
    resp = requests.post(
        f"{STREAMING_AUDIO_URL.rstrip('/')}/stream",
        files={"file": ("audio.wav", buffer.read(), "audio/wav")},
        data=data,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _status_text(state: dict[str, Any], event: str | None = None) -> str:
    if event == "start":
        return "ステータス: 発話開始"
    if event == "end":
        return "ステータス: 発話終了"
    if state.get("speaking"):
        return "ステータス: 発話中"
    return "ステータス: 待機中"


def _append_history(
    chat_history: list[Any],
    text: str,
    use_messages: bool,
) -> list[Any]:
    if use_messages:
        return chat_history + [{"role": "user", "content": text}]
    return chat_history + [(text, "")]


def _diff_final_text(full_text: str, prev_text: str) -> str:
    if not prev_text:
        return full_text
    if full_text == prev_text:
        return ""
    if full_text.startswith(prev_text):
        return full_text[len(prev_text) :].strip()
    prefix_len = 0
    for a, b in zip(full_text, prev_text):
        if a != b:
            break
        prefix_len += 1
    if prefix_len:
        return full_text[prefix_len:].strip()
    return full_text


def _extract_message(item: Any) -> tuple[str, str] | None:
    if isinstance(item, dict):
        role = item.get("role")
        content = item.get("content")
        if role is not None and content is not None:
            return str(role), str(content)
        return None
    role = getattr(item, "role", None)
    if role is None:
        return None
    content = getattr(item, "content", None)
    if content is None:
        content = getattr(item, "text", None)
    if content is None:
        content = getattr(item, "message", None)
    if content is None:
        return None
    return str(role), str(content)


def _normalize_history(chat_history: list[Any], use_messages: bool) -> list[Any]:
    if not chat_history:
        return chat_history
    first = chat_history[0]
    if use_messages:
        normalized: list[dict[str, str]] = []
        for item in chat_history:
            message = _extract_message(item)
            if message:
                role, content = message
                normalized.append({"role": role, "content": content})
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user, assistant = item
                if user:
                    normalized.append({"role": "user", "content": str(user)})
                if assistant:
                    normalized.append({"role": "assistant", "content": str(assistant)})
        return normalized
    if isinstance(first, (list, tuple)):
        return chat_history
    normalized_pairs: list[tuple[str, str]] = []
    for item in chat_history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized_pairs.append((str(item[0]), str(item[1])))
            continue
        message = _extract_message(item)
        if message:
            role, content = message
            if role == "user":
                normalized_pairs.append((content, ""))
            elif role == "assistant":
                if normalized_pairs:
                    prev_user, prev_assistant = normalized_pairs[-1]
                    if not prev_assistant:
                        normalized_pairs[-1] = (prev_user, content)
                    else:
                        normalized_pairs.append(("", content))
                else:
                    normalized_pairs.append(("", content))
    return normalized_pairs


def _detect_chatbot_use_messages(chatbot: gr.Chatbot) -> bool:
    try:
        chatbot.postprocess([{"role": "user", "content": "ping"}])
        return True
    except Exception:
        pass
    try:
        chatbot.postprocess([("ping", "")])
        return False
    except Exception:
        return True


def _create_chatbot() -> tuple[gr.Chatbot, bool]:
    kwargs: dict[str, Any] = {"label": "チャット履歴", "height": 420}
    preferred = "messages" if CHATBOT_FORMAT == "messages" else "tuples"
    fallback = "tuples" if preferred == "messages" else "messages"
    if CHATBOT_SUPPORTS_TYPE:
        for fmt in (preferred, fallback):
            try:
                chatbot = gr.Chatbot(**kwargs, type=fmt)
                return chatbot, fmt == "messages"
            except Exception as exc:
                if not isinstance(exc, TypeError) and exc.__class__.__name__ != "Error":
                    raise
    chatbot = gr.Chatbot(**kwargs)
    if CHATBOT_FORMAT in {"messages", "tuples"}:
        return chatbot, CHATBOT_FORMAT == "messages"
    chatbot_use_messages = _detect_chatbot_use_messages(chatbot)
    return chatbot, chatbot_use_messages


def _extract_share_url(launch_result: Any, ui: gr.Blocks) -> str | None:
    if isinstance(launch_result, (tuple, list)):
        for item in launch_result:
            if isinstance(item, str) and item.startswith("https://"):
                return item
    for obj in (launch_result, ui):
        share_url = getattr(obj, "share_url", None)
        if isinstance(share_url, str) and share_url:
            return share_url
    return None


def _local_url() -> str:
    host = GRADIO_SERVER_NAME
    if host in {"0.0.0.0", "::"}:
        host = "localhost"
    return f"http://{host}:{GRADIO_SERVER_PORT}"


def _wait_for_share_url(launch_result: Any, ui: gr.Blocks) -> str | None:
    share_url = _extract_share_url(launch_result, ui)
    if share_url:
        return share_url
    deadline = time.time() + GRADIO_SHARE_WAIT
    while time.time() < deadline:
        share_url = _extract_share_url(None, ui)
        if share_url:
            return share_url
        time.sleep(0.5)
    return None


def _block_forever(ui: gr.Blocks, use_block: bool) -> None:
    if not use_block:
        return
    if hasattr(ui, "block_thread"):
        ui.block_thread()
        return
    while True:
        time.sleep(3600)


def _wait_for_audio_service() -> None:
    if not WAIT_FOR_AUDIO:
        return
    health_url = f"{STREAMING_AUDIO_URL.rstrip('/')}/health"
    deadline = time.time() + WAIT_FOR_AUDIO_TIMEOUT
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.ok:
                return
        except Exception as exc:
            if GRADIO_DEBUG:
                print(f"[gradio-client] wait for audio failed: {exc}", flush=True)
        time.sleep(WAIT_FOR_AUDIO_INTERVAL)
    print(f"[gradio-client] audio health check timed out: {health_url}", flush=True)


def stream_handler(state: dict[str, Any] | None, audio: Any, chat_history: list[Any] | None):
    if state is None:
        state = _init_state()
    use_messages = bool(state.get("chatbot_use_messages", DEFAULT_CHATBOT_USE_MESSAGES))
    state["chatbot_use_messages"] = use_messages
    history = state.get("history") or []
    if not history:
        history = chat_history or []
        if isinstance(history, tuple):
            history = list(history)
        history = _normalize_history(history, use_messages)
        state["history"] = history

    audio_tuple = _normalize_audio(audio)
    if audio_tuple is None:
        return state, history, _status_text(state), state.get("session_id", "")

    try:
        payload = _post_stream(state.get("session_id"), audio_tuple)
    except Exception as exc:
        if GRADIO_DEBUG:
            print(f"[gradio-client] stream error: {exc}", flush=True)
        return state, history, "ステータス: 通信エラー", state.get("session_id", "")

    session_id = payload.get("session_id") or state.get("session_id")
    text = payload.get("text", "") or ""
    is_final = bool(payload.get("is_final", False))

    event = None
    if text and not state.get("speaking"):
        state["speaking"] = True
        event = "start"

    if is_final:
        final_text = text.strip()
        if final_text:
            prev_text = str(state.get("last_final_text", "") or "")
            delta_text = _diff_final_text(final_text, prev_text)
            append_text = delta_text or final_text
            history = _append_history(history, append_text, use_messages)
            state["history"] = history
            state["last_final_text"] = final_text
            if GRADIO_DEBUG:
                print(
                    f"[gradio-client] history size: {len(history)} delta_len={len(delta_text)}",
                    flush=True,
                )
        state["speaking"] = False
        state["last_text"] = ""
        event = "end"
    else:
        state["last_text"] = text

    state["session_id"] = session_id
    return state, history, _status_text(state, event), session_id or ""


def reset_session(state: dict[str, Any] | None):
    session_id = None
    chatbot_use_messages = None
    if isinstance(state, dict):
        session_id = state.get("session_id")
        chatbot_use_messages = state.get("chatbot_use_messages")
    if session_id:
        try:
            requests.post(
                f"{STREAMING_AUDIO_URL.rstrip('/')}/session/reset",
                json={"session_id": session_id},
                timeout=10,
            )
        except Exception:
            pass
    new_state = _init_state(chatbot_use_messages)
    return new_state, [], "ステータス: 待機中", ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Streaming Audio Client") as demo:
        gr.Markdown("# 音声ストリーミング クライアント")
        gr.Markdown(f"接続先: `{STREAMING_AUDIO_URL}`")

        status = gr.Markdown("ステータス: 待機中")
        session_id_box = gr.Textbox(label="session_id", interactive=False)

        mic = gr.Audio(
            label="マイク入力",
            sources=["microphone"],
            type="numpy",
            streaming=True,
        )

        chatbot, chatbot_use_messages = _create_chatbot()
        if GRADIO_DEBUG:
            fmt = "messages" if chatbot_use_messages else "tuples"
            print(f"[gradio-client] chatbot format: {fmt}", flush=True)

        state = gr.State(value=_init_state(chatbot_use_messages))
        mic.stream(
            fn=stream_handler,
            inputs=[state, mic, chatbot],
            outputs=[state, chatbot, status, session_id_box],
            stream_every=STREAM_EVERY,
        )

        reset_btn = gr.Button("セッションリセット", variant="secondary")
        reset_btn.click(
            fn=reset_session,
            inputs=[state],
            outputs=[state, chatbot, status, session_id_box],
        )

    return demo


if __name__ == "__main__":
    _wait_for_audio_service()
    ui = build_ui()
    print(f"GRADIO_LOCAL_URL={_local_url()}", flush=True)
    launch_kwargs: dict[str, Any] = {}
    if LAUNCH_SUPPORTS_PREVENT_THREAD_LOCK:
        launch_kwargs["prevent_thread_lock"] = True

    launch_result = ui.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        **launch_kwargs,
    )
    if GRADIO_SHARE:
        share_url = _wait_for_share_url(launch_result, ui)
        if share_url:
            print(f"GRADIO_SHARE_URL={share_url}", flush=True)
        else:
            print("GRADIO_SHARE_URL=not_available", flush=True)
    _block_forever(ui, bool(launch_kwargs.get("prevent_thread_lock")))

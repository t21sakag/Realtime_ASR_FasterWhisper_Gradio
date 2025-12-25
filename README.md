# Realtime_ASR_FasterWhisper_Gradio

Gradio UI から分離したリアルタイム音声処理サービスです。VAD による区間検出と ASR を FastAPI で提供します。

## 起動

```bash
docker compose -f docker-compose.yml up --build
```

## UI 連携

```bash
export STREAMING_AUDIO_URL=http://localhost:19100
```

Docker 連携では UI 側の環境変数に `STREAMING_AUDIO_URL` を渡してください。

## サンプル UI

音声認識の入力/終了を確認する Gradio サンプルは `gradio_client` にあります。

```bash
python gradio_client/app.py
```

Docker で音声サービスとサンプル UI を別コンテナで立ち上げる場合はこちらを使います。

```bash
docker compose -f gradio_client/docker-compose.yml up --build
```

## システム図
<img width="1541" height="842" alt="image" src="https://github.com/user-attachments/assets/a68effff-2e34-4981-9964-acdc6690695d" />

## データフロー図
<img width="1898" height="745" alt="image" src="https://github.com/user-attachments/assets/dca9fbc8-e9a8-4a90-bcae-66f327d9832f" />

## セッション管理図
<img width="1687" height="920" alt="image" src="https://github.com/user-attachments/assets/78b4c299-f27a-4ede-bc0d-e5ce590ec143" />


## 主要エンドポイント

- `GET /health`
- `POST /stream` (multipart: file, session_id, end_of_stream)
- `POST /vad` (multipart: file)
- `POST /transcribe` (multipart: file, use_vad)
- `POST /session/reset`

`/stream` は `end_of_stream=1` を付けると強制的に確定できます。`/transcribe` は `use_vad=1` で VAD 区間のみ文字起こしします。

## 環境変数

- `ASR_URL` (任意): 既存 ASR サービスへの転送先
- `WHISPER_MODEL_ID`
- `LANGUAGE`
- `VAD_THRESHOLD`
- `MIN_SPEECH_DURATION_MS`
- `MIN_SILENCE_DURATION_MS`
- `SPEECH_PAD_MS`
- `MAX_CONTINUOUS_SPEECH_S`
- `PRE_ROLL_MS`
- `STREAMING_AUDIO_DEBUG` (VAD の検出状況をログ出力)

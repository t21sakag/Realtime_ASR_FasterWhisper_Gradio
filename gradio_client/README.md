# Gradio サンプルクライアント

音声ストリーミング API に対して、`Audio.stream` で認識結果を確認するための最小 UI です。  
LLM は統合せず、発話終了時に自分の発話のみがチャット履歴に追加されます。

## 前提

- ローカル起動時は音声サービスが `http://localhost:19100` で起動していること
- `STREAMING_AUDIO_URL` を変更する場合は環境変数で指定

## ローカル起動

```bash
export STREAMING_AUDIO_URL=http://localhost:19100
python gradio_client/app.py
```

## Docker 起動 (音声サービスと別コンテナ)

```bash
docker compose -f gradio_client/docker-compose.yml up --build
```

この compose は `audio-stream` と `gradio-client` を別コンテナで起動します。外部の音声サービスを使う場合は `STREAMING_AUDIO_URL` を上書きし、`gradio-client` のみ起動してください。

```bash
STREAMING_AUDIO_URL=http://localhost:19100 \
  docker compose -f gradio_client/docker-compose.yml up --build --no-deps gradio-client
```

Linux の場合に `host.docker.internal` を使えるよう `extra_hosts` を同梱しています。

## HTTPS 共有リンク (share)

ブラウザのマイク許可が必要な場合は `GRADIO_SHARE=true` を指定します。

```bash
GRADIO_SHARE=true \
  docker compose -f gradio_client/docker-compose.yml up --build
```

起動ログに `GRADIO_SHARE_URL=...` が出力されるので、その URL を開いてください。

通信エラーなどの詳細を出したい場合は `GRADIO_DEBUG=true` を指定します。
`GRADIO_DEBUG` は `0/1` で指定してください（空文字だと Gradio 側が例外になります）。
`GRADIO_SHARE_URL` が出ない場合は `GRADIO_SHARE_WAIT` で待機時間を増やせます（秒）。

## チャット履歴の表示形式

発話ごとに別メッセージで表示したい場合は `CHATBOT_FORMAT=tuples` を使います（既定）。
messages 形式を使いたい場合は `CHATBOT_FORMAT=messages` を指定してください。

## 起動待ち

`audio-stream` の起動完了を待ってから UI を起動したい場合は `WAIT_FOR_AUDIO=true` を使います。
既定で有効化しており、`/health` が応答するまで最大 300 秒待機します。

## UI 挙動

- 発話開始/終了のステータス表示
- 発話終了時にチャット履歴へ user メッセージのみ追加

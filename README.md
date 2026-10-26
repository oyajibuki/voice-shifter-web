---
title: Voice Shifter
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ボイスシフター Web

録音した声を AI（[Seed-VC](https://github.com/Plachtaa/seed-vc), GPL-3.0）で別人の声に変換する。
iOS 版「ボイスシフター」（82.iosapp/50.VoiceChanger）と聞き比べるための Web 版。

- FastAPI + Docker（Hugging Face Spaces の無料 CPU 枠）
- モデル: seed-uvit-whisper-small-wavenet（v1、22.05kHz、ゼロショット）
- 変換は1件ずつ順番に処理。音声は 30 分で削除
- `voices/` のお手本の声は Seed-VC リポジトリの examples/reference から仮に借りたもの。公開前に権利のはっきりした声に差し替える

ローカルで動かす:

```
docker build -t voice-shifter . && docker run -p 7860:7860 voice-shifter
```

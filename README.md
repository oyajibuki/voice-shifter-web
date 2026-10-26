# ボイスシフター Web

録音した声を AI（[Seed-VC](https://github.com/Plachtaa/seed-vc), GPL-3.0）で別人の声に変換する。
iOS 版「ボイスシフター」（82.iosapp/50.VoiceChanger）と聞き比べるための Web 版。

- FastAPI + Docker、GCP Cloud Run（プロジェクト `voice-shifter`、東京リージョン）
- モデル: seed-uvit-whisper-small-wavenet（v1、22.05kHz、ゼロショット）。声の高さはお手本の声に寄る
- 変換は1件ずつ順番に処理。音声は 30 分で削除
- `voices/` のお手本の声は Seed-VC リポジトリの examples/reference から仮に借りたもの。公開前に権利のはっきりした声に差し替える

## Cloud Run での動かし方

課金は「リクエスト中だけ CPU」方式（使った分だけ）。最大1台、使われないと0台に戻る。
リクエストが無いと CPU が止められて裏の変換が進まないので、画面は変換の間 `POST /api/jobs/{id}/run` を開いたままにする。

デプロイ（`web/` で実行）:

```
gcloud builds submit --project voice-shifter --region asia-northeast1 \
  --tag asia-northeast1-docker.pkg.dev/voice-shifter/voice-shifter/app:vN \
  --timeout=3600 --machine-type=e2-highcpu-8 .
gcloud run deploy voice-shifter --project voice-shifter --region asia-northeast1 \
  --image asia-northeast1-docker.pkg.dev/voice-shifter/voice-shifter/app:vN
```

サービスの設定（初回に指定済み。上の deploy では引き継がれる）:
`--cpu 4 --memory 8Gi --max-instances 1 --min-instances 0 --concurrency 20 --timeout 3600 --cpu-boost --allow-unauthenticated`

ローカルで動かす:

```
docker build -t voice-shifter . && docker run -p 7860:7860 voice-shifter
```

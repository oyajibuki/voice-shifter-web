FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Seed-VC 本体（GPL-3.0）。動作確認したコミットに固定する
RUN git clone https://github.com/Plachtaa/seed-vc.git /app/seed-vc \
    && cd /app/seed-vc && git checkout 51383ef

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

ENV SEED_VC_DIR=/app/seed-vc \
    HF_HUB_CACHE=/app/seed-vc/checkpoints/hf_cache \
    PYTHONUNBUFFERED=1

# モデル（約 1GB）をビルド時に落としておく。起動のたびにダウンロードしないため
COPY converter.py .
RUN python -c "import converter; converter.load()"

# Hugging Face Spaces は uid 1000 で動かす
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
COPY --chown=user . .

# Cloud Run は PORT を渡してくる（無いときは 7860）
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

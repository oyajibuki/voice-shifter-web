"""ボイスシフター Web: 録音した声を Seed-VC で別人の声に変換する。

無料 CPU 枠で動かすので、変換は1件ずつ順番に処理する（同時にやると両方遅くなる）。
音声は変換のためだけに受け取り、結果を渡したら（または 30 分たったら）消す。
"""
import json
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

import converter

ROOT = Path(__file__).parent
WORK = Path("/tmp/voice-shifter")
WORK.mkdir(parents=True, exist_ok=True)
VOICES = json.loads((ROOT / "voices" / "voices.json").read_text())
MAX_SECONDS = 180
MAX_UPLOAD = 50 * 1024 * 1024
KEEP_SECONDS = 30 * 60

app = FastAPI()
jobs: dict[str, dict] = {}
pending: "queue.Queue[str]" = queue.Queue()
order: list[str] = []  # 順番待ちの表示用
lock = threading.Lock()


def to_wav(src: Path, dst: Path, limit: int | None = None):
    """どんな形式でも 22.05kHz モノラル wav にそろえる。小さく録れた声も聞きやすい大きさにする"""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "22050",
           "-af", "loudnorm=I=-18:TP=-1.5:LRA=11"]
    if limit:
        cmd += ["-t", str(limit)]
    subprocess.run(cmd + [str(dst)], check=True, timeout=120)


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def worker():
    converter.load()
    while True:
        job_id = pending.get()
        job = jobs.get(job_id)
        if job is None:
            continue
        with lock:
            if job_id in order:
                order.remove(job_id)
        job["status"] = "running"
        try:
            def report(p):
                job["progress"] = round(float(p), 3)

            wave = converter.convert(job["source"], job["reference"], steps=job["steps"], progress=report)
            peak = float(np.max(np.abs(wave))) or 1.0
            wave = wave / peak * 0.89  # 割れない範囲でいっぱいに
            out = job["dir"] / "converted.wav"
            sf.write(str(out), wave, converter.SR)
            job["output"] = out
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 — 画面に理由を出すため
            job["status"] = "error"
            job["error"] = f"変換に失敗しました（{type(e).__name__}）。別のファイルで試してください。"
            print("convert error:", repr(e), flush=True)


def janitor():
    """古い作業ファイルを消す"""
    while True:
        time.sleep(300)
        now = time.time()
        for job_id, job in list(jobs.items()):
            if now - job["created"] > KEEP_SECONDS and job["status"] in ("done", "error"):
                shutil.rmtree(job["dir"], ignore_errors=True)
                jobs.pop(job_id, None)


threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=janitor, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text()


@app.get("/api/voices")
def voices():
    return [{"id": v["id"], "emoji": v["emoji"], "name": v["name"]} for v in VOICES]


@app.post("/api/convert")
async def convert(source: UploadFile = File(...), voice: str = Form(...),
                  reference: UploadFile | None = File(None), steps: int = Form(25)):
    job_id = uuid.uuid4().hex
    d = WORK / job_id
    d.mkdir()
    try:
        raw = d / ("source-" + Path(source.filename or "a.bin").name)
        data = await source.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "ファイルが大きすぎます（50MB まで）。")
        raw.write_bytes(data)
        src = d / "source.wav"
        try:
            to_wav(raw, src, limit=MAX_SECONDS)
        except Exception:
            raise HTTPException(400, "音声として読み込めませんでした。mp3 / m4a / wav などのファイルを選んでください。")
        if duration(src) < 1:
            raise HTTPException(400, "音声が短すぎます。1 秒以上話してください。")

        if voice == "custom":
            if reference is None:
                raise HTTPException(400, "お手本の声のファイルを選んでください。")
            ref_raw = d / ("ref-" + Path(reference.filename or "r.bin").name)
            ref_raw.write_bytes(await reference.read())
            ref = d / "reference.wav"
            try:
                to_wav(ref_raw, ref, limit=25)
            except Exception:
                raise HTTPException(400, "お手本の声を読み込めませんでした。")
        else:
            match = next((v for v in VOICES if v["id"] == voice), None)
            if match is None:
                raise HTTPException(400, "声を選び直してください。")
            ref = ROOT / "voices" / match["file"]
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise

    jobs[job_id] = {
        "status": "queued", "progress": 0.0, "created": time.time(), "dir": d,
        "source": str(src), "reference": str(ref), "steps": max(5, min(int(steps), 50)),
    }
    with lock:
        order.append(job_id)
    pending.put(job_id)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "変換が見つかりません。もう一度変換してください。")
    with lock:
        ahead = order.index(job_id) if job_id in order else 0
    return {"status": job["status"], "progress": job["progress"], "ahead": ahead, "error": job.get("error")}


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    job = jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "変換後の音声がありません。")
    return FileResponse(job["output"], media_type="audio/wav", filename="voice-shifter.wav")

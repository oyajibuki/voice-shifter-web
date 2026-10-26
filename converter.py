"""Seed-VC（v1 / seed-uvit-whisper-small-wavenet）で声を変換する。

Seed-VC の inference.py を元に、次の点だけ変えている:
- モデルは起動時に1回だけ読み込み、変換のたびに使い回す
- CPU（Hugging Face Spaces の無料枠）前提。Whisper も float32 で動かす（CPU の float16 は遅い）
- 進み具合をコールバックで返す
"""
import os
import sys

SEED_VC_DIR = os.environ.get("SEED_VC_DIR", "/app/seed-vc")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(SEED_VC_DIR, "checkpoints", "hf_cache"))
sys.path.insert(0, SEED_VC_DIR)

import numpy as np
import torch
import torchaudio
import librosa
import yaml

device = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))

SR = 22050
HOP = 256
_models = None


def load():
    """モデルを読み込む（初回だけ。Docker のビルド時にも呼んでダウンロードを済ませておく）"""
    global _models
    if _models is not None:
        return _models
    cwd = os.getcwd()
    os.chdir(SEED_VC_DIR)  # Seed-VC は ./checkpoints に相対パスで保存する
    try:
        from modules.commons import recursive_munch, build_model, load_checkpoint
        from hf_utils import load_custom_model_from_hf
        from modules.campplus.DTDNN import CAMPPlus
        from modules.bigvgan import bigvgan
        from modules.audio import mel_spectrogram
        from transformers import AutoFeatureExtractor, WhisperModel

        ckpt, cfg_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        )
        config = yaml.safe_load(open(cfg_path, "r"))
        params = recursive_munch(config["model_params"])
        params.dit_type = "DiT"
        model = build_model(params, stage="DiT")
        model, _, _, _ = load_checkpoint(model, None, ckpt, load_only_params=True,
                                         ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        campplus_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
        campplus.eval().to(device)

        vocoder = bigvgan.BigVGAN.from_pretrained(params.vocoder.name, use_cuda_kernel=False)
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)

        whisper_name = params.speech_tokenizer.name
        whisper = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float32).to(device)
        del whisper.decoder
        whisper_fe = AutoFeatureExtractor.from_pretrained(whisper_name)

        spect = config["preprocess_params"]["spect_params"]
        mel_args = {
            "n_fft": spect["n_fft"],
            "win_size": spect["win_length"],
            "hop_size": spect["hop_length"],
            "num_mels": spect["n_mels"],
            "sampling_rate": SR,
            "fmin": spect.get("fmin", 0),
            "fmax": None if spect.get("fmax", "None") == "None" else 8000,
            "center": False,
        }
    finally:
        os.chdir(cwd)

    def semantic(waves_16k):
        inputs = whisper_fe([waves_16k.squeeze(0).cpu().numpy()], return_tensors="pt",
                            return_attention_mask=True)
        feats = whisper._mask_input_features(inputs.input_features,
                                             attention_mask=inputs.attention_mask).to(device)
        with torch.no_grad():
            out = whisper.encoder(feats, head_mask=None, output_attentions=False,
                                  output_hidden_states=False, return_dict=True)
        s = out.last_hidden_state.to(torch.float32)
        return s[:, : waves_16k.size(-1) // 320 + 1]

    _models = {
        "model": model,
        "campplus": campplus,
        "vocoder": vocoder,
        "semantic": semantic,
        "to_mel": lambda x: mel_spectrogram(x, **mel_args),
    }
    return _models


def _semantic_long(semantic, waves_16k):
    """Whisper は 30 秒までしか一度に読めないので、5 秒ずつ重ねて区切る（inference.py と同じ）"""
    if waves_16k.size(-1) <= 16000 * 30:
        return semantic(waves_16k)
    overlap = 5
    parts, buffer, done = [], None, 0
    while done < waves_16k.size(-1):
        if buffer is None:
            chunk = waves_16k[:, done:done + 16000 * 30]
        else:
            chunk = torch.cat([buffer, waves_16k[:, done:done + 16000 * (30 - overlap)]], dim=-1)
        s = semantic(chunk)
        parts.append(s if done == 0 else s[:, 50 * overlap:])
        buffer = chunk[:, -16000 * overlap:]
        done += 30 * 16000 if done == 0 else chunk.size(-1) - 16000 * overlap
    return torch.cat(parts, dim=1)


def _crossfade(a, b, n):
    fade_out = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, n)) ** 2
    m = min(len(b), n)
    b[:m] = b[:m] * fade_in[:m] + (a[-n:] * fade_out)[:m]
    return b


@torch.no_grad()
def convert(source_path, reference_path, steps=25, cfg_rate=0.7, progress=lambda p: None):
    """source の話し方のまま、reference の声色に変える。22.05kHz の numpy 配列を返す"""
    m = load()
    model, to_mel = m["model"], m["to_mel"]

    source = librosa.load(source_path, sr=SR)[0]
    ref = librosa.load(reference_path, sr=SR)[0][: SR * 25]  # お手本は 25 秒まで使う
    source = torch.tensor(source).unsqueeze(0).float().to(device)
    ref = torch.tensor(ref).unsqueeze(0).float().to(device)
    progress(0.02)

    src_16k = torchaudio.functional.resample(source, SR, 16000)
    ref_16k = torchaudio.functional.resample(ref, SR, 16000)
    s_alt = _semantic_long(m["semantic"], src_16k)
    s_ori = m["semantic"](ref_16k)
    progress(0.1)

    mel = to_mel(source.float())
    mel2 = to_mel(ref.float())
    target_len = torch.LongTensor([mel.size(2)]).to(device)
    target2_len = torch.LongTensor([mel2.size(2)]).to(device)

    fbank = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    style = m["campplus"](fbank.unsqueeze(0))

    cond = model.length_regulator(s_alt, ylens=target_len, n_quantizers=3, f0=None)[0]
    prompt = model.length_regulator(s_ori, ylens=target2_len, n_quantizers=3, f0=None)[0]

    max_window = SR // HOP * 30 - mel2.size(2)
    overlap_frames = 16
    overlap_wave = overlap_frames * HOP
    chunks, prev, done = [], None, 0
    total = cond.size(1)
    while done < total:
        chunk = cond[:, done:done + max_window]
        last = done + max_window >= total
        cat = torch.cat([prompt, chunk], dim=1)
        target = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(device),
                                     mel2, style, None, steps, inference_cfg_rate=cfg_rate)
        target = target[:, :, mel2.size(-1):]
        wave = m["vocoder"](target.float()).squeeze()[None, :]
        if prev is None:
            if last:
                chunks.append(wave[0].cpu().numpy())
                break
            chunks.append(wave[0, :-overlap_wave].cpu().numpy())
        elif last:
            chunks.append(_crossfade(prev, wave[0].cpu().numpy(), overlap_wave))
            break
        else:
            chunks.append(_crossfade(prev, wave[0, :-overlap_wave].cpu().numpy(), overlap_wave))
        prev = wave[0, -overlap_wave:].cpu().numpy()
        done += target.size(2) - overlap_frames
        progress(0.1 + 0.88 * min(1.0, done / total))
    progress(1.0)
    return np.concatenate(chunks)

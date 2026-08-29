"""
Voice conversion service (port 8002) — Seed-VC backend.

Wraps Plachtaa/seed-vc (github.com/Plachtaa/seed-vc) offline inference in a FastAPI server
per the synthetic_pipeline/CONTRACT.md contract:

  GET  /health   -> {"status": "ok", "backend": "seed-vc"}
  POST /convert  -> multipart form-data (source_audio, reference_audio) -> raw WAV bytes,
                     mono 24000 Hz, Content-Type: audio/wav
"""
import io
import logging
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import yaml
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_VC_DIR = os.path.join(BASE_DIR, "seed-vc")
sys.path.insert(0, SEED_VC_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vc_service")

# seed-vc's hf_utils.py caches checkpoints to a path relative to cwd ("./checkpoints"),
# so run everything with cwd = the seed-vc clone.
os.chdir(SEED_VC_DIR)

from modules.commons import build_model, load_checkpoint, recursive_munch  # noqa: E402
from hf_utils import load_custom_model_from_hf  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16 = torch.cuda.is_available()
OUTPUT_SR = 24000  # contract-mandated output sample rate

app = FastAPI()

STATE = {}


def load_models():
    log.info("Loading Seed-VC models on device=%s fp16=%s", DEVICE, FP16)
    t0 = time.time()

    dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    )
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    model, _, _, _ = load_checkpoint(
        model, None, dit_checkpoint_path,
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(DEVICE)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(DEVICE)

    # vocoder: bigvgan for this preset
    from modules.bigvgan import bigvgan

    bigvgan_name = model_params.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(DEVICE)
    vocoder_fn = bigvgan_model

    # speech tokenizer: whisper (per this preset)
    from transformers import AutoFeatureExtractor, WhisperModel

    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(DEVICE)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
        ).to(DEVICE)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
        return S_ori

    mel_fn_args = {
        "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
        "win_size": config["preprocess_params"]["spect_params"]["win_length"],
        "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
        "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
        "sampling_rate": sr,
        "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
        "fmax": None if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000,
        "center": False,
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    STATE.update(
        model=model,
        semantic_fn=semantic_fn,
        vocoder_fn=vocoder_fn,
        campplus_model=campplus_model,
        to_mel=to_mel,
        sr=sr,
        hop_length=hop_length,
        overlap_frame_len=16,
    )
    log.info("Seed-VC models loaded in %.1fs", time.time() - t0)


@app.on_event("startup")
def _startup():
    try:
        load_models()
    except Exception:
        log.error("Model loading failed:\n%s", traceback.format_exc())
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "backend": "seed-vc"}


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
def run_voice_conversion(source_path, ref_path, diffusion_steps=10, length_adjust=1.0, inference_cfg_rate=0.7):
    model = STATE["model"]
    semantic_fn = STATE["semantic_fn"]
    vocoder_fn = STATE["vocoder_fn"]
    campplus_model = STATE["campplus_model"]
    mel_fn = STATE["to_mel"]
    sr = STATE["sr"]
    hop_length = STATE["hop_length"]
    overlap_frame_len = STATE["overlap_frame_len"]

    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(ref_path, sr=sr)[0]

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(DEVICE)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(DEVICE)

    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time : traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1,
                )
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(DEVICE).float())
    mel2 = mel_fn(ref_audio.to(DEVICE).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    cond, _, _, _, _ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)
    prompt_condition, _, _, _, _ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if FP16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = vocoder_fn(vc_target.float())[0]
        if vc_wave.ndim == 1:
            vc_wave = vc_wave.unsqueeze(0)

        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    full_wave = np.concatenate(generated_wave_chunks)
    return full_wave, sr


@app.post("/convert")
async def convert(source_audio: UploadFile = File(...), reference_audio: UploadFile = File(...)):
    src_tmp = ref_tmp = None
    try:
        src_bytes = await source_audio.read()
        ref_bytes = await reference_audio.read()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(src_bytes)
            src_tmp = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(ref_bytes)
            ref_tmp = f.name

        t0 = time.time()
        wave, model_sr = run_voice_conversion(src_tmp, ref_tmp)
        log.info("Converted %s + %s in %.2fs (output %.2fs audio)",
                  source_audio.filename, reference_audio.filename, time.time() - t0, len(wave) / model_sr)

        # Resample to contract-mandated 24000 Hz mono
        if model_sr != OUTPUT_SR:
            wave = librosa.resample(wave.astype(np.float32), orig_sr=model_sr, target_sr=OUTPUT_SR)

        buf = io.BytesIO()
        sf.write(buf, wave, OUTPUT_SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        log.error("convert failed:\n%s", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for p in (src_tmp, ref_tmp):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8002)))

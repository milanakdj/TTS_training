"""TTS service — port 8001.

Wraps three Nepali-capable TTS engines behind one HTTP API:
- mms:    facebook/mms-tts-npi (VITS, transformers)
- parler: ai4bharat/indic-parler-tts-pretrained (parler-tts package)
- edge:   edge-tts (Microsoft cloud TTS, no GPU/model download)

See CONTRACT.md (synthetic_pipeline/CONTRACT.md) for the full API contract.
"""

import asyncio
import io
import logging
import sys
import traceback

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("tts_service")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_SR = 24000

DEFAULT_PARLER_DESCRIPTION = (
    "A female Nepali speaker with a clear, moderate-pitched voice speaks at a "
    "normal pace with minimal background noise, in a neutral tone."
)

DEFAULT_EDGE_VOICE = "ne-NP-HemkalaNeural"
ALLOWED_EDGE_VOICES = {"ne-NP-HemkalaNeural", "ne-NP-SagarNeural"}

# CONTRACT.md specifies facebook/mms-tts-npi for the "mms" engine. That repo id does not
# exist: neither on the HF Hub (404, verified via API) nor in Meta's original fairseq MMS-TTS
# release (dl.fbaipublicfiles.com/mms/tts/npi.tar.gz and nep.tar.gz both 403 = not present).
# MMS-TTS's public language set (~1143 langs, see facebook/mms-tts README) does not include
# Nepali at all. Falling back to facebook/mms-tts-hin (Hindi) — same VITS/transformers plumbing
# the contract asks for, same Devanagari script family, closest available MMS checkpoint. Feeding
# it Nepali text will mispronounce Nepali-specific sounds/conjuncts since it's phonologically a
# Hindi model; see final report for a quality note.
MMS_MODEL_ID = "facebook/mms-tts-hin"

app = FastAPI(title="TTS service")

_state = {"mms": None, "parler": None}


class SynthesizeRequest(BaseModel):
    text: str
    engine: str
    speaker_description: str | None = None
    voice: str | None = None


def load_mms():
    if _state["mms"] is not None:
        return _state["mms"]
    log.info("Loading MMS-TTS model %s on %s ...", MMS_MODEL_ID, DEVICE)
    from transformers import VitsModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL_ID)
    # VITS's flow/duration-predictor ops mix float32 buffers with the model's linear layers;
    # running it in fp16 throws "mat1 and mat2 must have the same dtype" (Float vs Half). The
    # model is tiny (~36M params) so fp32 is still fast — keep it in fp32 regardless of DEVICE.
    model = VitsModel.from_pretrained(MMS_MODEL_ID, torch_dtype=torch.float32)
    model = model.to(DEVICE)
    model.eval()
    _state["mms"] = (model, tokenizer)
    log.info("MMS-TTS loaded. native sample rate=%s", model.config.sampling_rate)
    return _state["mms"]


def load_parler():
    if _state["parler"] is not None:
        return _state["parler"]
    log.info("Loading Parler-TTS model ai4bharat/indic-parler-tts-pretrained on %s ...", DEVICE)
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts-pretrained", torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts-pretrained")
    # description tokenizer is the model's own text_encoder tokenizer; same tokenizer works fine
    desc_tokenizer = tokenizer
    _state["parler"] = (model, tokenizer, desc_tokenizer)
    sr = model.config.sampling_rate
    log.info("Parler-TTS loaded. native sample rate=%s", sr)
    return _state["parler"]


def resample_to_target(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] < audio.shape[-1] else audio.mean(axis=-1)
    if orig_sr == TARGET_SR:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synth_mms(text: str) -> tuple[np.ndarray, int]:
    model, tokenizer = load_mms()
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**inputs).waveform
    audio = output[0].to(torch.float32).cpu().numpy()
    sr = model.config.sampling_rate
    return audio, sr


def synth_parler(text: str, description: str | None) -> tuple[np.ndarray, int]:
    model, tokenizer, desc_tokenizer = load_parler()
    desc = description or DEFAULT_PARLER_DESCRIPTION
    input_ids = desc_tokenizer(desc, return_tensors="pt").input_ids.to(DEVICE)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids, prompt_input_ids=prompt_input_ids
        )
    audio = generation.to(torch.float32).cpu().numpy().squeeze()
    sr = model.config.sampling_rate
    return audio, sr


async def synth_edge(text: str, voice: str | None) -> tuple[np.ndarray, int]:
    import edge_tts

    v = voice or DEFAULT_EDGE_VOICE
    if v not in ALLOWED_EDGE_VOICES:
        raise ValueError(f"unknown edge voice '{v}', must be one of {sorted(ALLOWED_EDGE_VOICES)}")

    communicate = edge_tts.Communicate(text, v)
    mp3_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
    if not mp3_bytes:
        raise RuntimeError("edge-tts returned no audio data")

    import librosa

    audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
    return audio, sr


@app.get("/health")
def health():
    return {"status": "ok", "engines": ["mms", "parler", "edge"]}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    try:
        if req.engine not in ("mms", "parler", "edge"):
            return JSONResponse(
                status_code=400,
                content={"error": f"unknown engine '{req.engine}', must be 'mms', 'parler' or 'edge'"},
            )
        if not req.text or not req.text.strip():
            return JSONResponse(status_code=400, content={"error": "text must not be empty"})

        log.info(
            "synthesize engine=%s text=%r desc=%r voice=%r",
            req.engine, req.text, req.speaker_description, req.voice,
        )

        if req.engine == "mms":
            audio, sr = await asyncio.to_thread(synth_mms, req.text)
        elif req.engine == "parler":
            audio, sr = await asyncio.to_thread(synth_parler, req.text, req.speaker_description)
        else:
            audio, sr = await synth_edge(req.text, req.voice)

        audio = resample_to_target(audio, sr)
        data = wav_bytes(audio, TARGET_SR)
        log.info(
            "synthesize OK engine=%s native_sr=%s out_samples=%d out_sr=%d bytes=%d",
            req.engine, sr, len(audio), TARGET_SR, len(data),
        )
        return Response(content=data, media_type="audio/wav")
    except Exception as e:
        log.error("synthesize FAILED: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))

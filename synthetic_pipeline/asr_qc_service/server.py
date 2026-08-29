import io
import logging
import re
import string
import time
import unicodedata

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import jiwer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)
log = logging.getLogger("asr_qc_service")

MODEL_NAME = "large-v3-turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
CER_PASS_THRESHOLD = 0.3

app = FastAPI(title="ASR QC Service", version="1.0")

log.info("Loading faster-whisper model %s on %s (%s)...", MODEL_NAME, DEVICE, COMPUTE_TYPE)
_t0 = time.time()
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
log.info("Model loaded in %.1fs", time.time() - _t0)

# Punctuation set: standard ASCII punctuation plus common Devanagari punctuation
# (danda U+0964, double danda U+0965) not covered by string.punctuation.
_EXTRA_PUNCT = "।॥‘’“”–—…"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + _EXTRA_PUNCT)


def normalize_text(text: str) -> str:
    """Unicode NFC normalize, strip punctuation, collapse whitespace, lowercase."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.translate(_PUNCT_TABLE)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def transcribe_bytes(audio_bytes: bytes, filename: str = "audio.wav"):
    """Run faster-whisper transcription on raw audio bytes via an in-memory buffer."""
    buf = io.BytesIO(audio_bytes)
    buf.name = filename
    segments, info = model.transcribe(buf, beam_size=1, language="ne", vad_filter=False)
    text = "".join(seg.text for seg in segments).strip()
    language = info.language
    return text, language


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        log.info("Received /transcribe request: filename=%s size=%d bytes", audio.filename, len(audio_bytes))
        text, language = transcribe_bytes(audio_bytes, audio.filename or "audio.wav")
        log.info("Transcribed: lang=%s text=%r", language, text)
        return {"text": text, "language": language}
    except Exception as e:
        log.exception("Error in /transcribe")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/qc")
async def qc(audio: UploadFile = File(...), expected_text: str = Form(...)):
    try:
        audio_bytes = await audio.read()
        log.info(
            "Received /qc request: filename=%s size=%d bytes expected_text=%r",
            audio.filename, len(audio_bytes), expected_text,
        )
        transcript, language = transcribe_bytes(audio_bytes, audio.filename or "audio.wav")

        norm_transcript = normalize_text(transcript)
        norm_expected = normalize_text(expected_text)

        # jiwer >=3 API: process_characters / process_words
        cer_result = jiwer.process_characters(norm_expected, norm_transcript)
        wer_result = jiwer.process_words(norm_expected, norm_transcript)
        cer = float(cer_result.cer)
        wer = float(wer_result.wer)
        passed = cer < CER_PASS_THRESHOLD

        log.info(
            "QC result: transcript=%r cer=%.4f wer=%.4f pass=%s (lang=%s)",
            transcript, cer, wer, passed, language,
        )
        return {"transcript": transcript, "cer": cer, "wer": wer, "pass": passed}
    except Exception as e:
        log.exception("Error in /qc")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8003)))

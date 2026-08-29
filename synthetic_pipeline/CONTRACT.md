# Synthetic Nepali data pipeline — service contract

Three independent local HTTP services, each in its own venv (avoid dependency conflicts),
each left running in the background on its assigned port. An orchestrator script (built
separately, not by you) calls all three over HTTP to run text -> synthetic audio -> QC.

General rules for every service:
- Own venv at `<service_dir>/.venv` (`python3 -m venv .venv`). Do not touch other services' venvs.
- FastAPI + uvicorn. Bind `0.0.0.0` on the assigned port.
- Start it with `nohup <service_dir>/.venv/bin/python server.py > <service_dir>/server.log 2>&1 & disown`
  so it survives after your task ends. Confirm it's actually up with `curl localhost:<port>/health`
  before you finish, and paste the curl output in your final report along with the PID.
- GPU: CUDA is available (H100 80GB in this session). Use it (`cuda`), fp16/bf16 where the
  model supports it, for speed. VRAM is not a constrained resource here — do not skip GPU use
  to "be safe," there's plenty of headroom for these three services to run concurrently.
- HF_TOKEN is already set in the environment — use it for any gated/private HF downloads.
- If a request errors, return a proper HTTP error with a JSON `{"error": "..."}` body, don't crash
  the server.
- Log enough to `server.log` that a failure is diagnosable without re-running.

## TTS service — port 8001, dir `tts_service/`

Wraps three Nepali-capable TTS engines behind one API:
- `mms` — `facebook/mms-tts-npi` (Meta MMS-TTS, VITS architecture) via `transformers` (`VitsModel` +
  `VitsTokenizer` / `AutoTokenizer`). Single speaker.
- `parler` — `ai4bharat/indic-parler-tts-pretrained` via the `parler-tts` package
  (`pip install git+https://github.com/huggingface/parler-tts.git`). Takes a natural-language
  speaker/style description string alongside the text. Default description if none given:
  `"A female Nepali speaker with a clear, moderate-pitched voice speaks at a normal pace with
  minimal background noise, in a neutral tone."`
- `edge` — Microsoft Edge's cloud TTS via the `edge-tts` package (`pip install edge-tts`). No model
  download, no GPU, just a network call — this one's cheap and quick to wire up, do it first if the
  other two are still installing. Default voice `ne-NP-HemkalaNeural` (female); accept an optional
  `voice` param in the request to allow `ne-NP-SagarNeural` (male) too. `edge_tts.Communicate`
  natively outputs MP3 — decode and resample to the standard WAV contract below like the other
  engines. Note in your report: this is a proprietary cloud service reached via an unofficial
  client, not an open-weight model — flag clearly if it's slow, rate-limited, or errors out under
  repeated calls, since that's expected behavior to know about, not a bug to chase.

Endpoints:
- `GET /health` -> `{"status": "ok", "engines": ["mms", "parler", "edge"]}`
- `POST /synthesize` — JSON body `{"text": str, "engine": "mms"|"parler"|"edge", "speaker_description": str|null, "voice": str|null}`
  -> response body is raw WAV bytes, `Content-Type: audio/wav`, **resampled to mono 24000 Hz**
  regardless of the model's native output rate. Non-200 on error with JSON error body.

## Voice conversion service — port 8002, dir `vc_service/`

Zero-shot voice conversion: takes a "content" audio clip (linguistic content to preserve) and a
"reference" audio clip (target speaker timbre), outputs the content re-rendered in the reference
speaker's voice. Content is Nepali speech — these VC models were not trained on Nepali/Devanagari,
so don't assume perfect content preservation; just get it working and note quality in your report.

Primary choice: **Seed-VC** (`github.com/Plachtaa/seed-vc`, GPL-3.0). It's an archived repo (no
more updates) but the code runs — clone it, install its requirements into the local venv, use its
inference entrypoint (offline/non-realtime mode is fine, don't need the real-time streaming path).

If Seed-VC proves too broken/unmaintained to get working within a reasonable effort (~30-45 min of
troubleshooting), fall back to **OpenVoice V2** (`myshell-ai/OpenVoiceV2` on HF /
`github.com/myshell-ai/OpenVoice`, MIT license, explicitly markets cross-lingual voice cloning) as
a substitute behind the same API contract. Whichever you end up using, say clearly which one in
your final report.

Endpoints:
- `GET /health` -> `{"status": "ok", "backend": "seed-vc"|"openvoice-v2"}`
- `POST /convert` — multipart form-data with two file fields: `source_audio` (content) and
  `reference_audio` (target timbre) -> response body is raw WAV bytes, mono 24000 Hz,
  `Content-Type: audio/wav`.

## ASR QC service — port 8003, dir `asr_qc_service/`

Whisper large-v3 (`openai/whisper-large-v3`) used ONLY to transcribe audio for comparison against
already-known ground-truth text — never as a labeling source. Use **faster-whisper**
(CTranslate2 backend, `pip install faster-whisper`) with `float16` compute type on CUDA for speed,
not the vanilla `transformers` pipeline — this needs to be fast since it'll run over many clips.

Endpoints:
- `GET /health` -> `{"status": "ok", "model": "large-v3"}`
- `POST /transcribe` — multipart form-data with file field `audio` -> JSON
  `{"text": str, "language": str}`
- `POST /qc` — multipart form-data with file field `audio` and form field `expected_text` (str)
  -> JSON `{"transcript": str, "cer": float, "wer": float, "pass": bool}`. Normalize both strings
  before comparing (Unicode NFC, strip punctuation, collapse whitespace) before computing
  character error rate (CER) and word error rate (WER) — use the `jiwer` package for this.
  `pass` = `true` when `cer < 0.3` (this threshold is a starting point, not sacred — note in your
  report if you think it should differ for Nepali).

## What to report back when done

For each endpoint: the exact curl command that proves it works, using a short real Nepali test
string (e.g. `"नेपाल एक सुन्दर देश हो।"`), and the response you got. Plus: PID, log file path,
model download sizes, total setup time, and any deviation from this contract (e.g. VC fallback).

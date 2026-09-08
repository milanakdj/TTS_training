"""
Emotion pilot (40 rows): 10 fixed Nepali sentences x 4 emotions.

Goal of the pilot: prove the mechanics of the emotional-TTS data recipe that
will later feed a finetune of `ai4bharat/indic-parler-tts` on the speaker
"Amrita", and produce real CER / throughput numbers.

Pipeline per row:
  Edge-TTS (`ne-NP-HemkalaNeural`, female, flat delivery)
    -> QC vs known text                        => cer_source / wer_source
    -> Seed-VC against an *emotional* Amrita reference clip
       (the emotion is carried by the reference, not by the text)
    -> QC vs the same known text               => cer_final / wer_final

Design decisions, deliberate:
  - The SAME 10 sentences are used for all 4 emotions so the emotions are A/B
    comparable on identical content. Edge-TTS is therefore called only 10
    times total (once per sentence), and its output is reused across emotions.
  - NO CER gate. Production used cer < 0.3 to drop rows; here every one of the
    40 rows is written to the manifest with its raw numbers, failures included,
    so we can see the distribution instead of a survivor-biased slice.
  - Reference clips rotate round-robin within each emotion folder so every
    reference clip gets used a roughly equal number of times, and the exact
    clip used is recorded per row (`vc_ref_path`).
  - `description` (the natural-language caption Indic Parler-TTS consumes) has
    3 paraphrase variants per emotion, assigned round-robin and recorded in
    `description_variant`. One memorized caption string per emotion would make
    the finetuned model brittle / overfit to that exact string.
  - Resumable: the manifest is append-only JSONL keyed by (sentence_id,
    emotion); a restart skips rows already present. Raw Edge clips and
    converted clips are also skipped if already on disk.
  - Round-robin across the 4 Seed-VC and 4 ASR-QC replicas, same as
    run_production.py.

Run: nohup .venv/bin/python -u run_emotion_pilot.py > emotion_pilot.log 2>&1 &
"""
import asyncio
import json
import os
import time

import httpx
import pandas as pd
import soundfile as sf

PIPE_ROOT = "/root/tts/TTS_training/synthetic_pipeline"
AUDIO_OUT = os.path.join(PIPE_ROOT, "audio_out", "emotion_pilot")
RAW_DIR = os.path.join(AUDIO_OUT, "raw_edge")
MANIFEST_PATH = os.path.join(PIPE_ROOT, "manifests", "emotion_pilot_manifest.jsonl")
TEXT_POOL_PATH = os.path.join(PIPE_ROOT, "manifests", "text_pool.parquet")
DONE_IDS_PATH = os.path.join(PIPE_ROOT, "manifests", "production_done_ids.txt")
EMOTIONS_ROOT = os.path.join(PIPE_ROOT, "emotions")
TIMING_PATH = os.path.join(PIPE_ROOT, "manifests", "emotion_pilot_timings.json")

TTS_URLS = ["http://localhost:8001"]
VC_URLS = ["http://localhost:8002", "http://localhost:8012", "http://localhost:8022", "http://localhost:8032"]
QC_URLS = ["http://localhost:8003", "http://localhost:8013", "http://localhost:8023", "http://localhost:8033"]

EDGE_VOICE = "ne-NP-HemkalaNeural"  # female only; a single consistent voice is the point
VC_BACKEND = "seed-vc-v1"
SPEAKER = "Amrita"
LANGUAGE = "ne"

EMOTIONS = ["angry", "sad", "happy", "neutral"]

N_SENTENCES = 10
MIN_WORDS, MAX_WORDS = 8, 20
SENTENCE_ID_FLOOR = 850_000  # high ids, untouched by the production run

CONCURRENCY = 8

# 3 paraphrase variants per emotion. Every variant: (a) starts with "Amrita"
# (the Nepali speaker token in the pretrained model -- drop it and the voice
# drifts between runs), (b) names the emotion, (c) describes pitch / pace /
# expressivity consistently with that emotion, (d) ends with a
# recording-quality clause.
DESCRIPTIONS = {
    "angry": [
        "Amrita speaks in an angry, harsh tone with a raised pitch and fast, forceful delivery. "
        "Her voice is very expressive and animated. The recording is very high quality, with her "
        "voice sounding clear and very close up.",
        "Amrita sounds angry and sharp, her pitch pushed high and her words coming out quickly and "
        "with force. The delivery is highly animated and strongly expressive. The recording quality "
        "is excellent, her voice captured cleanly and right up close.",
        "Amrita delivers the line in an angry, biting tone, pitching her voice up and driving through "
        "the words at a fast, hard-edged pace. She is intensely expressive throughout. It is a very "
        "clean, high-quality recording in which her voice sits clear and very close to the microphone.",
    ],
    "sad": [
        "Amrita speaks in a sad, sorrowful tone with a low pitch and a slow, trembling delivery. "
        "Her voice is expressive and heavy with emotion. The recording is very high quality, with her "
        "voice sounding clear and very close up.",
        "Amrita sounds sad and mournful, keeping her pitch low and letting the words come slowly, with "
        "a tremble in them. Her delivery is expressive and weighed down by feeling. The recording "
        "quality is excellent, her voice captured cleanly and right up close.",
        "Amrita reads the line with sadness and grief, her pitch sunk low and her pace dragging and "
        "unsteady. The emotion sits heavy in an expressive delivery. It is a very clean, high-quality "
        "recording in which her voice sits clear and very close to the microphone.",
    ],
    "happy": [
        "Amrita speaks in a happy, cheerful tone with a high pitch and a slightly fast, lively delivery. "
        "Her voice is very expressive and full of warmth. The recording is very high quality, with her "
        "voice sounding clear and very close up.",
        "Amrita sounds happy and bright, her pitch lifted high and her words moving along at a brisk, "
        "lively clip. The delivery is very expressive and full of warmth. The recording quality is "
        "excellent, her voice captured cleanly and right up close.",
        "Amrita delivers the line cheerfully and with delight, pitching her voice up and keeping the "
        "pace quick and buoyant. She is warmly and richly expressive. It is a very clean, high-quality "
        "recording in which her voice sits clear and very close to the microphone.",
    ],
    "neutral": [
        "Amrita speaks in a neutral, even tone at a moderate pitch and a steady, moderate pace. "
        "Her voice is clear and only slightly expressive. The recording is very high quality, with her "
        "voice sounding clear and very close up.",
        "Amrita sounds neutral and level, holding a middling pitch and an unhurried, steady pace. Her "
        "voice comes through clearly with only mild expressiveness. The recording quality is excellent, "
        "her voice captured cleanly and right up close.",
        "Amrita reads the line plainly and without inflection, staying at a moderate pitch and an even, "
        "measured pace. The delivery is clear and only lightly expressive. It is a very clean, "
        "high-quality recording in which her voice sits clear and very close to the microphone.",
    ],
}

os.makedirs(RAW_DIR, exist_ok=True)
for e in EMOTIONS:
    os.makedirs(os.path.join(AUDIO_OUT, e), exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

_manifest_lock = asyncio.Lock()
_timing_lock = asyncio.Lock()
TIMINGS = {"tts": [], "vc": [], "qc": [], "row": []}


class RoundRobin:
    def __init__(self, urls):
        self.urls = urls
        self.i = 0

    def next(self):
        u = self.urls[self.i % len(self.urls)]
        self.i += 1
        return u


async def record(stage, dt):
    async with _timing_lock:
        TIMINGS[stage].append(dt)


async def tts_synthesize(client, rr, text):
    t0 = time.time()
    r = await client.post(
        f"{rr.next()}/synthesize",
        json={"text": text, "engine": "edge", "voice": EDGE_VOICE},
        timeout=120,
    )
    r.raise_for_status()
    await record("tts", time.time() - t0)
    return r.content


async def vc_convert(client, rr, source_bytes, reference_bytes):
    t0 = time.time()
    files = {
        "source_audio": ("source.wav", source_bytes, "audio/wav"),
        "reference_audio": ("reference.wav", reference_bytes, "audio/wav"),
    }
    r = await client.post(f"{rr.next()}/convert", files=files, timeout=300)
    r.raise_for_status()
    await record("vc", time.time() - t0)
    return r.content


async def qc_check(client, rr, wav_bytes, expected_text):
    t0 = time.time()
    files = {"audio": ("clip.wav", wav_bytes, "audio/wav")}
    r = await client.post(
        f"{rr.next()}/qc", files=files, data={"expected_text": expected_text}, timeout=180
    )
    r.raise_for_status()
    await record("qc", time.time() - t0)
    return r.json()


async def append_row(entry):
    async with _manifest_lock:
        with open(MANIFEST_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def wav_meta(path):
    info = sf.info(path)
    return round(info.duration, 3), info.samplerate


def select_sentences():
    """Deterministic slice: lowest 10 sentence_ids >= 850000 that are absent
    from the production run, 8-20 words, and free of the mangled orphan tokens
    the text pool contains (a stray single character, or a word truncated at a
    halant) -- those would inflate CER for reasons that have nothing to do with
    the audio."""
    done = set()
    if os.path.exists(DONE_IDS_PATH):
        with open(DONE_IDS_PATH) as f:
            done = {int(x.strip()) for x in f if x.strip()}
    pool = pd.read_parquet(TEXT_POOL_PATH)
    pool = pool[(pool["sentence_id"] >= SENTENCE_ID_FLOOR) & (~pool["sentence_id"].isin(done))]

    def ok(text):
        n = 0
        for tok in text.split():
            if tok == "।":
                continue
            if len(tok) <= 1 or tok.endswith("्"):
                return False
            n += 1
        return MIN_WORDS <= len(text.split()) <= MAX_WORDS

    pool = pool[pool["text"].map(ok)].sort_values("sentence_id")
    return pool.head(N_SENTENCES).reset_index(drop=True)


def load_references():
    refs = {}
    for e in EMOTIONS:
        d = os.path.join(EMOTIONS_ROOT, f"dataset_gemini_{e}")
        paths = sorted(
            os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")
        )
        if not paths:
            raise RuntimeError(f"no reference clips in {d}")
        blobs = []
        for p in paths:
            with open(p, "rb") as f:
                blobs.append((p, f.read()))
        refs[e] = blobs
    return refs


def load_existing_keys():
    keys = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                keys.add((int(e["sentence_id"]), e["emotion"]))
    return keys


async def synth_stage(client, tts_rr, qc_rr, sentences):
    """Edge-TTS once per sentence + source QC. Returns
    {sentence_id: (raw_path, wav_bytes, cer_source, wer_source)}."""
    out = {}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(row):
        sid = int(row["sentence_id"])
        text = row["text"]
        raw_path = os.path.join(RAW_DIR, f"{sid}.wav")
        qc_path = os.path.join(RAW_DIR, f"{sid}.qc.json")
        async with sem:
            if os.path.exists(raw_path) and os.path.exists(qc_path):
                with open(raw_path, "rb") as f:
                    wav = f.read()
                with open(qc_path) as f:
                    qc = json.load(f)
                print(f"[tts] {sid} cached cer_source={qc['cer']:.4f}", flush=True)
            else:
                wav = await tts_synthesize(client, tts_rr, text)
                with open(raw_path, "wb") as f:
                    f.write(wav)
                qc = await qc_check(client, qc_rr, wav, text)
                with open(qc_path, "w") as f:
                    json.dump(qc, f, ensure_ascii=False)
                print(f"[tts] {sid} cer_source={qc['cer']:.4f} wer={qc['wer']:.4f}", flush=True)
            out[sid] = (raw_path, wav, qc["cer"], qc["wer"])

    await asyncio.gather(*(one(r) for _, r in sentences.iterrows()))
    return out


async def convert_row(client, sem, vc_rr, qc_rr, row, emotion, raw, ref, desc_variant):
    sid = int(row["sentence_id"])
    text = row["text"]
    raw_path, raw_bytes, cer_source, wer_source = raw
    ref_path, ref_bytes = ref
    out_path = os.path.join(AUDIO_OUT, emotion, f"{sid}_{emotion}.wav")

    async with sem:
        t0 = time.time()
        entry = {
            "sentence_id": sid,
            "text": text,
            "emotion": emotion,
            "speaker": SPEAKER,
            "language": LANGUAGE,
            "description": DESCRIPTIONS[emotion][desc_variant],
            "description_variant": desc_variant,
            "audio_path": out_path,
            "raw_edge_path": raw_path,
            "duration_s": None,
            "sr": None,
            "edge_voice": EDGE_VOICE,
            "vc_backend": VC_BACKEND,
            "vc_ref_path": ref_path,
            "cer_source": cer_source,
            "wer_source": wer_source,
            "cer_final": None,
            "wer_final": None,
            "transcript_final": None,
        }
        try:
            if os.path.exists(out_path) and os.path.getsize(out_path) > 44:
                with open(out_path, "rb") as f:
                    vc_bytes = f.read()
                print(f"[vc] {sid}/{emotion} cached audio, re-QC only", flush=True)
            else:
                vc_bytes = await vc_convert(client, vc_rr, raw_bytes, ref_bytes)
                with open(out_path, "wb") as f:
                    f.write(vc_bytes)
            qc = await qc_check(client, qc_rr, vc_bytes, text)
            dur, sr = wav_meta(out_path)
            entry.update(
                duration_s=dur,
                sr=sr,
                cer_final=qc["cer"],
                wer_final=qc["wer"],
                transcript_final=qc["transcript"],
            )
            dt = time.time() - t0
            await record("row", dt)
            print(
                f"[row] {sid}/{emotion} v{desc_variant} ref={os.path.basename(ref_path)} "
                f"cer_final={qc['cer']:.4f} wer_final={qc['wer']:.4f} dur={dur:.2f}s "
                f"({dt:.1f}s)",
                flush=True,
            )
        except Exception as ex:
            entry["error"] = f"{type(ex).__name__}: {ex}"
            print(f"[row] {sid}/{emotion} ERROR {entry['error']}", flush=True)
        await append_row(entry)


async def main():
    t_start = time.time()
    sentences = select_sentences()
    if len(sentences) < N_SENTENCES:
        raise RuntimeError(f"only found {len(sentences)} candidate sentences")
    print(f"selected {len(sentences)} sentences: {list(sentences['sentence_id'])}", flush=True)
    for _, r in sentences.iterrows():
        print(f"  {r['sentence_id']} ({len(r['text'].split())}w) {r['text']}", flush=True)

    refs = load_references()
    for e in EMOTIONS:
        print(f"references[{e}] = {len(refs[e])} clips", flush=True)

    existing = load_existing_keys()
    if existing:
        print(f"resuming: {len(existing)} (sentence_id, emotion) rows already in manifest", flush=True)

    limits = httpx.Limits(max_connections=CONCURRENCY * 2, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        tts_rr, vc_rr, qc_rr = RoundRobin(TTS_URLS), RoundRobin(VC_URLS), RoundRobin(QC_URLS)

        t0 = time.time()
        raws = await synth_stage(client, tts_rr, qc_rr, sentences)
        print(f"[stage] edge synth + source QC of {len(raws)} clips in {time.time()-t0:.1f}s", flush=True)

        # Round-robin the reference clips and the description variants over the
        # (sentence, emotion) grid. Independent counters per emotion so every
        # reference clip inside a folder is used a roughly equal number of times.
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = []
        for emotion in EMOTIONS:
            for i, (_, row) in enumerate(sentences.iterrows()):
                sid = int(row["sentence_id"])
                ref = refs[emotion][i % len(refs[emotion])]
                variant = i % 3
                if (sid, emotion) in existing:
                    print(f"[skip] {sid}/{emotion} already in manifest", flush=True)
                    continue
                tasks.append(convert_row(client, sem, vc_rr, qc_rr, row, emotion, raws[sid], ref, variant))

        t0 = time.time()
        print(f"[stage] converting {len(tasks)} rows with concurrency={CONCURRENCY}...", flush=True)
        await asyncio.gather(*tasks)
        print(f"[stage] VC + final QC of {len(tasks)} rows in {time.time()-t0:.1f}s", flush=True)

    wall = time.time() - t_start
    summary = {
        "wall_clock_s": round(wall, 2),
        "n_rows_written_this_run": len(TIMINGS["row"]),
        "stage_calls": {k: len(v) for k, v in TIMINGS.items()},
        "stage_mean_s": {k: (round(sum(v) / len(v), 3) if v else None) for k, v in TIMINGS.items()},
        "stage_total_s": {k: round(sum(v), 2) for k, v in TIMINGS.items()},
        "concurrency": CONCURRENCY,
        "vc_replicas": len(VC_URLS),
        "qc_replicas": len(QC_URLS),
    }
    with open(TIMING_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE in {wall:.1f}s -> {MANIFEST_PATH}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

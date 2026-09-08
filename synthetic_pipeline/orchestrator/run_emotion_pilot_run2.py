"""
Emotion pilot, run 2 -- emotion-MATCHED text.

Run 1 (run_emotion_pilot.py) reused 10 neutral encyclopedia/news sentences from
text_pool.parquet across all four emotions. Good for isolating the acoustic
effect (identical content, only the emotion varies), bad as training data: it
teaches Parler that "angry" is an acoustic costume worn over arbitrary facts.

Here each emotion gets its own semantically congruent authored Nepali text, so
the caption, the prosody and the words all agree. 2 sentences x 4 emotions = 8
rows. Outputs carry a `_run_2` suffix and land in the same tree as run 1.

Pipeline unchanged: Edge-TTS (ne-NP-HemkalaNeural, female only) -> Seed-VC
against a Gemini "Amrita" reference of the matching emotion -> ASR QC.

Run: cd orchestrator && ./.venv/bin/python -u run_emotion_pilot_run2.py
"""
import asyncio
import json
import os
import time

import httpx
import soundfile as sf

PIPE_ROOT = "/root/tts/TTS_training/synthetic_pipeline"
AUDIO_OUT = os.path.join(PIPE_ROOT, "audio_out", "emotion_pilot")
RAW_DIR = os.path.join(AUDIO_OUT, "raw_edge")
MANIFEST_PATH = os.path.join(PIPE_ROOT, "manifests", "emotion_pilot_run2_manifest.jsonl")
EMOTIONS_ROOT = os.path.join(PIPE_ROOT, "emotions")

TTS_URLS = ["http://localhost:8001"]
VC_URLS = ["http://localhost:8002", "http://localhost:8012", "http://localhost:8022", "http://localhost:8032"]
QC_URLS = ["http://localhost:8003", "http://localhost:8013", "http://localhost:8023", "http://localhost:8033"]

EDGE_VOICE = "ne-NP-HemkalaNeural"  # female only, single consistent voice -- never ne-NP-SagarNeural
VC_BACKEND = "seed-vc-v1"
SPEAKER = "Amrita"
LANGUAGE = "ne"
RUN_TAG = "run_2"
CONCURRENCY = 8

EMOTIONS = ["angry", "sad", "happy", "neutral"]

# Authored, emotion-congruent Nepali. Synthetic ids in the 9_000_000 range so
# they can never collide with text_pool.parquet sentence_ids.
TEXTS = {
    "angry": [
        (9_000_001, "मैले पटक पटक भनेँ, तर तिमीले मेरो कुरा कहिल्यै सुनेनौ!"),
        (9_000_002, "यो अन्याय अब मैले सहन सक्दिनँ, मलाई तुरुन्तै जवाफ देऊ!"),
    ],
    "sad": [
        (9_000_011, "आज घर सुनसान छ, उहाँको आवाज सुन्ने आशा सधैंको लागि टुट्यो।"),
        (9_000_012, "मैले सबै गुमाएँ, अब मेरो साथमा एक्लोपन र आँसु मात्र बाँकी छ।"),
    ],
    "happy": [
        (9_000_021, "मेरो नतिजा आयो, मैले परीक्षामा सबैभन्दा राम्रो अङ्क ल्याएँ, म धेरै खुसी छु!"),
        (9_000_022, "धेरै वर्षपछि आज हामी सबै एकै ठाउँमा भेटियौं, कति रमाइलो भयो!"),
    ],
    "neutral": [
        (9_000_031, "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।"),
        (9_000_032, "नेपालको कुल क्षेत्रफल एक लाख सैंतालीस हजार वर्ग किलोमिटर रहेको छ।"),
    ],
}

# Same three paraphrase variants per emotion as run 1, verbatim, so run-1 and
# run-2 rows stay comparable. NOTE: the sad "low pitch" clause is known to
# contradict the measured references (sad refs sit at 278Hz vs neutral 207Hz);
# left unchanged here pending a decision, flagged in the report.
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

_manifest_lock = asyncio.Lock()


class RoundRobin:
    def __init__(self, urls):
        self.urls, self.i = urls, 0

    def next(self):
        u = self.urls[self.i % len(self.urls)]
        self.i += 1
        return u


async def tts_synthesize(client, rr, text):
    r = await client.post(f"{rr.next()}/synthesize",
                          json={"text": text, "engine": "edge", "voice": EDGE_VOICE}, timeout=120)
    r.raise_for_status()
    return r.content


async def vc_convert(client, rr, source_bytes, reference_bytes):
    files = {"source_audio": ("source.wav", source_bytes, "audio/wav"),
             "reference_audio": ("reference.wav", reference_bytes, "audio/wav")}
    r = await client.post(f"{rr.next()}/convert", files=files, timeout=300)
    r.raise_for_status()
    return r.content


async def qc_check(client, rr, wav_bytes, expected_text):
    files = {"audio": ("clip.wav", wav_bytes, "audio/wav")}
    r = await client.post(f"{rr.next()}/qc", files=files,
                          data={"expected_text": expected_text}, timeout=180)
    r.raise_for_status()
    return r.json()


def load_references():
    refs = {}
    for e in EMOTIONS:
        d = os.path.join(EMOTIONS_ROOT, f"dataset_gemini_{e}")
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav"))
        if not paths:
            raise RuntimeError(f"no reference clips in {d}")
        blobs = []
        for p in paths:
            with open(p, "rb") as f:
                blobs.append((p, f.read()))
        refs[e] = blobs
    return refs


async def append_row(entry):
    async with _manifest_lock:
        with open(MANIFEST_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def do_row(client, sem, tts_rr, vc_rr, qc_rr, emotion, sid, text, ref, variant):
    raw_path = os.path.join(RAW_DIR, f"{sid}_{RUN_TAG}.wav")
    out_path = os.path.join(AUDIO_OUT, emotion, f"{sid}_{emotion}_{RUN_TAG}.wav")
    ref_path, ref_bytes = ref

    async with sem:
        t0 = time.time()
        raw_bytes = await tts_synthesize(client, tts_rr, text)
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)
        src_qc = await qc_check(client, qc_rr, raw_bytes, text)

        vc_bytes = await vc_convert(client, vc_rr, raw_bytes, ref_bytes)
        with open(out_path, "wb") as f:
            f.write(vc_bytes)
        fin_qc = await qc_check(client, qc_rr, vc_bytes, text)

        info = sf.info(out_path)
        raw_info = sf.info(raw_path)
        entry = {
            "sentence_id": sid,
            "text": text,
            "emotion": emotion,
            "speaker": SPEAKER,
            "language": LANGUAGE,
            "description": DESCRIPTIONS[emotion][variant],
            "description_variant": variant,
            "audio_path": out_path,
            "raw_edge_path": raw_path,
            "duration_s": round(info.duration, 3),
            "raw_duration_s": round(raw_info.duration, 3),
            "sr": info.samplerate,
            "edge_voice": EDGE_VOICE,
            "vc_backend": VC_BACKEND,
            "vc_ref_path": ref_path,
            "cer_source": src_qc["cer"],
            "cer_final": fin_qc["cer"],
            "transcript_final": fin_qc["transcript"],
            "run": RUN_TAG,
            "text_source": "authored_emotion_matched",
        }
        await append_row(entry)
        print(f"[{emotion}] {sid} cer_src={src_qc['cer']:.3f} cer_fin={fin_qc['cer']:.3f} "
              f"dur {raw_info.duration:.2f}->{info.duration:.2f}s ref={os.path.basename(ref_path)} "
              f"({time.time()-t0:.1f}s)", flush=True)


async def main():
    refs = load_references()
    tts_rr, vc_rr, qc_rr = RoundRobin(TTS_URLS), RoundRobin(VC_URLS), RoundRobin(QC_URLS)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    tasks = []
    async with httpx.AsyncClient() as client:
        for emotion in EMOTIONS:
            for n, (sid, text) in enumerate(TEXTS[emotion]):
                ref = refs[emotion][n % len(refs[emotion])]
                tasks.append(do_row(client, sem, tts_rr, vc_rr, qc_rr,
                                    emotion, sid, text, ref, n % 3))
        await asyncio.gather(*tasks)
    print(f"DONE {len(tasks)} rows in {time.time()-t0:.1f}s -> {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

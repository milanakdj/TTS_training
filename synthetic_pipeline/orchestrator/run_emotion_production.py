"""
Emotional Nepali TTS dataset -- production run.

Target: 100 hours of kept audio, ~20 h in each of 5 emotions, one consistent
female voice ("Amrita"), ready to finetune ai4bharat/indic-parler-tts.

Pipeline per row:
  text -> Edge-TTS (ne-NP-HemkalaNeural + per-emotion rate/pitch/volume)
       -> ASR QC of the source (once per sentence, cached)
       -> Seed-VC against a screened v2 reference of the matching emotion
       -> ASR QC of the converted audio
       -> gate, tier, write FLAC, append manifest row

Design decisions and the measurements behind them
-------------------------------------------------
1. PACE COMES FROM EDGE, NOT FROM SEED-VC. Seed-VC takes F0 contour and timing
   from the source (contour r~0.70 to source vs r~0.15 to reference; output
   pause fraction is completely invariant to the reference). With
   length_adjust=1.0 every emotion came out byte-identical in duration to its
   source, making every "fast"/"slow" word in the captions a lie. Edge's own
   `rate` resynthesises at the new tempo instead of stretching a mel, so it is
   the cleaner lever. length_adjust is now exposed by the VC service but is
   deliberately left at 1.0 here -- one pace mechanism, not two compounding.

2. REFERENCES ARE THE v2 SET, SCREENED. The v1 references had Gemini shouting
   an octave above Amrita's register (angry 407 Hz / happy 420 Hz vs neutral
   206 Hz), which broke speaker identity (CAM++ cross-emotion 0.529 vs a
   same-speaker threshold ~0.562) and made angry and happy acoustically
   identical. v2 prompts carry an explicit register constraint; a screening
   pass keeps only clips in-register and on-anchor.

3. GATE ON CER *DELTA*, NOT ABSOLUTE CER. Clean Edge-TTS Nepali already scores
   CER 0.247 against the QC service, so absolute CER mostly measures Whisper's
   weakness on Nepali, not audio quality. One clip scored 0.453 purely because
   Whisper wrote "1,47,000" where the text said "एक लाख सैंतालीस हजार". The
   delta between pre- and post-conversion CER cancels that constant bias.
   A Devanagari-ratio floor additionally catches Whisper's romanised decodes,
   which produced bogus CER~1.0 on perfectly good audio.

4. TEXT IS MOSTLY EMOTION-NEUTRAL, ON PURPOSE. ~20% of sentences are rendered
   in all five emotions as contrastive minimal pairs; the rest get one emotion
   each. Parler must learn emotion from the CAPTION, not from text semantics --
   training only on emotion-congruent text would teach it to read sentiment off
   the words and ignore the description at inference. The minimal pairs make
   the caption the only thing that varies.

5. TARGET-DRIVEN, NOT ROW-DRIVEN. Each emotion accumulates until it reaches its
   hour target, then stops. Output durations vary too much for a row count to
   land on 100 h reliably.

Resumable: append-only manifest + done-keys file; a restart skips completed
(sentence_id, emotion) pairs and re-reads accumulated seconds from the manifest.

Run: nohup .venv/bin/python -u run_emotion_production.py > emotion_production.log 2>&1 &
"""
import argparse
import asyncio
import io
import json
import os
import random
import time

import httpx
import numpy as np
import pandas as pd
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--hours-per-emotion", type=float, default=20.0)
ap.add_argument("--suffix", default="")
ap.add_argument("--concurrency", type=int, default=24)
ap.add_argument("--limit-rows", type=int, default=None, help="testing only")
ap.add_argument("--ref-index", default=None, help="screened reference index parquet")
ap.add_argument("--vc-ports", default="8002,8012,8022,8032")
ap.add_argument("--qc-ports", default="8003,8013,8023,8033")
ap.add_argument("--keep-raw", action="store_true", help="persist every Edge source (~20GB)")
ap.add_argument("--emotions", default=None,
                help="comma-separated subset to produce (default: all five). Rows are keyed "
                     "by (sentence_id, emotion), so a later run with the same --suffix and a "
                     "wider set resumes and fills only the missing emotions.")
args = ap.parse_args()

PIPE = "/root/tts/TTS_training/synthetic_pipeline"
SUF = f"_{args.suffix}" if args.suffix else ""
AUDIO_OUT = os.path.join(PIPE, "audio_out", f"emotion_production{SUF}")
MANIFEST = os.path.join(PIPE, "manifests", f"emotion_production{SUF}.jsonl")
DONE_KEYS = os.path.join(PIPE, "manifests", f"emotion_production{SUF}_done.txt")
TEXT_POOL = os.path.join(PIPE, "manifests", "text_pool.parquet")
PRIOR_DONE = os.path.join(PIPE, "manifests", "production_done_ids.txt")
REF_INDEX = args.ref_index or os.path.join(PIPE, "manifests", "reference_pool_emotion_v2.parquet")

TTS_URL = "http://localhost:8001"
VC_URLS = [f"http://localhost:{p.strip()}" for p in args.vc_ports.split(",") if p.strip()]
QC_URLS = [f"http://localhost:{p.strip()}" for p in args.qc_ports.split(",") if p.strip()]

EDGE_VOICE = "ne-NP-HemkalaNeural"   # female only, single consistent voice
SPEAKER = "Amrita"                    # the Nepali speaker token in indic-parler-tts
LANGUAGE = "ne"
VC_BACKEND = "seed-vc-v1"
OUT_FORMAT = "FLAC"                   # ~55% the size of PCM16 WAV, lossless
OUT_EXT = "flac"

ALL_EMOTIONS = ["angry", "happy", "excited", "sad", "neutral"]
EMOTIONS = ALL_EMOTIONS
if args.emotions:
    want = [e.strip() for e in args.emotions.split(",") if e.strip()]
    unknown = [e for e in want if e not in ALL_EMOTIONS]
    if unknown:
        raise SystemExit("unknown emotion(s): " + ", ".join(unknown))
    EMOTIONS = [e for e in ALL_EMOTIONS if e in want]
MIN_REFS_PER_EMOTION = 4

# Edge prosody per emotion. Pace and pitch direction are injected HERE because
# Seed-VC preserves source timing and contour. Values chosen so the four
# emotional styles separate on tempo and energy, not on register -- register is
# what the reference supplies, and it must stay Amrita's throughout.
EMOTION_CFG = {
    "angry":   {"rate": "+8%",  "pitch": "+15Hz", "volume": "+20%", "length_adjust": 1.0},
    "happy":   {"rate": "+12%", "pitch": "+30Hz", "volume": "+5%",  "length_adjust": 1.0},
    "excited": {"rate": "+20%", "pitch": "+35Hz", "volume": "+12%", "length_adjust": 1.0},
    "sad":     {"rate": "-22%", "pitch": "-15Hz", "volume": "-10%", "length_adjust": 1.0},
    "neutral": {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%",  "length_adjust": 1.0},
}

# Gate. Delta is the real signal; the absolute ceiling only catches collapse.
MAX_CER_DELTA = 0.15
MAX_CER_ABS = 0.55
MIN_DEVANAGARI = 0.50
# Tier A is the high-confidence subset -- train on A alone for a conservative run.
TIER_A_DELTA, TIER_A_ABS = 0.08, 0.32

MIN_WORDS, MAX_WORDS = 6, 24
CONTRASTIVE_FRACTION = 0.20   # sentences rendered in all 5 emotions
SEED = 20260903

DESCRIPTIONS = {
    "angry": [
        "Amrita speaks in an angry, harsh tone with a tense, forceful delivery and hard, clipped "
        "emphasis. Her voice is very expressive and animated. The recording is very high quality, "
        "with her voice sounding clear and very close up.",
        "Amrita sounds angry and sharp, her words driven out quickly and with force, tight and "
        "bitten off. The delivery is highly animated and strongly expressive. The recording quality "
        "is excellent, her voice captured cleanly and right up close.",
        "Amrita delivers the line in an angry, biting tone, pushing through the words at a fast, "
        "hard-edged pace with clipped stresses. She is intensely expressive throughout. It is a very "
        "clean, high-quality recording in which her voice sits clear and very close to the microphone.",
    ],
    "happy": [
        "Amrita speaks in a happy, cheerful tone with a bright timbre and a lively, slightly quick "
        "delivery that lifts at the ends of phrases. Her voice is very expressive and full of warmth. "
        "The recording is very high quality, with her voice sounding clear and very close up.",
        "Amrita sounds happy and bright, her words moving along at a brisk, buoyant clip with an "
        "audible smile in her voice. The delivery is very expressive and warm. The recording quality "
        "is excellent, her voice captured cleanly and right up close.",
        "Amrita delivers the line cheerfully and with delight, light and open in tone and quick in "
        "pace. She is warmly and richly expressive. It is a very clean, high-quality recording in "
        "which her voice sits clear and very close to the microphone.",
    ],
    "excited": [
        "Amrita speaks in an excited, energetic tone, fast-paced and full of momentum, the words "
        "tumbling out with eager enthusiasm. Her voice is very expressive and animated. The recording "
        "is very high quality, with her voice sounding clear and very close up.",
        "Amrita sounds excited and thrilled, rushing forward at a quick pace with breathless "
        "anticipation. The delivery is highly animated and strongly expressive. The recording quality "
        "is excellent, her voice captured cleanly and right up close.",
        "Amrita delivers the line with eager excitement, speaking fast and leaning into every phrase "
        "with delighted urgency. She is intensely expressive throughout. It is a very clean, "
        "high-quality recording in which her voice sits clear and very close to the microphone.",
    ],
    "sad": [
        "Amrita speaks in a sad, sorrowful tone with a slow, heavy delivery that trails downward at "
        "the end of each phrase. Her voice is soft, expressive and weighed down with emotion. The "
        "recording is very high quality, with her voice sounding clear and very close up.",
        "Amrita sounds sad and mournful, letting the words come slowly and quietly, weary and "
        "subdued. Her delivery is expressive and heavy with feeling. The recording quality is "
        "excellent, her voice captured cleanly and right up close.",
        "Amrita reads the line with quiet sadness and resignation, her pace dragging and her voice "
        "soft and drained. The emotion sits heavy in an expressive delivery. It is a very clean, "
        "high-quality recording in which her voice sits clear and very close to the microphone.",
    ],
    "neutral": [
        "Amrita speaks in a neutral, even tone at a moderate pitch and a steady, moderate pace. Her "
        "voice is clear and only slightly expressive. The recording is very high quality, with her "
        "voice sounding clear and very close up.",
        "Amrita sounds neutral and level, holding a middling pitch and an unhurried, steady pace. Her "
        "voice comes through clearly with only mild expressiveness. The recording quality is "
        "excellent, her voice captured cleanly and right up close.",
        "Amrita reads the line plainly and without inflection, staying at a moderate pitch and an "
        "even, measured pace. The delivery is clear and only lightly expressive. It is a very clean, "
        "high-quality recording in which her voice sits clear and very close to the microphone.",
    ],
}

for e in EMOTIONS:
    os.makedirs(os.path.join(AUDIO_OUT, e), exist_ok=True)
os.makedirs(os.path.join(AUDIO_OUT, "raw_sample"), exist_ok=True)

_mlock = asyncio.Lock()
_slock = asyncio.Lock()
TARGET_S = args.hours_per_emotion * 3600.0
STATE = {
    "kept_s": {e: 0.0 for e in EMOTIONS},
    "kept": {e: 0 for e in EMOTIONS},
    "dropped": 0, "errors": 0, "processed": 0, "start": time.time(),
}


class RR:
    def __init__(self, urls):
        self.u, self.i = urls, 0

    def next(self):
        v = self.u[self.i % len(self.u)]
        self.i += 1
        return v


def load_done():
    if not os.path.exists(DONE_KEYS):
        return set()
    out = set()
    with open(DONE_KEYS) as f:
        for line in f:
            line = line.strip()
            if line and "\t" in line:
                sid, em = line.split("\t", 1)
                out.add((int(sid), em))
    return out


def reload_state():
    """Rebuild accumulated seconds from the manifest so a restart resumes at
    the right point instead of overshooting the hour targets."""
    if not os.path.exists(MANIFEST):
        return
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kept"):
                e = r["emotion"]
                if e in STATE["kept_s"]:
                    STATE["kept_s"][e] += r.get("duration_s", 0.0)
                    STATE["kept"][e] += 1


def devanagari_ratio(t):
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0900" <= c <= "\u097f" for c in letters) / len(letters)


def text_ok(t):
    """Reject the text pool's mangled rows: orphan single characters and words
    truncated at a halant. They inflate CER for reasons unrelated to audio.

    Also rejects non-Devanagari rows. The pool contains English sentences (e.g.
    sentence_id 863923, "Indian President Pranab Mukherjee has arrived at..."),
    which have no business in a Nepali set being read by a Nepali voice; before
    this check they were synthesized and converted in full and only thrown out at
    the very end by the Devanagari floor, after the GPU work was already spent."""
    if devanagari_ratio(t) < MIN_DEVANAGARI:
        return False
    toks = t.split()
    if not (MIN_WORDS <= len(toks) <= MAX_WORDS):
        return False
    for tok in toks:
        if tok == "।":
            continue
        if len(tok) <= 1 or tok.endswith("्"):
            return False
    return True


def build_plan(done):
    pool = pd.read_parquet(TEXT_POOL)
    prior = set()
    if os.path.exists(PRIOR_DONE):
        with open(PRIOR_DONE) as f:
            prior = {int(x.strip()) for x in f if x.strip()}
    pool = pool[~pool["sentence_id"].isin(prior)]
    pool = pool[pool["text"].map(text_ok)]
    pool = pool.sort_values("sentence_id").reset_index(drop=True)

    rng = random.Random(SEED)
    jobs = []
    for row in pool.itertuples(index=False):
        sid, text = int(row.sentence_id), row.text
        if rng.random() < CONTRASTIVE_FRACTION:
            for e in EMOTIONS:
                jobs.append((sid, text, e, True))
        else:
            jobs.append((sid, text, rng.choice(EMOTIONS), False))
    rng.shuffle(jobs)   # interleave emotions so all targets fill together
    jobs = [j for j in jobs if (j[0], j[2]) not in done]
    if args.limit_rows:
        jobs = jobs[: args.limit_rows]
    return jobs


def load_refs():
    idx = pd.read_parquet(REF_INDEX)
    # The screening manifest deliberately retains rejected clips so they stay visible,
    # so production MUST filter on `passed` -- otherwise the very clips the gate threw
    # out (whispered sad, 300+ Hz excited) would seed thousands of conversions.
    if "passed" in idx.columns:
        idx = idx[idx["passed"].astype(bool)]
    refs = {}
    for e in EMOTIONS:
        sub = idx[idx["emotion"] == e]
        if len(sub) < MIN_REFS_PER_EMOTION:
            raise RuntimeError(
                f"only {len(sub)} screened reference(s) for '{e}' in {REF_INDEX}; "
                f"need >= {MIN_REFS_PER_EMOTION}. Regenerate that emotion or drop it "
                f"from --emotions.")
        blobs = []
        for p in sub["path"]:
            with open(p, "rb") as f:
                blobs.append((p, f.read()))
        refs[e] = blobs
    return refs


async def tts(client, text, cfg):
    body = {"text": text, "engine": "edge", "voice": EDGE_VOICE,
            "rate": cfg["rate"], "pitch": cfg["pitch"], "volume": cfg["volume"]}
    r = await client.post(f"{TTS_URL}/synthesize", json=body, timeout=120)
    r.raise_for_status()
    return r.content


async def vc(client, rr, src, ref, length_adjust):
    files = {"source_audio": ("s.wav", src, "audio/wav"),
             "reference_audio": ("r.wav", ref, "audio/wav")}
    data = {"length_adjust": str(length_adjust)}
    r = await client.post(f"{rr.next()}/convert", files=files, data=data, timeout=300)
    r.raise_for_status()
    return r.content


async def qc(client, rr, wav, expected):
    files = {"audio": ("c.wav", wav, "audio/wav")}
    r = await client.post(f"{rr.next()}/qc", files=files,
                          data={"expected_text": expected}, timeout=240)
    r.raise_for_status()
    return r.json()


async def append(entry, sid, emotion):
    async with _mlock:
        with open(MANIFEST, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(DONE_KEYS, "a") as f:
            f.write(f"{sid}\t{emotion}\n")


def acoustics(wav_bytes):
    """Cheap per-row acoustic fingerprint -- no pitch tracking (pyin costs
    ~0.5s/clip, unaffordable at this scale; a sampled QA pass covers F0)."""
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = 1024
    frames = y[: len(y) // n * n].reshape(-1, n) if len(y) >= n else y[None, :]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    denom = spec.sum(axis=1) + 1e-12
    centroid = float(np.median((spec * freqs).sum(axis=1) / denom))
    speech = rms > max(rms.max() * 0.08, 1e-4)
    return {
        "rms_p95": round(float(np.percentile(rms, 95)), 5),
        "centroid_hz": round(centroid, 1),
        "speech_fraction": round(float(speech.mean()), 4),
        "peak": round(float(np.abs(y).max()), 4),
    }


async def do_row(client, sem, vc_rr, qc_rr, src_cache, refs, sid, text, emotion, contrastive):
    cfg = EMOTION_CFG[emotion]
    async with sem:
        if STATE["kept_s"][emotion] >= TARGET_S:
            return
        try:
            # Edge source is per (sentence, emotion) because prosody args differ,
            # but source QC only depends on the audio, so cache by that key.
            ck = (sid, emotion)
            if ck in src_cache:
                raw, cer_src, dev_src = src_cache[ck]
            else:
                raw = await tts(client, text, cfg)
                q = await qc(client, qc_rr, raw, text)
                cer_src, dev_src = q["cer"], q.get("devanagari_ratio", 1.0)
                src_cache[ck] = (raw, cer_src, dev_src)

            # Deterministic reference pick. Python's str hash() is salted per
            # process, so hash(emotion) would choose different references after
            # a resume and silently split the dataset across two ref mappings.
            # If the Edge source itself is unintelligible to the QC model, the row
            # cannot survive the gate no matter how well VC performs, so skip the
            # conversion instead of paying for it. In the smoke run 3 of 6 drops
            # were this case. The row is still recorded, so the drop stays visible.
            if cer_src > MAX_CER_ABS or dev_src < MIN_DEVANAGARI:
                await append({
                    "sentence_id": sid, "text": text, "emotion": emotion,
                    "speaker": SPEAKER, "language": LANGUAGE,
                    "audio_path": None, "audio_relpath": None,
                    "cer_source": round(cer_src, 4), "devanagari_ratio_source": round(dev_src, 4),
                    "kept": False, "tier": "-", "drop_reason": "source_failed_qc",
                }, sid, emotion)
                async with _slock:
                    STATE["processed"] += 1
                    STATE["dropped"] += 1
                    STATE["dropped_at_source"] = STATE.get("dropped_at_source", 0) + 1
                return

            ref_i = (sid + EMOTIONS.index(emotion) * 7) % len(refs[emotion])
            ref_path, ref_bytes = refs[emotion][ref_i]
            out = await vc(client, vc_rr, raw, ref_bytes, cfg["length_adjust"])
            qf = await qc(client, qc_rr, out, text)

            cer_fin = qf["cer"]
            dev_fin = qf.get("devanagari_ratio", 1.0)
            delta = cer_fin - cer_src
            keep = (delta <= MAX_CER_DELTA and cer_fin <= MAX_CER_ABS
                    and dev_fin >= MIN_DEVANAGARI)
            tier = "A" if (keep and delta <= TIER_A_DELTA and cer_fin <= TIER_A_ABS) else ("B" if keep else "-")

            y, sr = sf.read(io.BytesIO(out), dtype="float32")
            dur = len(y) / sr
            rel = os.path.join(emotion, f"{sid}_{emotion}.{OUT_EXT}")
            apath = os.path.join(AUDIO_OUT, rel)
            if keep:
                sf.write(apath, y, sr, format=OUT_FORMAT)
            ac = acoustics(out)

            raw_y, raw_sr = sf.read(io.BytesIO(raw), dtype="float32")
            raw_dur = len(raw_y) / raw_sr
            # Keep a 2% sample of the Edge sources for A/B listening; persisting
            # every one would cost ~20GB on a disk that is already 89% full.
            if args.keep_raw or (sid % 50 == 0):
                sf.write(os.path.join(AUDIO_OUT, "raw_sample", f"{sid}_{emotion}.{OUT_EXT}"),
                         raw_y, raw_sr, format=OUT_FORMAT)

            entry = {
                "sentence_id": sid, "text": text, "emotion": emotion,
                "speaker": SPEAKER, "language": LANGUAGE,
                "description": DESCRIPTIONS[emotion][sid % 3],
                "description_variant": sid % 3,
                "audio_path": apath if keep else None,
                "audio_relpath": rel if keep else None,
                "duration_s": round(dur, 3), "raw_duration_s": round(raw_dur, 3),
                "sr": sr, "format": OUT_EXT,
                "edge_voice": EDGE_VOICE,
                "edge_rate": cfg["rate"], "edge_pitch": cfg["pitch"], "edge_volume": cfg["volume"],
                "vc_backend": VC_BACKEND, "vc_ref_path": ref_path,
                "length_adjust": cfg["length_adjust"],
                "cer_source": round(cer_src, 4), "cer_final": round(cer_fin, 4),
                "cer_delta": round(delta, 4),
                "devanagari_ratio_final": round(dev_fin, 4),
                "transcript_final": qf["transcript"],
                "contrastive_pair": bool(contrastive),
                "kept": bool(keep), "tier": tier,
                "drop_reason": "" if keep else (
                    "cer_delta" if delta > MAX_CER_DELTA else
                    "cer_abs" if cer_fin > MAX_CER_ABS else "devanagari"),
                **ac,
            }
            await append(entry, sid, emotion)
            async with _slock:
                STATE["processed"] += 1
                if keep:
                    STATE["kept_s"][emotion] += dur
                    STATE["kept"][emotion] += 1
                else:
                    STATE["dropped"] += 1
        except Exception as ex:
            async with _slock:
                STATE["errors"] += 1
                STATE["processed"] += 1
            if STATE["errors"] % 200 == 1:
                print(f"[err] {type(ex).__name__}: {str(ex)[:180]}", flush=True)


async def progress():
    while True:
        await asyncio.sleep(60)
        el = (time.time() - STATE["start"]) / 60
        tot_h = sum(STATE["kept_s"].values()) / 3600
        per = " ".join(f"{e[:3]}={STATE['kept_s'][e]/3600:.2f}h" for e in EMOTIONS)
        rate = STATE["processed"] / max(time.time() - STATE["start"], 1)
        remain = sum(max(0.0, TARGET_S - STATE["kept_s"][e]) for e in EMOTIONS)
        secs_per_row = (sum(STATE["kept_s"].values()) / max(sum(STATE["kept"].values()), 1))
        eta = (remain / max(secs_per_row, 0.1)) / max(rate, 0.01) / 3600
        print(f"[progress] {tot_h:.2f}h/{args.hours_per_emotion*len(EMOTIONS):.0f}h  {per}  "
              f"rows={STATE['processed']} drop={STATE['dropped']} err={STATE['errors']} "
              f"rate={rate:.2f}/s elapsed={el:.1f}m eta={eta:.1f}h", flush=True)
        if all(STATE["kept_s"][e] >= TARGET_S for e in EMOTIONS):
            return


async def main():
    done = load_done()
    reload_state()
    print(f"resume: {len(done)} rows already done, "
          f"{sum(STATE['kept_s'].values())/3600:.2f}h already kept", flush=True)
    refs = load_refs()
    print("references: " + ", ".join(f"{e}={len(refs[e])}" for e in EMOTIONS), flush=True)
    jobs = build_plan(done)
    print(f"plan: {len(jobs)} candidate rows, target {args.hours_per_emotion}h x "
          f"{len(EMOTIONS)} emotions, VC replicas={len(VC_URLS)}, QC replicas={len(QC_URLS)}",
          flush=True)

    vc_rr, qc_rr = RR(VC_URLS), RR(QC_URLS)
    sem = asyncio.Semaphore(args.concurrency)
    src_cache = {}
    limits = httpx.Limits(max_connections=args.concurrency * 3,
                          max_keepalive_connections=args.concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        pt = asyncio.create_task(progress())
        batch = []
        for sid, text, emotion, contrastive in jobs:
            batch.append(do_row(client, sem, vc_rr, qc_rr, src_cache, refs,
                                sid, text, emotion, contrastive))
            if len(batch) >= args.concurrency * 8:
                await asyncio.gather(*batch)
                batch = []
                src_cache.clear()          # bound memory
                if all(STATE["kept_s"][e] >= TARGET_S for e in EMOTIONS):
                    break
        if batch:
            await asyncio.gather(*batch)
        pt.cancel()

    print("=== FINAL ===", flush=True)
    for e in EMOTIONS:
        print(f"  {e:8s} {STATE['kept_s'][e]/3600:6.2f}h  {STATE['kept'][e]} clips", flush=True)
    print(f"  total {sum(STATE['kept_s'].values())/3600:.2f}h  "
          f"processed={STATE['processed']} dropped={STATE['dropped']} errors={STATE['errors']}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())

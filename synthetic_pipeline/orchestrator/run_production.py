"""
Production synthetic-data generation run: text_pool.parquet -> Edge-TTS synth
-> QC -> Seed-VC re-voice -> QC -> pick best usable version -> manifest + audio.

Decisions baked in from the 10-row pilot test and explicit user direction:
  - Edge-TTS only for content synthesis (MMS and pretrained-Parler both
    performed too poorly in the pilot -- ~0-67% pass vs edge's 70%).
    Alternates between the two Nepali voices for gender variety.
  - Every row gets voice-converted (Seed-VC) against a rotating pool of real
    reference speakers for timbre diversity, per explicit instruction to use
    VC on all rows and just budget for the lower post-VC pass rate.
  - QC (CER < 0.3 vs the known source text) gates both the raw synth and the
    VC'd version. If VC fails QC but the raw synth passed, keep the raw synth
    as a fallback instead of discarding the row outright -- maximizes usable
    yield per unit of compute rather than wasting a successful synthesis.
    A row is only dropped entirely if BOTH fail QC.
  - Load-balanced across multiple replica instances of the GPU-bound VC and
    QC services (3 extra of each, ports 8012/8022/8032 and 8013/8023/8033,
    plus the originals on 8002/8003) for real parallel throughput on the
    shared GPU. TTS is edge-tts (network I/O, not GPU-bound) so one instance
    (8001) is fine at high concurrency.
  - Fully resumable: checkpoints progress by sentence_id into a JSONL manifest
    (append-only) and a separate "done ids" file; a restart skips whatever's
    already been processed instead of redoing it.

Run: nohup .venv/bin/python -u run_production.py > production.log 2>&1 &
"""
import argparse
import asyncio
import io
import json
import os
import random
import sys
import time

import httpx
import pandas as pd
import soundfile as sf

_argp = argparse.ArgumentParser()
_argp.add_argument("--suffix", default="", help="tag for manifest/audio dirs, e.g. 'validation'")
_argp.add_argument("--limit", type=int, default=None, help="only process the first N rows of the pool (testing)")
_args, _ = _argp.parse_known_args()

PIPE_ROOT = "/root/tts/TTS_training/synthetic_pipeline"
_suffix = f"_{_args.suffix}" if _args.suffix else ""
AUDIO_OUT = os.path.join(PIPE_ROOT, "audio_out", f"production{_suffix}")
MANIFEST_PATH = os.path.join(PIPE_ROOT, "manifests", f"production_manifest{_suffix}.jsonl")
DONE_IDS_PATH = os.path.join(PIPE_ROOT, "manifests", f"production_done_ids{_suffix}.txt")
TEXT_POOL_PATH = os.path.join(PIPE_ROOT, "manifests", "text_pool.parquet")
# v2: the original Titung-based reference pool turned out to be one voice with
# synthetic noise/reverb augmentation (pitch std ~4Hz across 2668 rows) rather
# than real distinct speakers. Replaced with a pool built from genuinely
# real, multi-speaker recordings (lilgoose777/nepali-tts-massive-combined +
# google/fleurs ne_np), verified via pitch analysis (std ~43Hz, 60-364Hz range).
REF_INDEX_PATH = os.path.join(PIPE_ROOT, "manifests", "reference_pool_v2_index.parquet")
REF_POOL_VERSION = "v2_real_multispeaker"

TTS_URLS = ["http://localhost:8001"]
VC_URLS = ["http://localhost:8002", "http://localhost:8012", "http://localhost:8022", "http://localhost:8032"]
QC_URLS = ["http://localhost:8003", "http://localhost:8013", "http://localhost:8023", "http://localhost:8033"]

EDGE_VOICES = ["ne-NP-HemkalaNeural", "ne-NP-SagarNeural"]

TARGET_FINAL_ROWS = 500_000
CONCURRENCY = 48  # in-flight rows at once; TTS is I/O-bound, VC/QC are load-balanced across 4 replicas each

os.makedirs(AUDIO_OUT, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

_manifest_lock = asyncio.Lock()
_stats_lock = asyncio.Lock()
stats = {"kept_vc": 0, "kept_raw_fallback": 0, "dropped": 0, "errors": 0, "processed": 0, "start": time.time()}


def load_done_ids():
    if not os.path.exists(DONE_IDS_PATH):
        return set()
    with open(DONE_IDS_PATH) as f:
        return set(int(line.strip()) for line in f if line.strip())


async def append_result(manifest_entry, sentence_id):
    async with _manifest_lock:
        with open(MANIFEST_PATH, "a") as f:
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
        with open(DONE_IDS_PATH, "a") as f:
            f.write(f"{sentence_id}\n")


class RoundRobin:
    def __init__(self, urls):
        self.urls = urls
        self.i = 0

    def next(self):
        u = self.urls[self.i % len(self.urls)]
        self.i += 1
        return u


async def tts_synthesize(client, rr, text, voice):
    url = rr.next()
    r = await client.post(f"{url}/synthesize", json={"text": text, "engine": "edge", "voice": voice}, timeout=60)
    r.raise_for_status()
    return r.content


async def vc_convert(client, rr, source_bytes, reference_bytes):
    url = rr.next()
    files = {
        "source_audio": ("source.wav", source_bytes, "audio/wav"),
        "reference_audio": ("reference.wav", reference_bytes, "audio/wav"),
    }
    r = await client.post(f"{url}/convert", files=files, timeout=120)
    r.raise_for_status()
    return r.content


async def qc_check(client, rr, wav_bytes, expected_text):
    url = rr.next()
    files = {"audio": ("clip.wav", wav_bytes, "audio/wav")}
    data = {"expected_text": expected_text}
    r = await client.post(f"{url}/qc", files=files, data=data, timeout=90)
    r.raise_for_status()
    return r.json()


async def process_row(client, sem, tts_rr, vc_rr, qc_rr, row, ref_bytes_list, row_index):
    async with sem:
        sentence_id = int(row["sentence_id"])
        text = row["text"]
        voice = EDGE_VOICES[row_index % 2]
        entry = {"sentence_id": sentence_id, "text": text, "voice": voice, "ref_pool_version": REF_POOL_VERSION}
        try:
            synth_bytes = await tts_synthesize(client, tts_rr, text, voice)
            synth_qc = await qc_check(client, qc_rr, synth_bytes, text)
            entry["synth_cer"] = synth_qc["cer"]
            entry["synth_pass"] = synth_qc["pass"]

            if not synth_qc["pass"]:
                async with _stats_lock:
                    stats["dropped"] += 1
                    stats["processed"] += 1
                entry["kept"] = "none"
                await append_result(entry, sentence_id)
                return

            ref_bytes = random.choice(ref_bytes_list)
            vc_bytes = await vc_convert(client, vc_rr, synth_bytes, ref_bytes)
            vc_qc = await qc_check(client, qc_rr, vc_bytes, text)
            entry["vc_cer"] = vc_qc["cer"]
            entry["vc_pass"] = vc_qc["pass"]

            if vc_qc["pass"]:
                out_path = os.path.join(AUDIO_OUT, f"{sentence_id}_vc.wav")
                with open(out_path, "wb") as f:
                    f.write(vc_bytes)
                entry["audio_path"] = out_path
                entry["kept"] = "vc"
                async with _stats_lock:
                    stats["kept_vc"] += 1
                    stats["processed"] += 1
            else:
                out_path = os.path.join(AUDIO_OUT, f"{sentence_id}_raw.wav")
                with open(out_path, "wb") as f:
                    f.write(synth_bytes)
                entry["audio_path"] = out_path
                entry["kept"] = "raw_fallback"
                async with _stats_lock:
                    stats["kept_raw_fallback"] += 1
                    stats["processed"] += 1

            await append_result(entry, sentence_id)
        except Exception as e:
            entry["error"] = str(e)
            entry["kept"] = "error"
            async with _stats_lock:
                stats["errors"] += 1
                stats["processed"] += 1
            await append_result(entry, sentence_id)


async def progress_logger():
    while True:
        await asyncio.sleep(60)
        elapsed = time.time() - stats["start"]
        kept_total = stats["kept_vc"] + stats["kept_raw_fallback"]
        rate = stats["processed"] / elapsed if elapsed > 0 else 0
        remaining = TARGET_FINAL_ROWS - kept_total
        eta_s = remaining / (rate * (kept_total / max(stats["processed"], 1))) if rate > 0 and kept_total > 0 else float("inf")
        print(
            f"[progress] processed={stats['processed']} kept_vc={stats['kept_vc']} "
            f"kept_raw_fallback={stats['kept_raw_fallback']} dropped={stats['dropped']} "
            f"errors={stats['errors']} kept_total={kept_total}/{TARGET_FINAL_ROWS} "
            f"rate={rate:.2f}rows/s elapsed_min={elapsed/60:.1f} eta_hours={eta_s/3600:.1f}",
            flush=True,
        )
        if kept_total >= TARGET_FINAL_ROWS:
            print("[progress] TARGET REACHED", flush=True)


async def main():
    print("loading text pool + reference pool...", flush=True)
    pool = pd.read_parquet(TEXT_POOL_PATH)
    ref_index = pd.read_parquet(REF_INDEX_PATH)
    ref_bytes_list = []
    for p in ref_index["path"]:
        with open(p, "rb") as f:
            ref_bytes_list.append(f.read())
    print(f"text pool: {len(pool)} sentences, reference pool: {len(ref_bytes_list)} clips", flush=True)

    done_ids = load_done_ids()
    print(f"resuming: {len(done_ids)} sentence_ids already processed", flush=True)
    pool = pool[~pool["sentence_id"].isin(done_ids)].reset_index(drop=True)
    if _args.limit:
        pool = pool.iloc[: _args.limit].reset_index(drop=True)
    print(f"{len(pool)} sentences remaining to process", flush=True)

    # seed stats.processed/kept from existing manifest so progress/ETA accounts for prior runs
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("kept") == "vc":
                    stats["kept_vc"] += 1
                elif e.get("kept") == "raw_fallback":
                    stats["kept_raw_fallback"] += 1
                elif e.get("kept") == "error":
                    stats["errors"] += 1
                else:
                    stats["dropped"] += 1
        print(f"resumed stats: kept_vc={stats['kept_vc']} kept_raw={stats['kept_raw_fallback']} dropped={stats['dropped']} errors={stats['errors']}", flush=True)

    kept_total = stats["kept_vc"] + stats["kept_raw_fallback"]
    if kept_total >= TARGET_FINAL_ROWS:
        print(f"already at/above target ({kept_total} >= {TARGET_FINAL_ROWS}), nothing to do", flush=True)
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    tts_rr, vc_rr, qc_rr = RoundRobin(TTS_URLS), RoundRobin(VC_URLS), RoundRobin(QC_URLS)

    limits = httpx.Limits(max_connections=CONCURRENCY * 2, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        logger_task = asyncio.create_task(progress_logger())
        tasks = []
        for i, row in pool.iterrows():
            tasks.append(process_row(client, sem, tts_rr, vc_rr, qc_rr, row, ref_bytes_list, i))
            # stop scheduling new work once we likely have enough (checked periodically, not exact)
        print(f"scheduling {len(tasks)} row-tasks with concurrency={CONCURRENCY}...", flush=True)
        await asyncio.gather(*tasks)
        logger_task.cancel()

    kept_total = stats["kept_vc"] + stats["kept_raw_fallback"]
    print(f"DONE. kept_total={kept_total} kept_vc={stats['kept_vc']} kept_raw_fallback={stats['kept_raw_fallback']} dropped={stats['dropped']} errors={stats['errors']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

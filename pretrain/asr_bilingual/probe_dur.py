"""Durations for the 2,215 h Nepali corpus manifest, which has none.

soundfile.info reads only the header, so this is a stat-bound pass, not a decode.
Threads (not processes) because it is I/O against the network FS.
"""
import json, sys, os
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf

SRC = "/root/tts/TTS_training/whisper/nepali_corpus_asr.jsonl"
OUT = "/root/tts/TTS_training/pretrain/asr_bilingual/manifests/ne_corpus_train.jsonl"
GOLD = {"rasa", "indicvoices-r", "indicvoices-r-long"}   # already in ne_train, skip

rows = []
for l in open(SRC):
    d = json.loads(l)
    if d.get("src") in GOLD: continue      # don't double-count the gold split
    rows.append(d)
print(f"probing {len(rows):,} pseudo-labelled rows", flush=True)

def dur(d):
    try:
        i = sf.info(d["audio"]); return i.frames / i.samplerate
    except Exception: return None

n = 0; tot = 0.0
with open(OUT, "w") as w, ThreadPoolExecutor(32) as pool:
    for d, s in zip(rows, pool.map(dur, rows)):
        if not s or s <= 0.5 or s > 30: continue
        w.write(json.dumps({"audio_filepath": d["audio"], "duration": s,
                            "text": d["text"], "lang": "ne"}, ensure_ascii=False) + "\n")
        n += 1; tot += s
        if n % 50000 == 0: print(f"  {n:,} kept {tot/3600:.0f} h", flush=True)
print(f"[done] {n:,} utts {tot/3600:.1f} h -> {OUT}", flush=True)

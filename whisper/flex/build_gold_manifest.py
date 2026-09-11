#!/usr/bin/env python3
"""Build a HUMAN-TRANSCRIBED-ONLY Nepali manifest for finetuning indic-transcribe-flex.

Why gold only. Measured 2026-09-11, flex on 25 clips per source:

    source                transcripts   WER     exact
    ans_snr40-50          saaras/canary 0.015   20/25
    mahadhwani            saaras/canary 0.022   17/25
    ai4bharat___rasa      human script  0.129   12/25
    indicvoices-r         human         0.177    4/25

flex is a nvidia/canary-1b-v2 derivative and our pseudo-labelled corpora are 64%
`transcript_used: canary`. Scoring flex there compares Canary to Canary: its output
matches `canary_transcript` at WER 0.019 but `saaras_transcript` at only 0.060.
Training on those labels would teach flex to imitate an ASR it already is, and
would move the apparent score without moving real accuracy.

The gold sources carry no `transcript_used` field: rasa is the recording script,
indicvoices-r ships a human `verbatim`. That is where flex is genuinely weak
(0.129-0.177) and therefore where there is something to learn.

  python3 build_gold_manifest.py
"""
import json, os, random, re

ROOT = "/workspace/proc_data_new"
OUT = "/root/tts/TTS_training/whisper/flex"
NE = re.compile(r'"language": ?"(ne|nep|nepali|Nepali)"|/Nepali/')
MAX_SEC = 30.0       # the checkpoint trains at max_duration 30
MIN_SEC = 0.5
SOURCES = [("rasa", "ai4bharat___rasa", "train/manifest.jsonl"),
           ("ivr", "indicvoices-r", "train/manifest.jsonl"),
           ("ivrl", "indicvoices-r-long", "train/manifest_train.jsonl")]

def resolve(p):
    if not p: return None
    for c in (p, p.replace("/projects/data/ttsteam/proc_data_new/", f"{ROOT}/"),
              p.replace("/projects/data/ttsteam/", "/workspace/")):
        if os.path.isfile(c): return c
    return None

rows = []
for tag, dirname, mf in SOURCES:
    path = f"{ROOT}/{dirname}/{mf}"
    kept = long_skip = noaudio = 0
    for line in open(path, errors="replace"):
        if not NE.search(line): continue
        try: d = json.loads(line)
        except Exception: continue
        if d.get("transcript_used"): continue          # pseudo-label -> never
        t = d.get("text") or d.get("normalised_text")
        if not t or not str(t).strip(): continue
        dur = float(d.get("duration") or 0)
        if dur <= MIN_SEC: continue
        if dur > MAX_SEC: long_skip += 1; continue
        a = resolve(d.get("audio_filepath"))
        if not a: noaudio += 1; continue
        rows.append({"audio": a, "text": str(t).strip(), "duration": dur, "src": tag})
        kept += 1
    print(f"  {tag:5} kept {kept:7,}  over-{MAX_SEC:.0f}s {long_skip:6,}  unresolved {noaudio:,}")

random.seed(1234)
random.shuffle(rows)
# Stratified held-out sets: the test split is what every number gets quoted from,
# so it must contain all three sources in proportion.
test, val, train = rows[:600], rows[600:1200], rows[1200:]
for name, part in (("test", test), ("val", val), ("train", train)):
    with open(f"{OUT}/gold_{name}.jsonl", "w") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = sum(r["duration"] for r in part) / 3600
    mix = {}
    for r in part: mix[r["src"]] = mix.get(r["src"], 0) + 1
    print(f"{name:6} {len(part):7,} rows  {h:7.1f} h  {mix}")

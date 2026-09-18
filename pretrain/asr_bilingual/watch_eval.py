#!/usr/bin/env python
"""Evaluate every new checkpoint on BOTH languages and append to a CSV.

`monitor: val_wer` in train.py validates on ne_val only, so checkpoint selection
is blind to English.  That is how the en-US collapse (normalised WER 0.109 -> 0.963)
ran unnoticed for 21 h.  This watches the checkpoint dir and scores ne_test and
en_test on each new *-last.ckpt, so English recovery after the prompt_mode=langID
fix is visible while the run is still in flight.
"""
import csv, json, os, subprocess, sys, time

CKDIR = "/root/tts/TTS_training/pretrain/asr_bilingual/ckpt/nemotron_ne_en/2026-09-12_10-08-33/checkpoints"
ROOT = "/root/tts/TTS_training/pretrain/asr_bilingual"
PY = "/workspace/venvs/nemo/bin/python"
CSV = f"{ROOT}/logs/ckpt_eval_history.csv"
# Base-model reference points, both measured on the same 600-clip manifests.
REF = {"ne": "base WER 1.440 / CER 0.878", "en": "base raw 0.283 / norm 0.109"}


def newest_last():
    c = [f for f in os.listdir(CKDIR) if f.endswith("-last.ckpt")]
    if not c:
        return None
    p = os.path.join(CKDIR, c[0])
    return p, os.path.getmtime(p)


def run(ckpt, lang):
    """Score one language; returns (raw_wer, raw_cer, norm_wer) or None."""
    man = f"{ROOT}/manifests/{lang}_test.jsonl"
    out = f"{ROOT}/logs/.watch_{lang}.json"
    r = subprocess.run([PY, f"{ROOT}/eval_ckpt.py", ckpt, "--manifest", man,
                        "--batch-size", "8", "--out", out],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(f"  [{lang}] FAILED rc={r.returncode}\n{r.stderr[-1500:]}", flush=True)
        return None
    try:
        s = json.load(open(out))["summary"]
        return s["wer"], s["cer"], s["wer_norm"]
    except Exception as e:
        print(f"  [{lang}] could not parse: {e}", flush=True)
        return None


def main():
    seen = None
    if not os.path.exists(CSV):
        with open(CSV, "w", newline="") as f:
            csv.writer(f).writerow(["utc", "ckpt", "ne_wer", "ne_cer", "ne_wer_norm",
                                    "en_wer", "en_cer", "en_wer_norm"])
    print(f"[watch] polling {CKDIR}\n[watch] reference: {REF}", flush=True)
    while True:
        cur = newest_last()
        if cur and cur[1] != seen:
            ckpt, seen = cur
            time.sleep(90)  # let the 7.6 GB write settle before reading it
            print(f"\n[watch] new checkpoint {os.path.basename(ckpt)}", flush=True)
            ne, en = run(ckpt, "ne"), run(ckpt, "en")
            row = ([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    os.path.basename(ckpt)]
                   + list(ne or ("", "", "")) + list(en or ("", "", "")))
            with open(CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
            if ne:
                print(f"  ne  WER {ne[0]:.3f}  CER {ne[1]:.3f}  (norm {ne[2]:.3f})", flush=True)
            if en:
                print(f"  en  WER {en[0]:.3f}  CER {en[1]:.3f}  (norm {en[2]:.3f})"
                      f"   <- base norm 0.109", flush=True)
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())

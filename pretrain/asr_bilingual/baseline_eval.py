"""Baseline: what does nemotron-3.5-asr score on our Nepali gold test, untrained?

ne-NP is in the model's prompt map even though the card lists it in no quality
tier, so the starting point is an open question -- and it decides whether this is
adaptation or teaching from scratch. Gold test only: the corpus labels are
Canary's, and this is not a Canary derivative, but the rule is the rule.
"""
import json, os, sys, time, torch
import nemo.collections.asr as nemo_asr

P = ("/workspace/hf_cache/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/"
     "snapshots/ea30d66debe3740a08b573244286791d423d6b3e/"
     "nemotron-3.5-asr-streaming-0.6b.nemo")
MAN = os.environ.get("MAN", "/root/tts/TTS_training/pretrain/asr_bilingual/manifests/ne_test.jsonl")
LANG = os.environ.get("LANG_ID", "ne-NP")
N = int(os.environ.get("N", 600))

rows = [json.loads(l) for l in open(MAN)][:N]
m = nemo_asr.models.ASRModel.restore_from(P, map_location="cuda")
m.eval()

t = time.time()
# language reaches the prompt head via each cut's supervision, i.e. the manifest
# `lang` field -- not a transcribe() kwarg. So feed it the manifest.
hyps = m.transcribe(MAN, batch_size=16, verbose=False)
el = time.time() - t
hyps = [h.text if hasattr(h, "text") else str(h) for h in hyps]

try:
    from jiwer import wer, cer
except ImportError:
    sys.exit("jiwer missing")
refs = [r["text"] for r in rows]
print(f"\n== nemotron base [{LANG}] n={len(rows)}  WER {wer(refs,hyps):.3f}  "
      f"CER {cer(refs,hyps):.3f}   [{el:.0f}s]")
for i in (0, 1, 2):
    print(f"  REF: {refs[i][:90]}")
    print(f"  HYP: {hyps[i][:90]}")
json.dump([{"id": i, "ref": r, "hyp": h} for i, (r, h) in enumerate(zip(refs, hyps))],
          open("/root/tts/TTS_training/pretrain/asr_bilingual/logs/baseline_ne.json", "w"), ensure_ascii=False)

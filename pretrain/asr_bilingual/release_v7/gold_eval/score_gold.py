"""Score a .nemo CTC checkpoint on the FLEURS ne_np gold set.

References are pushed through the *training* normalizer (pipeline_v2/normalize.py)
before scoring. That is not cosmetic: the model was trained on lowercased,
punctuation-stripped, digit-verbalised Devanagari, and FLEURS references carry
punctuation and ASCII digits ("सन् 1800 को"). Scoring raw references against a
model that was never taught to emit "1800" measures the normalisation mismatch,
not the model. Both sides get identical treatment.
"""
import argparse, importlib.util, json, re, sys
import torch

sys.path.insert(0, "/root/tts/TTS_training/pretrain/asr_bilingual/pipeline_v2")
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate

p = argparse.ArgumentParser()
p.add_argument("nemo")
p.add_argument("--manifest", default="fleurs_ne_test.raw.jsonl")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--label", default=None)
p.add_argument("--out", default=None)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.manifest)]
model = nemo_asr.models.EncDecCTCModelBPE.restore_from(a.nemo, map_location="cuda")
model.eval()

# the tokenizer the model actually carries, so "representable" means representable *here*
spec = importlib.util.spec_from_file_location(
    "normalize", "/root/tts/TTS_training/pretrain/asr_bilingual/pipeline_v2/normalize.py")
nzmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(nzmod)
sp = getattr(model.tokenizer, "tokenizer", model.tokenizer)
nz = nzmod.Normalizer(sp)

refs = [nz(r["text"], "ne-NP") for r in rows]
paths = [r["audio_filepath"] for r in rows]

with torch.no_grad():
    hyps = model.transcribe(paths, batch_size=a.batch, verbose=False)
hyps = [(h.text if hasattr(h, "text") else h) for h in hyps]
hyps = [nz(h, "ne-NP") for h in hyps]

def cer(hs, rs):
    return word_error_rate([" ".join(h.replace(" ", "")) for h in hs],
                           [" ".join(r.replace(" ", "")) for r in rs])

label = a.label or a.nemo.split("/")[-1]
wer, c = word_error_rate(hyps, refs), cer(hyps, refs)
empty = sum(1 for h in hyps if not h.strip())

# digits in the raw reference are the known normalisation fault line -- split it out
has_dig = [bool(re.search(r"[0-9०-९]", r["raw_text"])) for r in rows]
def sub(mask):
    h = [x for x, m in zip(hyps, mask) if m]; r = [x for x, m in zip(refs, mask) if m]
    return (len(h), word_error_rate(h, r), cer(h, r)) if h else (0, float("nan"), float("nan"))
n_d, wer_d, cer_d = sub(has_dig)
n_n, wer_n, cer_n = sub([not x for x in has_dig])

print(f"\n=== {label} on {a.manifest} ===")
print(f"n={len(rows)}  hours={sum(r['duration'] for r in rows)/3600:.3f}  empty_hyp={empty}")
print(f"  WER {wer:.4f}   CER {c:.4f}        (all)")
print(f"  WER {wer_n:.4f}   CER {cer_n:.4f}        (no digits in ref, n={n_n})")
print(f"  WER {wer_d:.4f}   CER {cer_d:.4f}        (digits in ref, n={n_d})")
print("\n  examples:")
for i in range(3):
    print(f"   ref: {refs[i][:95]}\n   hyp: {hyps[i][:95]}\n")

if a.out:
    json.dump({"label": label, "manifest": a.manifest, "n": len(rows),
               "wer": wer, "cer": c, "empty_hyp": empty,
               "wer_nodigit": wer_n, "cer_nodigit": cer_n, "n_nodigit": n_n,
               "wer_digit": wer_d, "cer_digit": cer_d, "n_digit": n_d,
               "pairs": [{"ref": r, "hyp": h} for r, h in zip(refs, hyps)]},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"  wrote {a.out}")

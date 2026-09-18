"""Re-score both models' hypotheses under ONE normalizer.

The two models were scored by their own native instruments, which normalise
differently (eval_ckpt.py keeps raw text and strips <xx-YY> tags; score_gold.py
runs the training normalizer). Comparing those numbers directly would repeat the
mistake the card already had to withdraw. Here both hypothesis sets are pulled
back to the same surface form and scored with the same metric.
"""
import importlib.util, json, re, unicodedata
from jiwer import wer as _wer, cer as _cer

LANGTAG = re.compile(r"<\|?[a-z]{2}(?:[-_][A-Za-z]{2})?\|?>")

spec = importlib.util.spec_from_file_location(
    "normalize", "/root/tts/TTS_training/pretrain/asr_bilingual/pipeline_v2/normalize.py")
nzmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(nzmod)
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="/workspace/asr_pretrain_v2/tokenizer/tokenizer.model")
nz = nzmod.Normalizer(sp)

def prep(s):
    """Strip emitted language tags BEFORE normalising: the normalizer maps
    unrepresentable characters to spaces, so an un-stripped '<ne-NP>' would turn
    into the two spurious words 'ne np' and be counted as insertions."""
    return nz(LANGTAG.sub(" ", unicodedata.normalize("NFC", s)), "ne-NP")

gold = [json.loads(l) for l in open("fleurs_ne_test.raw.jsonl")]
refs = [prep(r["text"]) for r in gold]

runs = {}
p = json.load(open("gold_parakeet110m.json"))
runs["parakeet-ctc-nepali-110m (110.8M, CTC)"] = [prep(x["hyp"]) for x in p["pairs"]]
try:
    n = json.load(open("gold_nemotron_raw.json"))
    runs["nemotron-asr-nepali-0.6b (600M, RNN-T)"] = [prep(r["hyp"]) for r in n["rows"]]
except FileNotFoundError:
    print("(nemotron hypotheses not present -- skipping)")

has_dig = [bool(re.search(r"[0-9०-९]", r["raw_text"])) for r in gold]

print(f"\n=== head-to-head: FLEURS ne_np test, n={len(refs)}, {sum(r['duration'] for r in gold)/3600:.2f} h")
print("    identical references, identical normalizer, identical metric\n")
print(f"{'model':44} {'WER':>7} {'CER':>7} {'WER(no-dig)':>12} {'empty':>6}")
out = {}
for k, hyps in runs.items():
    assert len(hyps) == len(refs), (k, len(hyps), len(refs))
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    R, H = [list(x) for x in zip(*pairs)]
    nd = [(r, h) for r, h, d in zip(refs, hyps, has_dig) if r and not d]
    Rn, Hn = [list(x) for x in zip(*nd)]
    w, c, wn = _wer(R, H), _cer(R, H), _wer(Rn, Hn)
    e = sum(1 for h in hyps if not h.strip())
    out[k] = {"wer": w, "cer": c, "wer_nodigit": wn, "empty": e, "n": len(R)}
    print(f"{k:44} {w:7.4f} {c:7.4f} {wn:12.4f} {e:6d}")
json.dump(out, open("head_to_head.json", "w"), indent=1, ensure_ascii=False)
print("\nwrote head_to_head.json")

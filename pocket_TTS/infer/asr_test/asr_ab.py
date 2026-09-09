"""Base Whisper vs the Nepali-finetuned Whisper, on audio NEITHER has seen.

Both models were trained/finetuned on `lilgoose7777/slr-combined-nepali-tts2`-derived
data or generic multilingual data; these 100 clips are mahadhwani, a different corpus
entirely, so this is a clean out-of-domain comparison.

CAVEAT that must travel with any number this prints: mahadhwani reference
transcripts are MACHINE-generated (`transcript_used: saaras`, kept only where three
ASRs agreed). So this measures agreement with saaras/canary/conformer, not ground
truth. It is a fair relative comparison between the two Whispers; it is not an
absolute WER.

    ASR_MODEL=base $ASR_QC_VENV  asr_ab.py     # faster-whisper large-v3-turbo
    ASR_MODEL=ft   $VC_VENV      asr_ab.py     # himalaya-ai Nepali finetune
"""
import json, os, re, string, unicodedata
import jiwer

MODE = os.environ.get("ASR_MODEL", "base")
F = "/root/tts/TTS_training/pocket_TTS/infer/asr_test"
rows = json.load(open(f"{F}/mahadhwani100.json"))
lim = int(os.environ.get("EVAL_LIMIT", "0"))
if lim:
    rows = rows[:lim]

_EXTRA = "।॥‘’“”–—…"
_TBL = str.maketrans("", "", string.punctuation + _EXTRA)
_PC = frozenset(string.punctuation + _EXTRA)


def norm(t):
    t = unicodedata.normalize("NFC", t or "").translate(_TBL).lower()
    return re.sub(r"\s+", " ", t).strip()


def dev_ratio(t):
    cs = [c for c in unicodedata.normalize("NFC", t or "")
          if not c.isspace() and not (c in _PC or unicodedata.category(c)[0] in ("P", "S"))]
    return sum(1 for c in cs if 0x900 <= ord(c) <= 0x97F) / len(cs) if cs else 0.0


if MODE == "base":
    from faster_whisper import WhisperModel
    m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    def transcribe(paths):
        out = []
        for p in paths:
            segs, _ = m.transcribe(p, language="ne", beam_size=5)
            out.append(" ".join(s.text for s in segs).strip())
        return out
    BS = 1
else:
    import torch, librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    # A medium Nepali finetune already exists and carries the same doubled-prefix
    # convention, so the identical decode loop scores either one.
    MID = os.environ.get("FT_MODEL", "himalaya-ai/whisper-large-v3-nepali-final")
    TOK = open("/root/.hf_milanakdj").read().strip()
    proc = WhisperProcessor.from_pretrained(MID, language="nepali", task="transcribe", token=TOK)
    m = WhisperForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16, token=TOK).to("cuda").eval()
    tk = proc.tokenizer
    tid = lambda t: tk.convert_tokens_to_ids(t)
    # How many <|startoftranscript|> tokens the checkpoint expects.
    #   2 -- models trained before the collator fix (the two published Nepali
    #        Whispers). With one sot they emit endless nonsense, WER > 400%.
    #   1 -- the standard prefix, for models trained with the fixed collator.
    # Getting this wrong does not error, it just returns garbage, so it must be
    # set per checkpoint rather than guessed.
    NSOT = int(os.environ.get("FT_NSOT", "2"))
    PREFIX = ([tid("<|startoftranscript|>")] * NSOT +
              [tid("<|ne|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")])
    print(f"decoding {MID} with {NSOT}x <|startoftranscript|>", flush=True)
    EOS = tid("<|endoftext|>")

    def transcribe(paths):
        audio = [librosa.load(p, sr=16000)[0] for p in paths]
        feats = proc.feature_extractor(audio, sampling_rate=16000,
                                       return_tensors="pt").input_features.to("cuda", torch.float16)
        cur = torch.tensor([PREFIX] * len(paths), device="cuda")
        out, past = cur, None
        done = torch.zeros(len(paths), dtype=torch.bool, device="cuda")
        with torch.no_grad():
            enc = m.get_encoder()(feats)
            for _ in range(225):
                r = m(encoder_outputs=enc, decoder_input_ids=cur, past_key_values=past, use_cache=True)
                past = r.past_key_values
                nxt = r.logits[:, -1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                out = torch.cat([out, nxt[:, None]], dim=1)
                done |= nxt == EOS
                if bool(done.all()):
                    break
                cur = nxt[:, None]
        return proc.batch_decode(out, skip_special_tokens=True)
    BS = 8

res = []
for i in range(0, len(rows), BS):
    chunk = rows[i:i + BS]
    hyps = transcribe([r["path"] for r in chunk])
    for r, hyp in zip(chunk, hyps):
        ref_n, hyp_n = norm(r["text"]), norm(hyp)
        if not ref_n:
            continue
        res.append({"id": r["id"], "wer": jiwer.wer(ref_n, hyp_n), "cer": jiwer.cer(ref_n, hyp_n),
                    "dev_ratio": dev_ratio(hyp), "hyp": hyp, "ref": r["text"]})
    print(f"  {min(i+BS, len(rows))}/{len(rows)}", flush=True)

good = [r for r in res if r["dev_ratio"] >= 0.5]
n_rom = len(res) - len(good)
print(f"\n== {MODE}: n={len(good)} (+{n_rom} discarded as romanized) "
      f"WER {sum(r['wer'] for r in good)/len(good):.3f}  "
      f"CER {sum(r['cer'] for r in good)/len(good):.3f}")
if MODE == "ft":
    tag = f"ft_{MID.split('/')[-1].removeprefix('whisper-')}_sot{NSOT}"
else:
    tag = MODE
json.dump(res, open(f"{F}/result_{tag}.json", "w"), ensure_ascii=False, indent=1)

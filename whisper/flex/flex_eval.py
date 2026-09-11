#!/usr/bin/env python3
"""Score bodhan-ai/indic-transcribe-flex on the same 100 unseen mahadhwani clips
every other Nepali ASR in this project was scored on.

Nepali is in the model's 27 languages but is ABSENT from its published Voice-of-India
table, so there is no vendor number to compare against -- this establishes it.

Normalisation is copied verbatim from pocket_TTS/infer/asr_test/asr_ab.py so the
numbers sit in the same table as the whisper runs.

  python3 flex_eval.py                      # base checkpoint
  FLEX_MODEL=/path/to/ft python3 flex_eval.py --tag ft
"""
import argparse, json, os, re, string, sys, time, unicodedata
import torch

M = os.environ.get("FLEX_MODEL", "/workspace/models/indic-transcribe-flex")
CODE = "/workspace/models/indic-transcribe-flex"          # custom code always from base
EVAL = "/root/tts/TTS_training/pocket_TTS/infer/asr_test/mahadhwani100.json"
sys.path.insert(0, CODE)

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

def main(a):
    import jiwer, soundfile as sf, torchaudio
    from transformers import AutoModelForSpeechSeq2Seq
    from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    from tokenization_indic_canary import IndicCanaryTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        M, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    fe = IndicCanaryFeatureExtractor.from_pretrained(CODE, device=dev)
    tok = IndicCanaryTokenizer.from_pretrained(CODE)
    prompt = tok.encode_prompt(a.lang, itn=False, romanized=False)
    print(f"[flex] {M}  lang={a.lang} mode=native  device={dev}", flush=True)

    rows = json.load(open(EVAL))
    if a.limit: rows = rows[:a.limit]
    res, t0 = [], time.time()
    for i, r in enumerate(rows):
        wav, sr = sf.read(r["path"], dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
        if sr != fe.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, fe.sample_rate,
                                                 resampling_method="sinc_interp_hann")
        # Same padding the shipped wrapper uses: clips under 1 s are CENTRE-padded
        # to 1 s, and the padded length is what the encoder is told.
        min_len = fe.sample_rate
        n = wav.shape[0]
        if n < min_len:
            batch = torch.zeros(1, min_len); off = round((min_len - n) / 2)
            batch[0, off:off + n] = wav; lens = torch.tensor([min_len])
        else:
            batch, lens = wav.unsqueeze(0), torch.tensor([n])
        feats, feat_lens = fe(batch.to(dev), lens.to(dev))
        mask = (torch.arange(feats.size(2), device=dev)[None, :] < feat_lens[:, None]).long()
        ids = torch.tensor([prompt], device=dev)
        with torch.inference_mode():
            out = model.generate(input_features=feats, attention_mask=mask,
                                 decoder_input_ids=ids, max_new_tokens=a.max_new_tokens)
        hyp = tok.decode(tok.strip_prompt_and_trim(out[0].tolist(), prompt))
        rn, hn = norm(r["text"]), norm(hyp)
        res.append({"id": r["id"], "wer": jiwer.wer(rn, hn), "cer": jiwer.cer(rn, hn),
                    "dev_ratio": dev_ratio(hyp), "hyp": hyp, "ref": r["text"]})
        if i < 3:
            print(f"  [{i}] REF : {r['text'][:90]}\n  [{i}] PRED: {hyp[:90]}", flush=True)
    good = [r for r in res if r["dev_ratio"] >= 0.5]
    n_rom = len(res) - len(good)
    wer = sum(r["wer"] for r in good)/len(good); cer = sum(r["cer"] for r in good)/len(good)
    print(f"\n== flex[{a.tag}]: n={len(good)} (+{n_rom} discarded as romanized) "
          f"WER {wer:.3f}  CER {cer:.3f}   [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(f"/root/tts/TTS_training/whisper/flex/result_flex_{a.tag}.json","w"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="ne"); p.add_argument("--tag", default="base")
    p.add_argument("--limit", type=int); p.add_argument("--max-new-tokens", type=int, default=256)
    main(p.parse_args())

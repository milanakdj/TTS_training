"""WER/CER on wav/calib_v4/th{TH}_*.wav per eos_threshold, split by language.
English -> base whisper large-v3-turbo (usable on English, CLAUDE.md). Nepali
-> himalaya-ai/whisper-large-v3-nepali-final with the doubled-sot decode loop
(canonical version copied from score_wer_ft.py -- plain generate()/pipeline()
decode nonsense on this checkpoint). Run once per venv:

    ASR_VENV  score_calib_v4.py en <thresholds>
    SIM_VENV  score_calib_v4.py ne <thresholds>
"""
import json, os, re, string, sys, unicodedata
import jiwer

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
lang = sys.argv[1]
THS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["0.0", "-1.0", "-2.0", "-4.0"]
items = [it for it in json.load(open(f"{F}/calib_v4.json")) if it["lang"] == lang]

_EXTRA_PUNCT = "।॥‘’“”–—…"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + _EXTRA_PUNCT)

def normalize_text(t):
    t = unicodedata.normalize("NFC", t or "").translate(_PUNCT_TABLE).lower()
    return re.sub(r"\s+", " ", t).strip()

results = {}
if lang == "en":
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    def transcribe_file(f):
        segs, _ = model.transcribe(f, language="en", beam_size=5)
        return " ".join(s.text for s in segs).strip()
else:
    import torch, librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    MODEL_ID = "himalaya-ai/whisper-large-v3-nepali-final"
    HF_TOK = open("/root/.hf_whisper_ne_token").read().strip()
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="nepali", task="transcribe", token=HF_TOK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, token=HF_TOK).to("cuda").eval()
    tok = processor.tokenizer
    tid = lambda t: tok.convert_tokens_to_ids(t)
    PREFIX = [tid("<|startoftranscript|>"), tid("<|startoftranscript|>"),
              tid("<|ne|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")]
    EOS = tid("<|endoftext|>")

    def transcribe_batch(paths):
        audio = [librosa.load(p, sr=16000)[0] for p in paths]
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=model.dtype)
        cur = torch.tensor([PREFIX] * len(paths), device="cuda")
        out, past = cur, None
        done = torch.zeros(len(paths), dtype=torch.bool, device="cuda")
        with torch.no_grad():
            enc = model.get_encoder()(feats)
            for _ in range(225):
                res = model(encoder_outputs=enc, decoder_input_ids=cur, past_key_values=past, use_cache=True)
                past = res.past_key_values
                nxt = res.logits[:, -1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                out = torch.cat([out, nxt[:, None]], dim=1)
                done |= nxt == EOS
                if bool(done.all()):
                    break
                cur = nxt[:, None]
        return processor.batch_decode(out, skip_special_tokens=True)

for th in THS:
    wers, cers = [], []
    valid_items = [it for it in items if os.path.exists(f"{F}/wav/calib_v4/th{th}_{it['id']}.wav")]
    if lang == "en":
        for it in valid_items:
            hyp = transcribe_file(f"{F}/wav/calib_v4/th{th}_{it['id']}.wav")
            ref_n, hyp_n = normalize_text(it["text"]), normalize_text(hyp)
            if not ref_n:
                continue
            wers.append(jiwer.wer(ref_n, hyp_n))
            cers.append(jiwer.cer(ref_n, hyp_n))
    else:
        paths = [f"{F}/wav/calib_v4/th{th}_{it['id']}.wav" for it in valid_items]
        hyps = transcribe_batch(paths) if paths else []
        for it, hyp in zip(valid_items, hyps):
            ref_n, hyp_n = normalize_text(it["text"]), normalize_text(hyp)
            if not ref_n:
                continue
            wers.append(jiwer.wer(ref_n, hyp_n))
            cers.append(jiwer.cer(ref_n, hyp_n))
    n = len(wers)
    results[th] = {"n": n, "wer": sum(wers) / max(n, 1), "cer": sum(cers) / max(n, 1)}
    print(f"  lang={lang} th={th}: n={n} WER={results[th]['wer']:.3f} CER={results[th]['cer']:.3f}", flush=True)

out_path = f"{F}/calib_eos_v4_content_{lang}.json"
json.dump(results, open(out_path, "w"), indent=1)
print(f"wrote {out_path}")

"""Score the bilingual 2x2 set, per cell, with the right ASR for each language.

    $SIM_VENV score_2x2.py --part ne    <name> ...   # Nepali cells + ALL speaker sim
    $ASR_VENV score_2x2.py --part en    <name> ...   # English cells
    $SIM_VENV score_2x2.py --part merge <name> ...   # combine and print

Three passes because no single venv holds all three models: the Nepali Whisper
and resemblyzer live in vc_service/.venv, faster_whisper in asr_qc_service/.venv.

Nepali is scored with himalaya-ai/whisper-large-v3-nepali-final -- base Whisper is
blind on Nepali, rating REAL human speech at 0.917 WER. English is scored with
base large-v3-turbo, which is strong on English. The two columns are therefore NOT
comparable to each other; each is only readable against the real-human row in its
own cell.
"""
import json, os, re, sys, string, unicodedata

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
args = sys.argv[1:]
PART = "all"
if args and args[0] == "--part":
    PART, args = args[1], args[2:]
assert PART in ("ne", "en", "merge"), f"bad --part {PART!r}"
names = ["real_human"] + args
items = json.load(open(f"{F}/pairs_2x2.json"))
_TAB = str.maketrans("", "", string.punctuation + "।॥‘’“”–—…")

def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t or "").translate(_TAB).lower()).strip()

def dev_ratio(t):
    ch = [c for c in unicodedata.normalize("NFC", t or "")
          if not c.isspace() and unicodedata.category(c)[0] not in ("P", "S")]
    return sum(1 for c in ch if 0x900 <= ord(c) <= 0x97F) / len(ch) if ch else 0.0

def wav_of(name, it):
    return it["real_audio"] if name == "real_human" else f"{F}/wav/2x2_{name}/{it['id']}.wav"

def out_path(part):
    return f"{F}/eval_2x2_{part}.json"

# --------------------------------------------------------------------- merge
if PART == "merge":
    import statistics
    # JOIN on id, do not concatenate: the ne pass emits sim-only rows for the
    # English cells and the en pass emits wer/cer-only rows for those same ids,
    # so appending would double every English row and leave half of them without
    # a "wer"/"dev" key.
    merged = {}
    for part in ("ne", "en"):
        p = out_path(part)
        if not os.path.exists(p):
            sys.exit(f"missing {p}; run --part {part} first")
        for name, rows in json.load(open(p)).items():
            bucket = merged.setdefault(name, {})
            for r in rows:
                cur = bucket.setdefault(r["id"], {})
                cur.update({k: v for k, v in r.items() if v is not None or k not in cur})
    data = {n: list(rows.values()) for n, rows in merged.items()}
    for n, rows in data.items():
        miss = [r["id"] for r in rows if "cer" not in r]
        if miss:
            print(f"WARNING {n}: {len(miss)} rows never scored (e.g. {miss[:3]})")
    m = lambda xs: statistics.mean(xs) if xs else float("nan")
    print(f"\n{'system':<18} {'cell':<20} {'n':>4} {'WER':>7} {'CER':>7} {'SIM':>7}  note")
    print("-" * 84)
    for name in names:
        for cell in sorted({r["cell"] for r in data.get(name, [])}):
            g = [r for r in data[name] if r["cell"] == cell]
            good = [r for r in g if r.get("dev", 1.0) >= 0.5 and "cer" in r]
            sims = [r["sim"] for r in g if r.get("sim") is not None]
            note = "" if all(r["sim_ceiling"] for r in g) else "sim=cross-speaker floor"
            print(f"{name:<18} {cell:<20} {len(good):>4} {m([r['wer'] for r in good]):>7.3f} "
                  f"{m([r['cer'] for r in good]):>7.3f} {m(sims):>7.3f}  {note}")
    json.dump(data, open(f"{F}/eval_2x2.json", "w"), ensure_ascii=False, indent=1)
    print("\n2x2 MERGE DONE -> infer/final/eval_2x2.json")
    sys.exit(0)

# ---------------------------------------------------------------- scoring
# Import per branch: the ASR venv (faster_whisper, for English) has NEITHER
# torch NOR librosa, and the VC venv has no faster_whisper. Importing both
# unconditionally is what broke the first en pass.
import jiwer
want_lang = PART

if PART == "ne":
    import librosa, numpy as np, torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    NE_ID = "himalaya-ai/whisper-large-v3-nepali-final"
    TOK = open("/root/.hf_whisper_ne_token").read().strip()
    proc = WhisperProcessor.from_pretrained(NE_ID, language="nepali", task="transcribe", token=TOK)
    model = WhisperForConditionalGeneration.from_pretrained(
        NE_ID, torch_dtype=torch.float16, token=TOK).to(DEV).eval()
    tk = proc.tokenizer
    tid = lambda x: tk.convert_tokens_to_ids(x)
    # The doubled <|startoftranscript|> is required by this checkpoint.
    PREFIX = [tid("<|startoftranscript|>"), tid("<|startoftranscript|>"),
              tid("<|ne|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")]
    EOS = tid("<|endoftext|>")

    def transcribe(paths):
        audio = [librosa.load(p, sr=16000)[0] for p in paths]
        feats = proc.feature_extractor(audio, sampling_rate=16000,
            return_tensors="pt").input_features.to(DEV, dtype=model.dtype)
        cur = torch.tensor([PREFIX] * len(paths), device=DEV)
        out, past = cur, None
        done = torch.zeros(len(paths), dtype=torch.bool, device=DEV)
        with torch.no_grad():
            enc = model.get_encoder()(feats)
            for _ in range(225):
                r = model(encoder_outputs=enc, decoder_input_ids=cur,
                          past_key_values=past, use_cache=True)
                past = r.past_key_values
                nxt = r.logits[:, -1].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                out = torch.cat([out, nxt[:, None]], 1); done |= nxt == EOS
                if bool(done.all()):
                    break
                cur = nxt[:, None]
        return proc.batch_decode(out, skip_special_tokens=True)

    from resemblyzer import VoiceEncoder, preprocess_wav
    _enc = VoiceEncoder(device=DEV, verbose=False)
    _cache = {}
    def emb(p):
        if p not in _cache:
            try: _cache[p] = _enc.embed_utterance(preprocess_wav(p)).astype(np.float32)
            except Exception: _cache[p] = None
        return _cache[p]
    def sim(a, b):
        x, y = emb(a), emb(b)
        return None if x is None or y is None else \
            float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
else:
    DEV = os.environ.get("ASR_DEVICE", "cuda")   # no torch here to probe with
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3-turbo", device=DEV, compute_type="float16")
    def transcribe(paths):
        outs = []
        for p in paths:
            segs, _ = model.transcribe(p, language="en", beam_size=1)
            outs.append(" ".join(s.text.strip() for s in segs))
        return outs
    sim = lambda a, b: None          # speaker similarity is computed in the ne pass

results = {}
for name in names:
    rows = []
    for cell in sorted({i["cell"] for i in items}):
        grp = [i for i in items if i["cell"] == cell and os.path.exists(wav_of(name, i))]
        # the ne pass also covers speaker similarity for the English cells
        if not grp or (grp[0]["text_lang"] != want_lang and PART == "en"):
            continue
        score_text = grp[0]["text_lang"] == want_lang
        for i in range(0, len(grp), 8):
            ch = grp[i:i + 8]
            paths = [wav_of(name, x) for x in ch]
            hyps = transcribe(paths) if score_text else [None] * len(ch)
            for it, h, p in zip(ch, hyps, paths):
                ref = norm(it["text"])
                if not ref:
                    continue
                row = {"cell": cell, "lang": it["text_lang"], "id": it["id"],
                       "sim": sim(p, it["prompt"]), "sim_ceiling": it["sim_ceiling"]}
                if score_text:
                    hyp = norm(h)
                    row.update({"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp),
                                "dev": dev_ratio(h) if it["text_lang"] == "ne" else 1.0,
                                "hyp": h})
                rows.append(row)
    results[name] = rows
    n = len([r for r in rows if "cer" in r])
    print(f"  {name}: {n} scored, {len(rows)} rows", flush=True)

# merge into whatever this part already wrote (ne writes sim for every cell)
prev = json.load(open(out_path(PART))) if os.path.exists(out_path(PART)) else {}
prev.update(results)
json.dump(prev, open(out_path(PART), "w"), ensure_ascii=False, indent=1)
print(f"2x2 {PART.upper()} DONE -> {out_path(PART)}")

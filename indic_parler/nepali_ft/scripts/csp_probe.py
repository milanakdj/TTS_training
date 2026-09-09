"""CSP-FT stage 1: find WHICH decoder layers carry emotion.

Rationale (arXiv 2501.14273, CSP-FT): on four codec-LM TTS models, full fine-tuning
raised emotion recognition to 95.6% but TRIPLED WER (8.4 -> 25.1) and collapsed NMOS
4.25 -> 2.85. Fine-tuning just TWO layers chosen by an emotion probe matched the
emotion gain (95.5%) with WER essentially unchanged (9.2). Their ablation is U-shaped:
2 layers beats 3-4 layers, so picking the right two matters more than picking more.

That is precisely our failure -- three full fine-tunes, all "very bad quality".

Method here: freeze everything, teacher-force real audio codes through the decoder,
mean-pool each layer's hidden states, and train one cheap linear probe PER LAYER on
the emotion label. Layers whose probe scores highest are the ones that encode emotion.
"""
import json, os, random, collections
import numpy as np, soundfile as sf, torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

REPO = "ai4bharat/indic-parler-tts"; dev = "cuda:0"
R = "/workspace/proc_data_new/ai4bharat___rasa"
SRC, DST = "/projects/data/ttsteam/proc_data_new", "/workspace/proc_data_new"
EMO = {"ANGER": "angry", "HAPPY": "happy", "SAD": "sad",
       "FEAR": "fear", "DISGUST": "disgust", "SURPRISE": "surprise"}
PER = int(os.environ.get("PER_EMO", "150"))

rows = collections.defaultdict(list)
for split in ("train", "val"):
    for line in open(f"{R}/{split}/manifest.jsonl"):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("language") != "Nepali" or r.get("style") not in EMO: continue
        d = float(r.get("duration") or 0)
        if 1.5 <= d <= 8.0: rows[EMO[r["style"]]].append(r)
rng = random.Random(0)
sel = []
for e, v in rows.items():
    rng.shuffle(v); sel += [(e, x) for x in v[:PER]]
rng.shuffle(sel)
print("probe clips:", collections.Counter(e for e, _ in sel), flush=True)

model = ParlerTTSForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float16).to(dev).eval()
dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
NL = model.decoder.config.num_hidden_layers
print("decoder layers:", NL, flush=True)

# one fixed neutral caption for every clip: we want the layer signal to come from the
# AUDIO, not from the caption text
di = dtok("Amrita's neutral tone, with a moderate pitch voice, is captured with excellent "
          "quality in a close-sounding environment.", return_tensors="pt").to(dev)

feats, labels = [], []
with torch.inference_mode():
    enc = model.text_encoder(input_ids=di.input_ids, attention_mask=di.attention_mask).last_hidden_state
    for k, (e, r) in enumerate(sel):
        p = r["audio_filepath"].replace(SRC, DST)
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        wav = torch.from_numpy(y).view(1, 1, -1).to(dev).to(torch.float16)
        codes = model.audio_encoder.encode(input_values=wav)["audio_codes"]      # (1,B,K,T)
        codes = codes.squeeze(0)                                                  # (B,K,T)
        # No delay pattern needed: we only want representations, not generation.
        # Decoder consumes (bsz*num_codebooks, seq_len).
        K = model.decoder.num_codebooks
        dec_in = codes.reshape(codes.shape[0] * codes.shape[1], -1)[:K]
        out = model.decoder.model.decoder(
            input_ids=dec_in, encoder_hidden_states=enc,
            output_hidden_states=True, return_dict=True,
        )
        hs = out.hidden_states                       # tuple len NL+1, each (1,T,1024)
        v = np.stack([h.float().mean(1).squeeze(0).cpu().numpy() for h in hs[1:]])   # (NL,1024)
        feats.append(v); labels.append(e)
        if (k + 1) % 100 == 0: print(f"  {k+1}/{len(sel)}", flush=True)

X = np.stack(feats); y = np.array(labels)
print("features:", X.shape, flush=True)
np.save("/workspace/milan_nepali_parler_ft/csp_feats.npy", X)
np.save("/workspace/milan_nepali_parler_ft/csp_labels.npy", y)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
scores = []
for L in range(X.shape[1]):
    acc = cross_val_score(LogisticRegression(max_iter=3000), X[:, L, :], y, cv=4).mean()
    scores.append(acc); print(f"  layer {L:2d}  emotion-probe acc {acc:.3f}", flush=True)
order = np.argsort(scores)[::-1]
print("\nRANKED layers by emotion content:", [(int(i), round(scores[i], 3)) for i in order[:6]])
print("CSP-FT picks highest + lowest weighted layer:",
      f"best={int(order[0])} worst={int(order[-1])}")
json.dump({"scores": [float(s) for s in scores], "best": int(order[0]), "worst": int(order[-1])},
          open("/workspace/milan_nepali_parler_ft/csp_probe.json", "w"), indent=1)

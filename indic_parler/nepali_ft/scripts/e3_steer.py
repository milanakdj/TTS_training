"""E3: training-free emotion control by steering the caption embedding.

Parler routes the whole description through ONE tensor. NOTE: enc_to_dec_proj is
only built when the text-encoder and decoder hidden sizes differ; here both are 1024
so it does not exist and the text_encoder output IS the description tensor. Verified
config.prompt_cross_attention = False, so the transcript goes through embed_prompts
into the decoder input and never joins this tensor -- hooking the text encoder steers
the description alone.

Operator from EmoSteer-TTS (arXiv 2508.03543): u = mean(H_emo) - mean(H_neutral)
over paired captions, L2-normalised, applied as h' = h + a*u then renormalised to
the original norm. The renormalisation is what keeps it stable; EmoSteer reports
a=2 works and a=3 destroys intelligibility, so we sweep and score.

This is the highest-uncertainty item in the research: EmoSteer targets flow-matching
models and explicitly does not cover autoregressive codebook decoders. Treat as a
research probe, not a recipe.
"""
import os, sys, torch, numpy as np, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

OUT = "/workspace/milan_nepali_parler_ft/e3"
REPO = "ai4bharat/indic-parler-tts"
dev = "cuda:0"
model = ParlerTTSForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float16).to(dev).eval()
tok = AutoTokenizer.from_pretrained(REPO)
dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
SR = model.config.sampling_rate
print("prompt_cross_attention =", getattr(model.config, "prompt_cross_attention", None), flush=True)

# --- caption pairs differing ONLY in the style word (attribute-robust template, the
# variant that scored best at 38.9%) -------------------------------------------------
PITCH = ["high pitch", "low pitch", "slightly high pitch", "moderate pitch", "sharp"]
ENVS  = ["close-sounding", "slightly close-sounding", "very close-sounding"]
QUAL  = ["exceptional quality", "excellent quality", "great quality"]
def cap(style, i):
    return (f"Amrita's {style} tone, with a {PITCH[i % len(PITCH)]} voice, is captured with "
            f"{QUAL[i % len(QUAL)]} in a {ENVS[i % len(ENVS)]} environment.")

@torch.inference_mode()
def desc_hidden(text):
    d = dtok(text, return_tensors="pt").to(dev)
    h = model.text_encoder(input_ids=d.input_ids, attention_mask=d.attention_mask).last_hidden_state
    m = d.attention_mask[..., None].to(h.dtype)
    return (h * m).sum(1) / m.sum(1)          # mean over unmasked description tokens

EMOS = ["angry", "happy", "sad", "fear", "disgust", "surprise"]
N = 48
vecs = {}
base = torch.stack([desc_hidden(cap("neutral", i)).squeeze(0) for i in range(N)]).mean(0)
for e in EMOS:
    m = torch.stack([desc_hidden(cap(e, i)).squeeze(0) for i in range(N)]).mean(0)
    u = (m - base)
    vecs[e] = (u / u.norm()).to(torch.float16)
print("steering vectors built; norms:", {e: round(float(v.norm()), 3) for e, v in vecs.items()}, flush=True)

# --- hook enc_to_dec_proj so every generate() call is steered ------------------------
orig_te_forward = model.text_encoder.forward
STATE = {"u": None, "alpha": 0.0}
def steered(*a, **kw):
    out = orig_te_forward(*a, **kw)
    if STATE["u"] is not None and STATE["alpha"]:
        h = out.last_hidden_state
        n0 = h.norm(dim=-1, keepdim=True)
        h = h + STATE["alpha"] * STATE["u"].to(h.dtype).view(1, 1, -1)
        out.last_hidden_state = h * (n0 / h.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    return out
model.text_encoder.forward = steered

TEXTS = ["आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
         "उहाँ हिजो साँझ काठमाडौंबाट फर्कनुभयो।",
         "यो बाटो सिधै बजारसम्म पुग्छ।",
         "सबै जना भोलि यहाँ भेला हुनेछन्।",
         "पानी परेपछि सडक चिप्लो भयो।",
         "त्यो किताब टेबुलमाथि राखिएको छ।"]
ALPHAS = [float(a) for a in (sys.argv[1:] or ["1.0", "2.0", "3.0"])]

for a in ALPHAS:
    for e in EMOS:
        STATE["u"], STATE["alpha"] = vecs[e], a
        d = os.path.join(OUT, f"alpha{a:g}", e); os.makedirs(d, exist_ok=True)
        di = dtok(cap(e, 0), return_tensors="pt").to(dev)   # caption ALSO names the emotion
        for i, t in enumerate(TEXTS):
            torch.manual_seed(1234 + i)
            pi = tok(t, return_tensors="pt").to(dev)
            with torch.inference_mode():
                g = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                                   prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask,
                                   do_sample=True, temperature=0.8, max_new_tokens=1500)
            sf.write(os.path.join(d, f"{e}_{i:02d}.wav"),
                     g.to(torch.float32).cpu().numpy().squeeze(), SR)
        print(f"  alpha={a:g} {e:9s} done", flush=True)
print("DONE ->", OUT)

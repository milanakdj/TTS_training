"""E1: use the caption templates indic-parler was ACTUALLY TRAINED ON.

indic-parler is the RASMALAI model (arXiv 2505.18609). Its Table 1 gives the three
literal caption styles Llama-3.1-8B generated from structured attribute tags, and
the training style tag is a NAME ("Anger", "Sad"), phrased as "The intended style
is anger." Nothing I tried before used that string.

Speaker: Amrita only. The model card lists exactly one Nepali voice, and "Srijana"
-- which I had been using as the strong baseline -- is NOT in the model. That name
was out-of-vocabulary, which is why the listener judged it non-native and low
quality: the caption was falling back to unconstrained voice space.
"""
import os, torch, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

OUT = "/workspace/milan_nepali_parler_ft/e1"
REPO = "ai4bharat/indic-parler-tts"
dev = "cuda:0"
model = ParlerTTSForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float16).to(dev).eval()
tok = AutoTokenizer.from_pretrained(REPO)
dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
SR = model.config.sampling_rate
print("prompt_cross_attention:", getattr(model.config, "prompt_cross_attention", None), flush=True)

TEXTS = [
    "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
    "उहाँ हिजो साँझ काठमाडौंबाट फर्कनुभयो।",
    "यो बाटो सिधै बजारसम्म पुग्छ।",
    "सबै जना भोलि यहाँ भेला हुनेछन्।",
    "पानी परेपछि सडक चिप्लो भयो।",
    "त्यो किताब टेबुलमाथि राखिएको छ।",
]
# dir name -> RASMALAI style tag (the model card's own style vocabulary)
STYLES = {"angry": "anger", "happy": "happy", "sad": "sad",
          "fear": "fear", "disgust": "disgust", "surprise": "surprise"}
# per-emotion attribute clauses; attribute adherence (F0 96.6%, rate 97.1%) is far
# more reliable than the categorical tag, so make the attributes agree with it
ATTR = {
    "angry":    ("high pitch", "slightly fast"), "happy": ("high pitch", "slightly fast"),
    "sad":      ("low pitch", "slightly slow"),  "fear":  ("high pitch", "slightly fast"),
    "disgust":  ("low pitch", "moderate"),       "surprise": ("high pitch", "fast"),
}

def descriptive(s, p, r):
    return (f"Amrita, a female speaker, delivers speech in a close-sounding environment "
            f"with a {p}, expressive tone. She speaks at a {r} pace, with excellent "
            f"overall speech quality. The intended style is {s}.")
def concise(s, p, r):
    return (f"Amrita delivers {p}, expressive speech in a close-sounding environment at a "
            f"{r} pace, with excellent quality and a {s} tone.")
def attribute(s, p, r):
    return (f"Amrita's {s} tone, with a {p} voice, is captured with exceptional quality "
            f"in a close-sounding environment.")

VARIANTS = {"descriptive": descriptive, "concise": concise, "attribute": attribute}

for vname, fn in VARIANTS.items():
    for emo, style in STYLES.items():
        p, r = ATTR[emo]
        d = os.path.join(OUT, vname, emo); os.makedirs(d, exist_ok=True)
        desc = fn(style, p, r)
        di = dtok(desc, return_tensors="pt").to(dev)
        for i, t in enumerate(TEXTS):
            torch.manual_seed(1234 + i)
            pi = tok(t, return_tensors="pt").to(dev)
            with torch.inference_mode():
                g = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                                   prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask,
                                   do_sample=True, temperature=0.8, max_new_tokens=1500)
            sf.write(os.path.join(d, f"{emo}_{i:02d}.wav"),
                     g.to(torch.float32).cpu().numpy().squeeze(), SR)
        print(f"  {vname:12s} {emo:9s} | {desc[:88]}", flush=True)
print("DONE ->", OUT)

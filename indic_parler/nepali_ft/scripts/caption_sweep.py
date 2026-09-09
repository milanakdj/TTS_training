"""Caption sweep on the UNTOUCHED base model with Amrita.

Every finetune so far degraded audio quality (user: "very bad quality"), because
freezing the text encoder while training the full audio decoder damages exactly the
part that renders sound. Captions cost nothing and cannot hurt quality.

Caption wording is already known to matter enormously here: the same sentences went
from 37.5% to 87.5% angry purely on rewording + speaker name. The model card also
documents an "Expressivity" axis ("from monotone to highly expressive") that none of
the captions tried so far actually used -- that is the main thing under test.
"""
import os, json, torch, numpy as np, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

OUT = "/workspace/milan_nepali_parler_ft/sweep"
REPO = "ai4bharat/indic-parler-tts"
dev = "cuda:0"
model = ParlerTTSForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float16).to(dev).eval()
tok = AutoTokenizer.from_pretrained(REPO)
dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
SR = model.config.sampling_rate
S = "Amrita"
QUALITY = " The recording is very high quality, with her voice sounding clear and very close up."

TEXTS = [
    "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
    "उहाँ हिजो साँझ काठमाडौंबाट फर्कनुभयो।",
    "यो बाटो सिधै बजारसम्म पुग्छ।",
    "सबै जना भोलि यहाँ भेला हुनेछन्।",
]
# variant name -> template with {e} = emotion word
VARIANTS = {
    "plain":       "{S} speaks in a {e} tone.",
    "rasa_style":  "{S} speaks in an {e} tone, forceful and tense at a moderate pace.",
    "expressive":  "{S} speaks in a {e} tone. Her delivery is highly expressive and very animated.",
    "expr_pace":   "{S} speaks in a {e} tone, highly expressive and animated, at a fast pace with strong emphasis.",
    "intense":     "{S} sounds intensely {e}. Her voice is highly expressive, with dramatic pitch movement and heavy stress on each word.",
}
EMOS = ["angry", "happy", "sad"]

for vname, tpl in VARIANTS.items():
    for e in EMOS:
        d = os.path.join(OUT, vname, e); os.makedirs(d, exist_ok=True)
        desc = tpl.format(S=S, e=e) + QUALITY
        di = dtok(desc, return_tensors="pt").to(dev)
        for i, t in enumerate(TEXTS):
            torch.manual_seed(1234 + i)
            pi = tok(t, return_tensors="pt").to(dev)
            with torch.inference_mode():
                g = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                                   prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask,
                                   do_sample=True, temperature=0.8, max_new_tokens=1500)
            y = g.to(torch.float32).cpu().numpy().squeeze()
            sf.write(os.path.join(d, f"{e}_{i:02d}.wav"), y, SR)
        print(f"  {vname:12s} {e:6s} done", flush=True)
print("DONE ->", OUT)

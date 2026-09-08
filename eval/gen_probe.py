"""Generate an emotion probe set. Same captions, two languages -> isolates whether
weak emotion is a NEPALI gap or an artefact of the SER model not reading Nepali.
"""
import os, sys, numpy as np, soundfile as sf, torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

OUT = os.environ["OUTDIR"]; LANG = os.environ.get("LANG_SET", "ne")
REPO = "ai4bharat/indic-parler-tts"
dev = "cuda:0"
model = ParlerTTSForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float16).to(dev).eval()
tok  = AutoTokenizer.from_pretrained(REPO)
dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
SR = model.config.sampling_rate

NE = ["आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
      "उहाँ हिजो साँझ काठमाडौंबाट फर्कनुभयो।",
      "यो बाटो सिधै बजारसम्म पुग्छ।",
      "मैले त्यो चिठी आफैले पढेको हुँ।",
      "सबै जना भोलि यहाँ भेला हुनेछन्।",
      "त्यो किताब टेबुलमाथि राखिएको छ।",
      "पानी परेपछि सडक चिप्लो भयो।",
      "गाडी स्टेसनबाट बिहान छ बजे छुट्छ।"]
EN = ["The meeting will begin at ten o'clock in the hall.",
      "He returned from the city yesterday evening.",
      "This road goes straight to the market.",
      "I read that letter myself.",
      "Everyone will gather here tomorrow.",
      "The book is kept on the table.",
      "The road became slippery after the rain.",
      "The bus leaves the station at six in the morning."]
TEXTS = NE if LANG == "ne" else EN

# the user's own caption wording, verbatim
QUALITY = " The recording is very high quality, clear and close-sounding, with no background noise."
EMOTIONS = {
 "neutral": "Amrita speaks in a neutral tone at a moderate pace with balanced pitch.",
 "happy":   "Amrita speaks in a happy, cheerful tone, expressive and lively at a slightly fast pace.",
 "sad":     "Amrita speaks in a sad, sorrowful tone, subdued and slow, her delivery trailing downward.",
 "angry":   "Amrita speaks in an angry, forceful tone, sharp and tense with hard emphasis on each word.",
}

for emo, desc in EMOTIONS.items():
    d = os.path.join(OUT, emo); os.makedirs(d, exist_ok=True)
    di = dtok(desc + QUALITY, return_tensors="pt").to(dev)
    for i, t in enumerate(TEXTS):
        torch.manual_seed(1234 + i)
        pi = tok(t, return_tensors="pt").to(dev)
        with torch.inference_mode():
            g = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                               prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask,
                               do_sample=True, temperature=0.8, max_new_tokens=1500)
        y = g.to(torch.float32).cpu().numpy().squeeze()
        sf.write(os.path.join(d, f"{emo}_{i:02d}.wav"), y, SR)
        print(f"  {LANG} {emo:8s} {i} {len(y)/SR:5.2f}s", flush=True)
print("DONE")

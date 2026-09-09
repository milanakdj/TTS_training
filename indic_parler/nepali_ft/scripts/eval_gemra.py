"""A/B the gem+replay finetune against the untouched base, on BOTH caption styles.

Why two caption sets: the finetune's training captions (data_gem) use Gemini TTS
speaker names (Puck, Aoede, Kore, ...) and free-form prosody descriptions. The
base model's best-known config uses "Amrita" + the wording in best/CAPTIONS.txt.
Scoring only one style would be unfair to one of the two models, so score both:

  amrita_best -> does the finetune preserve the shippable base config?
  gem_style   -> did the finetune actually learn the new expressive control?

Protocol copied from best/CAPTIONS.txt: two tokenizers (description through
flan-t5-large, prompt through the repo), do_sample, temperature 0.8.
"""
import os, torch, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

ROOT = "/workspace/milan_nepali_parler_ft/eval_gemr"
MODELS = {
          "gemra": "/workspace/milan_nepali_parler_ft/out_gemra"}
dev = "cuda:0"

TEXTS = [
    "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
    "उहाँ हिजो साँझ काठमाडौंबाट फर्कनुभयो।",
    "यो बाटो सिधै बजारसम्म पुग्छ।",
    "सबै जना भोलि यहाँ भेला हुनेछन्।",
]
Q = " The recording is very high quality, with her voice sounding clear and very close up."
# verbatim from best/CAPTIONS.txt (sad had no winning variant; use the angry shape)
AMRITA = {
 "angry":   "Amrita sounds intensely angry. Her voice is highly expressive, with dramatic pitch movement and heavy stress on each word." + Q,
 "happy":   "Amrita speaks in a happy tone, highly expressive and animated, at a fast pace with strong emphasis." + Q,
 "neutral": "Amrita speaks in a neutral tone at a moderate pace with balanced pitch." + Q,
 "sad":     "Amrita sounds intensely sad. Her voice is subdued and heavy, with a low pitch and a slow, weary pace." + Q,
}
# data_gem's own caption shape: "<GeminiSpeaker> speaks in Nepali. <prosody>. <quality>"
GQ = " The recording is very high quality, with her voice sounding clear and very close up."
GEM = {
 "angry":   "Kore speaks in Nepali. Furious and confrontational, sharp fast pace, heavy stress on every word with a hard edge in the voice." + GQ,
 "happy":   "Kore speaks in Nepali. Delighted and animated, bright brisk pace, with a warm rising lilt throughout." + GQ,
 "neutral": "Kore speaks in Nepali. Matter-of-fact and even, steady moderate pace, balanced pitch throughout." + GQ,
 "sad":     "Kore speaks in Nepali. Sorrowful and downcast, slow heavy pace, with a falling pitch and a quiet, weary tone." + GQ,
}
CAPSETS = {"amrita_best": AMRITA}

for mname, mpath in MODELS.items():
    model = ParlerTTSForConditionalGeneration.from_pretrained(mpath, torch_dtype=torch.float16).to(dev).eval()
    tok  = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")          # prompt (Nepali text)
    dtok = AutoTokenizer.from_pretrained("google/flan-t5-large")  # description
    SR = model.config.sampling_rate
    for cname, caps in CAPSETS.items():
        for emo, desc in caps.items():
            d = f"{ROOT}/{mname}__{cname}/{emo}"
            os.makedirs(d, exist_ok=True)
            di = dtok(desc, return_tensors="pt").to(dev)
            n = 0
            for si, seed in enumerate((1234, 5678)):   # 4 texts x 2 seeds = 8 clips
                torch.manual_seed(seed)
                for ti, txt in enumerate(TEXTS):
                    pi = tok(txt, return_tensors="pt").to(dev)
                    with torch.no_grad():
                        a = model.generate(input_ids=di.input_ids,
                                           attention_mask=di.attention_mask,
                                           prompt_input_ids=pi.input_ids,
                                           prompt_attention_mask=pi.attention_mask,
                                           do_sample=True, temperature=0.8,
                                           max_new_tokens=1500)
                    sf.write(f"{d}/{emo}_{si}{ti}.wav",
                             a.cpu().numpy().squeeze().astype("float32"), SR)
                    n += 1
            print(f"{mname}/{cname}/{emo}: {n} clips", flush=True)
    del model; torch.cuda.empty_cache()
print("DONE ->", ROOT)

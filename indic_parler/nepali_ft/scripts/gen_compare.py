"""Generate the same emotion probe from the BASE and the FINETUNED checkpoint.

Captions use the wording the finetune was trained on (Rasa style vocabulary +
speaker name), so the comparison asks the fair question: given the caption this
model was taught, does the emotion come out?

Writes <out>/<tag>/<emotion>/*.wav, ready for eval/ser_eval.py.
"""
import argparse, os, torch, numpy as np, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="/workspace/milan_nepali_parler_ft/samples")
ap.add_argument("--speaker", default="Srijana")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--greedy", action="store_true")
args = ap.parse_args()

dev = "cuda:0"
model = ParlerTTSForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to(dev).eval()
# prompt tokenizer always from the base repo: the finetune does not change the vocab
tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
SR = model.config.sampling_rate

TEXTS = [
    "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
    "उहाँ हिजो साँझ काठमाडौंबाट फर्कनुभयो।",
    "यो बाटो सिधै बजारसम्म पुग्छ।",
    "मैले त्यो चिठी आफैले पढेको हुँ।",
    "सबै जना भोलि यहाँ भेला हुनेछन्।",
    "त्यो किताब टेबुलमाथि राखिएको छ।",
    "पानी परेपछि सडक चिप्लो भयो।",
    "गाडी स्टेसनबाट बिहान छ बजे छुट्छ।",
][: args.n]

S = args.speaker
QUALITY = " The recording is very high quality, with her voice sounding clear and very close up."
EMOTIONS = {
    "angry":    f"{S} speaks in an angry tone, forceful and tense at a moderate pace.",
    "happy":    f"{S} speaks in a happy, cheerful tone at a moderate pace.",
    "sad":      f"{S} speaks in a sad, sorrowful tone at a moderate pace.",
    "fear":     f"{S} speaks in a fearful, anxious tone at a moderate pace.",
    "surprise": f"{S} speaks in a surprised tone at a moderate pace.",
    "neutral":  f"{S} reads the line in a neutral, informative style at a moderate pace.",
}

root = os.path.join(args.out, args.tag)
for emo, desc in EMOTIONS.items():
    d = os.path.join(root, emo); os.makedirs(d, exist_ok=True)
    di = dtok(desc + QUALITY, return_tensors="pt").to(dev)
    for i, t in enumerate(TEXTS):
        torch.manual_seed(1234 + i)
        pi = tok(t, return_tensors="pt").to(dev)
        kw = dict(do_sample=False) if args.greedy else dict(do_sample=True, temperature=0.8)
        with torch.inference_mode():
            g = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                               prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask,
                               max_new_tokens=1500, **kw)
        y = g.to(torch.float32).cpu().numpy().squeeze()
        sf.write(os.path.join(d, f"{emo}_{i:02d}.wav"), y, SR)
        print(f"  {args.tag} {emo:9s} {i} {len(y)/SR:5.2f}s", flush=True)
print("DONE ->", root)

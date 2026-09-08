"""Does emotion transfer through pocket-TTS's VOICE PROMPT?

indic-parler carries emotion in a text caption, and four finetunes plus a layer
probe plus the user's ear all say that path does not strengthen. pocket-TTS
conditions on 5 s of reference AUDIO instead, and its teacher saw 13.7 h of Rasa
Nepali across six emotions. So: prompt it with an ANGRY Srijana clip and read a
NEUTRAL sentence it has never seen. If the output is angry, emotion rides the
audio prompt and the caption bottleneck is irrelevant.

References are Srijana only, so any prosody change is emotion and not a speaker
swap. Reference clips come from the training split (the valid split has almost no
styled Nepali), so this measures reproduction, not generalisation to new speakers --
but the TEXT is new, so it is not clip memorisation.
"""
import json, os, time, torch, scipy.io.wavfile as wav
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
OUT = f"{R}/infer/emo"
REFS = json.load(open(f"{R}/infer/emo_refs.json"))
# emotion-neutral sentences, none of them Rasa text
TEXTS = [
    "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
    "उहाँ हिजो साँझ काठमाडौंबाट फर्कनुभयो।",
    "यो बाटो सिधै बजारसम्म पुग्छ।",
    "सबै जना भोलि यहाँ भेला हुनेछन्।",
]
model = TTSModel.load_model(config=f"{R}/infer/nepali_teacher_24l.yaml")
model.to("cpu")
SR = model.config.mimi.sample_rate
t0 = time.time()
for emo, refs in sorted(REFS.items()):
    d = f"{OUT}/{emo}"; os.makedirs(d, exist_ok=True)
    for ri, ref in enumerate(refs):
        st = model.get_state_for_audio_prompt(ref["path"])
        for ti, txt in enumerate(TEXTS):
            a = model.generate_audio(st, txt, copy_state=True)
            wav.write(f"{d}/{emo}_r{ri}t{ti}.wav", SR,
                      a.detach().cpu().numpy().squeeze())
    print(f"{emo}: {len(refs)*len(TEXTS)} clips  [{time.time()-t0:.0f}s]", flush=True)
# keep the reference clips beside the output so they can be compared by ear
os.makedirs(f"{OUT}/_references", exist_ok=True)
import shutil
for emo, refs in REFS.items():
    for ri, ref in enumerate(refs):
        shutil.copy(ref["path"], f"{OUT}/_references/{emo}_ref{ri}.wav")
print("DONE ->", OUT)

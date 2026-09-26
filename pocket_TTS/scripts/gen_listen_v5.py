"""5 clips per text/voice cell from a v5 teacher checkpoint, guided (cfg 2.0), for listening.

Same sampling as infer/final/gen_2x2_guided.py (seed, temp, cfg) so clips match
the full 2x2 ids. Writes listen/v5_teacher/<TAG>/{nepali,english}/{..}_voice/.
    cd repo && CKPT=... TAG=80k .venv/bin/python3 ../scripts/gen_listen_v5.py
"""
import json, os, sys, time
import soundfile as sf
import torch

sys.path.insert(0, "/root/tts/TTS_training/pocket_TTS/repo")
from training.eval.librispeech import load_run, load_mono, latents_to_wav

R = "/root/tts/TTS_training/pocket_TTS"; F = f"{R}/infer/final"
CKPT, TAG = os.environ["CKPT"], os.environ["TAG"]
N = int(os.environ.get("N", "5"))
LANG = {"ne": "nepali", "en": "english"}
DEV = torch.device("cuda")

items, seen = [], {}
for it in json.load(open(f"{F}/pairs_2x2.json")):
    if seen.setdefault(it["cell"], 0) < N:
        seen[it["cell"]] += 1
        items.append(it)

model, mimi, step = load_run("/workspace/v5_teacher_24l", DEV, use_ema=True, checkpoint=CKPT)
tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
print(f"{TAG}: step {step}, {len(items)} clips", flush=True)
t0 = time.time()
for it in items:
    out = f"{R}/listen/v5_teacher/{TAG}/{LANG[it['text_lang']]}/{LANG[it['voice_lang']]}_voice"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(1234 + sum(map(ord, it["id"])))
    toks = torch.tensor(tokenize(it["text"]), dtype=torch.long)
    with torch.no_grad():
        ref = mimi.encode_to_latent(load_mono(it["prompt"], mimi.sample_rate)[None, None].to(DEV))[0]
        lat = model.generate([toks], [ref], max_frames=375, temp=0.3, n_steps=1, cfg_coef=2.0)[0]
        w = latents_to_wav(mimi, lat, DEV)
    if w is None:
        print(f"  EMPTY {it['id']}", flush=True)
        continue
    sf.write(f"{out}/{it['id']}.wav", w.float().cpu().numpy(), mimi.sample_rate)
    with open(f"{out}/texts.tsv", "a") as f:
        f.write(f"{it['id']}.wav\t{it['text']}\t{it['prompt']}\n")
print(f"{TAG} DONE in {time.time()-t0:.0f}s", flush=True)

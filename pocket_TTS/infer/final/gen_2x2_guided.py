"""Bilingual 2x2 + code-mix probes from a TRAINING checkpoint, guided (cfg 2.0).

The 7.5k-step v5 probes are teachers: the inference package has no cfg_coef, and
unguided teacher numbers mislead (memory: pocket-tts-teacher-eval-needs-cfg). The
guided output is what a student distils from, so compare probe arms here, all at
the same step and cfg. Writes wav/2x2_<NAME>/ (read by score_2x2.py) and
wav/codemix_<NAME>/ (codemix_probes.json, for listening). Run with repo/.venv:

    cd repo && RUN=/workspace/v5s NAME=v5s_7k5 .venv/bin/python3 ../infer/final/gen_2x2_guided.py
"""
import json, os, sys, time
import soundfile as sf
import torch

sys.path.insert(0, "/root/tts/TTS_training/pocket_TTS/repo")
from training.eval.librispeech import load_run, load_mono, latents_to_wav

F = "/root/tts/TTS_training/pocket_TTS/infer/final"
RUN, NAME = os.environ["RUN"], os.environ["NAME"]
CKPT = os.environ.get("CKPT") or None
CFG = float(os.environ.get("CFG", "2.0"))
TEMP = float(os.environ.get("TEMP", "0.3"))
DEV = torch.device("cuda")

model, mimi, step = load_run(RUN, DEV, use_ema=True, checkpoint=CKPT)
tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
print(f"{NAME}: step {step} from {RUN}, cfg {CFG}, temp {TEMP}", flush=True)
prompts = {}


def prompt_latent(path):
    if path not in prompts:
        with torch.no_grad():
            prompts[path] = mimi.encode_to_latent(
                load_mono(path, mimi.sample_rate)[None, None].to(DEV))[0]
    return prompts[path]


def run(items, out):
    os.makedirs(out, exist_ok=True)
    for it in items:                       # stale audio poisons results: purge
        f = f"{out}/{it['id']}.wav"
        if os.path.exists(f):
            os.remove(f)
    t0, empty = time.time(), 0
    for it in items:
        torch.manual_seed(1234 + sum(map(ord, it["id"])))
        toks = torch.tensor(tokenize(it["text"]), dtype=torch.long)
        with torch.no_grad():
            lat = model.generate([toks], [prompt_latent(it["prompt"])], max_frames=375,
                                 temp=TEMP, n_steps=1, cfg_coef=CFG)[0]
            w = latents_to_wav(mimi, lat, DEV)
        if w is None:
            empty += 1
            continue
        sf.write(f"{out}/{it['id']}.wav", w.float().cpu().numpy(), mimi.sample_rate)
    print(f"  {len(items)-empty}/{len(items)} -> {out} ({empty} empty) [{time.time()-t0:.0f}s]",
          flush=True)


run(json.load(open(f"{F}/pairs_2x2.json")), f"{F}/wav/2x2_{NAME}")
run(json.load(open(f"{F}/codemix_probes.json")), f"{F}/wav/codemix_{NAME}")
print("GUIDED 2x2 DONE")

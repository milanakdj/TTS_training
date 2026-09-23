"""Synthesize the bilingual 2x2 set with one model. Run with repo/.venv.

    cd repo && .venv/bin/python3 ../infer/final/gen_2x2.py <name> <config.yaml>
    EOS_THRESHOLD=0.0 TEMP=0.3 DEVICE=cuda ... optional overrides

Writes wav/2x2_<name>/<id>.wav. Scoring is a separate script because the ASR and
speaker-similarity models live in a different venv.
"""
import json, os, sys, time
import torch
import scipy.io.wavfile as wav
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
name, config = sys.argv[1], sys.argv[2]
items = json.load(open(f"{F}/pairs_2x2.json"))
if int(os.environ.get("EVAL_LIMIT", "0")):
    items = items[: int(os.environ["EVAL_LIMIT"])]

kw = {}
if os.environ.get("EOS_THRESHOLD"):
    kw["eos_threshold"] = float(os.environ["EOS_THRESHOLD"])
if os.environ.get("TEMP"):
    kw["temp"] = float(os.environ["TEMP"])
torch.set_num_threads(int(os.environ.get("GEN_THREADS", "8")))
out = f"{F}/wav/2x2_{name}"
os.makedirs(out, exist_ok=True)
# Stale audio from another checkpoint silently entering results is the known way
# to get a wrong answer out of this harness. Purge, never skip.
for it in items:
    f = f"{out}/{it['id']}.wav"
    if os.path.exists(f):
        os.remove(f)

m = TTSModel.load_model(config=config, **kw)
m.to(os.environ.get("DEVICE", "cpu"))
SR = m.config.mimi.sample_rate
print(f"== {name}: {config}  kw={kw or 'defaults'}  n={len(items)}", flush=True)

t0, done, fail = time.time(), 0, []
for it in items:
    try:
        st = m.get_state_for_audio_prompt(it["prompt"])
        torch.manual_seed(1234 + int(it["id"].split("_")[1]))
        a = m.generate_audio(st, it["text"], copy_state=True)
        x = a.detach().cpu().numpy().squeeze()
        if len(x) / SR < 0.3:
            raise RuntimeError(f"degenerate {len(x)/SR:.2f}s")
        wav.write(f"{out}/{it['id']}.wav", SR, x)
        done += 1
    except Exception as e:
        fail.append((it["id"], str(e)[:100]))
    if (done + len(fail)) % 25 == 0:
        print(f"  {done+len(fail)}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
print(f"== {name}: {done} ok, {len(fail)} failed in {time.time()-t0:.0f}s -> {out}", flush=True)
for i, e in fail[:10]:
    print(f"   FAIL {i}: {e}")
print("2x2 GEN DONE")

"""Synthesize the held-out eval set with both models, on CPU.

Same prompt, same text, same seed per utterance for both models, so the only
difference between the two output sets is backbone depth (24 -> 6 layers).
"""
import json, os, time, sys
import torch
import scipy.io.wavfile as wav
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
CFG = {"teacher_24l": f"{R}/infer/nepali_teacher_24l.yaml",
       "student_6l":  f"{R}/infer/nepali_student_6l.yaml"}
pairs = json.load(open(f"{F}/pairs.json"))
_lim = int(os.environ.get("EVAL_LIMIT", "0"))
if _lim:
    pairs = pairs[:_lim]
only = sys.argv[1] if len(sys.argv) > 1 else None

torch.set_num_threads(int(os.environ.get("GEN_THREADS", "8")))
for name, cfg in CFG.items():
    if only and name != only:
        continue
    out = f"{F}/wav/{name}"
    os.makedirs(out, exist_ok=True)
    m = TTSModel.load_model(config=cfg); m.to("cpu")
    SR = m.config.mimi.sample_rate
    t_start, done, failed = time.time(), 0, []
    for p in pairs:
        dst = f"{out}/{p['id']}.wav"
        if os.path.exists(dst):
            done += 1
            continue
        try:
            st = m.get_state_for_audio_prompt(p["prompt"])
            torch.manual_seed(1234 + int(p["id"]))
            a = m.generate_audio(st, p["text"], copy_state=True)
            n = a.detach().cpu().numpy().squeeze()
            if len(n) / SR < 0.3:
                raise RuntimeError(f"degenerate output, {len(n)/SR:.2f}s")
            wav.write(dst, SR, n)
            done += 1
        except Exception as e:
            failed.append((p["id"], str(e)[:120]))
        if (done + len(failed)) % 10 == 0:
            print(f"  {name}: {done+len(failed)}/{len(pairs)} "
                  f"({time.time()-t_start:.0f}s)", flush=True)
    print(f"== {name}: {done} ok, {len(failed)} failed in {time.time()-t_start:.0f}s", flush=True)
    for fid, err in failed[:10]:
        print(f"   FAIL {fid}: {err}", flush=True)
    del m
print("GEN DONE")

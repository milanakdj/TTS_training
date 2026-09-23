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
CFG = {"teacher_24l":    f"{R}/infer/nepali_teacher_24l.yaml",
       "student_6l":     f"{R}/infer/nepali_student_6l.yaml",
       "teacher_24l_v3": f"{R}/infer/nepali_teacher_24l_v3.yaml",
       "student_6l_v3":  f"{R}/infer/nepali_student_6l_v3.yaml"}

# Per-model EOS threshold. The v1/v2 models terminate correctly at the shipped
# default (-4.0). The v3 models do not: their backbone regressed onto the
# teacher's cfg-2.0-combined activations while out_eos stayed frozen, so the
# head is miscalibrated against the z it now sees and fires within a few frames.
# These values are chosen on calib_v3.json -- 40 valid-split utterances with the
# 100 eval ids removed -- by duration fidelity, never on pairs.json.
# See calib_eos_{model}.json and docs/V3_PLAN.md.
EOS = json.load(open(f"{F}/eos_thresholds.json")) if os.path.exists(f"{F}/eos_thresholds.json") else {}
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
    kw = {"eos_threshold": EOS[name]} if name in EOS else {}
    m = TTSModel.load_model(config=cfg, **kw); m.to("cpu")
    print(f"== {name}: eos_threshold={kw.get('eos_threshold', 'default -4.0')}", flush=True)
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

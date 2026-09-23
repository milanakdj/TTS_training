"""Sweep eos_threshold on the calibration set and report duration fidelity.

Criterion: median |log(generated / reference duration)|, plus the share of
degenerate clips (<0.3 s, the same floor gen_final.py rejects on). A model that
terminates correctly tracks the reference duration; one whose EOS head is
miscalibrated does not, and no amount of temperature fixes it.
"""
import json, os, sys, time, math, statistics
import numpy as np, torch
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
name = sys.argv[1]
cfgs = {"student_6l_v3": f"{R}/infer/nepali_student_6l_v3.yaml",
        "teacher_24l_v3": f"{R}/infer/nepali_teacher_24l_v3.yaml",
        "student_6l": f"{R}/infer/nepali_student_6l.yaml"}
THS = [float(x) for x in os.environ.get("THS", "-4,-2,-1,0,1,2,4").split(",")]
items = json.load(open(f"{F}/calib_v3.json"))[: int(os.environ.get("N", "40"))]
torch.set_num_threads(int(os.environ.get("GEN_THREADS", "8")))

DEV = os.environ.get("CALIB_DEVICE", "cpu")
print(f"{name}: device={DEV}, n={len(items)}", flush=True)
res = {}
for th in THS:
    m = TTSModel.load_model(config=cfgs[name], eos_threshold=th)
    m.to(os.environ.get("CALIB_DEVICE", "cpu"))
    SR = m.config.mimi.sample_rate
    ratios, degen, t0 = [], 0, time.time()
    for it in items:
        st = m.get_state_for_audio_prompt(it["prompt"])
        torch.manual_seed(1234)
        x = m.generate_audio(st, it["text"], copy_state=True).detach().cpu().numpy().squeeze()
        d = len(x) / SR
        if d < 0.3:
            degen += 1
        ratios.append(math.log(max(d, 1e-3) / it["duration"]))
    med = statistics.median(abs(r) for r in ratios)
    bias = statistics.median(ratios)
    res[th] = {"abs_log_err": med, "median_log_ratio": bias, "degenerate": degen,
               "mean_dur_ratio": float(np.exp(statistics.mean(ratios)))}
    print(f"  eos_threshold {th:>5}: |log err| {med:.3f}  bias {bias:+.3f} "
          f"(x{math.exp(bias):.2f} duration)  degenerate {degen}/{len(items)}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    del m
best = min(res, key=lambda t: (res[t]["degenerate"] > 0, res[t]["abs_log_err"]))
print(f"\nBEST eos_threshold for {name}: {best}  (|log err| {res[best]['abs_log_err']:.3f})")
json.dump({"model": name, "n": len(items), "results": {str(k): v for k, v in res.items()},
           "best": best}, open(f"{F}/calib_eos_{name}.json", "w"), indent=1)

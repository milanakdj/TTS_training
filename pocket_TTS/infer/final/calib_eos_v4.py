"""Sweep eos_threshold on calib_v4.json (bilingual, disjoint from pairs_2x2.json)
for the v4 student. Reports duration fidelity AND WER/CER per language, because
duration is a known-lying proxy here (v3 trims trailing silence in training
while manifest duration still counts it) -- see pocket-tts-v3-eos-miscalibrated.
English is scored with base whisper (usable on English, per CLAUDE.md), Nepali
with the finetuned Nepali whisper (doubled-sot prefix).
"""
import json, os, sys, time, math, statistics
import numpy as np, torch
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
CFG = f"{R}/infer/nepali_student_6l_v4.yaml"
THS = [float(x) for x in os.environ.get("THS", "0,-1,-2,-4").split(",")]
items = json.load(open(f"{F}/calib_v4.json"))
DEV = os.environ.get("CALIB_DEVICE", "cuda")
torch.set_num_threads(int(os.environ.get("GEN_THREADS", "8")))

out_dir = f"{F}/wav/calib_v4"
os.makedirs(out_dir, exist_ok=True)

res = {}
for th in THS:
    m = TTSModel.load_model(config=CFG, eos_threshold=th)
    m.to(DEV)
    SR = m.config.mimi.sample_rate
    ratios, degen = [], 0
    t0 = time.time()
    import scipy.io.wavfile as wav
    for it in items:
        st = m.get_state_for_audio_prompt(it["prompt"])
        torch.manual_seed(1234)
        x = m.generate_audio(st, it["text"], copy_state=True).detach().cpu().numpy().squeeze()
        d = len(x) / SR
        if d < 0.3:
            degen += 1
        ratios.append(math.log(max(d, 1e-3) / it["duration"]))
        wav.write(f"{out_dir}/th{th}_{it['id']}.wav", SR, x)
    med = statistics.median(abs(r) for r in ratios)
    bias = statistics.median(ratios)
    res[th] = {"abs_log_err": med, "median_log_ratio": bias, "degenerate": degen,
               "mean_dur_ratio": float(np.exp(statistics.mean(ratios)))}
    print(f"  eos_threshold {th:>5}: |log err| {med:.3f}  bias {bias:+.3f} "
          f"(x{math.exp(bias):.2f} duration)  degenerate {degen}/{len(items)}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    del m
    torch.cuda.empty_cache()

json.dump({"n": len(items), "results": {str(k): v for k, v in res.items()}},
          open(f"{F}/calib_eos_v4_duration.json", "w"), indent=1)
print("DURATION SWEEP DONE -- now run score_calib_v4.py for WER/CER per threshold")

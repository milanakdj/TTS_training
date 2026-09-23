"""Generate the calibration utterances at several eos_threshold values.

Duration fidelity is only a proxy. v3 trims trailing silence during training
(the aligner gives it word timings that v2 partly lacked), while the manifest
duration still counts that silence -- so a duration ratio slightly under 1.0 is
correct behaviour, not truncation. The threshold therefore has to be chosen on
content: score these clips with score_calib_wavs.py and take the best CER.

    cd repo && .venv/bin/python3 ../infer/final/gen_calib_wavs.py student_6l_v3
"""
import json, os, sys, time
import torch
import scipy.io.wavfile as wav
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
name = sys.argv[1]
CFG = {"student_6l_v3": f"{R}/infer/nepali_student_6l_v3.yaml",
       "teacher_24l_v3": f"{R}/infer/nepali_teacher_24l_v3.yaml"}[name]
THS = [float(x) for x in os.environ.get("THS", "0,1,2,4").split(",")]
DEV = os.environ.get("CALIB_DEVICE", "cpu")
items = json.load(open(f"{F}/calib_v3.json"))
torch.set_num_threads(int(os.environ.get("GEN_THREADS", "8")))

for th in THS:
    tag = f"_t{os.environ['TEMP']}" if os.environ.get("TEMP") else ""
    out = f"{F}/wav/calib_{name}{tag}_th{th:g}"
    os.makedirs(out, exist_ok=True)
    for it in items:                      # purge: stale audio silently enters results
        f = f"{out}/{it['id']}.wav"
        if os.path.exists(f):
            os.remove(f)
    TEMP = os.environ.get("TEMP")
    m = TTSModel.load_model(config=CFG, eos_threshold=th,
                            **({"temp": float(TEMP)} if TEMP else {}))
    m.to(DEV)
    SR = m.config.mimi.sample_rate
    t0 = time.time()
    for it in items:
        st = m.get_state_for_audio_prompt(it["prompt"])
        torch.manual_seed(1234)
        x = m.generate_audio(st, it["text"], copy_state=True).detach().cpu().numpy().squeeze()
        wav.write(f"{out}/{it['id']}.wav", SR, x)
    print(f"  th {th:>5}: wrote {len(items)} clips to {out} [{time.time()-t0:.0f}s]", flush=True)
    del m
print("CALIB GEN DONE")

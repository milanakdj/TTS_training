"""Teacher (24L) vs student (6L) on CPU: does the distill deliver the speedup?

The teacher measured 0.73x real-time, i.e. slower than real-time and not
deployable -- that is the entire reason the distill stage exists. Same voice
prompt, same sentences, same temperature for both, so the only difference is
backbone depth (24 -> 6 layers).
"""
import time, torch, json, os
import scipy.io.wavfile as wav
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
VOICE = "/workspace/proc_data_new/indicvoices-r/train/audio/Nepali/mdx_extra/v2/train/audios/3940649674021891_chunk_1.wav"
TEXTS = [
    "नेपाल विश्वको सबैभन्दा सुन्दर देशहरू मध्ये एक हो।",
    "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
    "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ।",
]
CFG = {"teacher_24l": f"{R}/infer/nepali_teacher_24l.yaml",
       "student_6l":  f"{R}/infer/nepali_student_6l.yaml"}
# Oversubscribing (16 threads on a 16-core box that the distill job already
# had at load ~13.6) made both models thrash and measured 0.15x/0.17x RT --
# no speedup, which is what profile_stages.py disproved. Pin the thread count.
torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "4")))
res = {}
for name, cfg in CFG.items():
    m = TTSModel.load_model(config=cfg); m.to("cpu")
    SR = m.config.mimi.sample_rate
    st = m.get_state_for_audio_prompt(VOICE)
    d = f"{R}/infer/bench/{name}"; os.makedirs(d, exist_ok=True)
    rows = []
    for i, t in enumerate(TEXTS):
        t0 = time.time()
        a = m.generate_audio(st, t, copy_state=True)
        el = time.time() - t0
        n = a.detach().cpu().numpy().squeeze()
        secs = len(n) / SR
        wav.write(f"{d}/bench_{i}.wav", SR, n)
        rows.append({"audio_s": round(secs, 2), "wall_s": round(el, 2),
                     "rtf": round(secs / el, 2)})
        print(f"{name} [{i}] {secs:5.2f}s audio in {el:6.2f}s -> {secs/el:4.2f}x RT", flush=True)
    res[name] = rows
    mean = sum(r["rtf"] for r in rows) / len(rows)
    print(f"== {name}: mean {mean:.2f}x real-time ==\n", flush=True)
    del m
json.dump(res, open(f"{R}/infer/bench/results.json", "w"), indent=2)
print("DONE")

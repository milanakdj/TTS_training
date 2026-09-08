"""Why is the 6L student no faster than the 24L teacher on CPU?

bench_student.py measured 0.15x RT (teacher) vs 0.17x RT (student) -- a 1.13x
speedup where cutting the backbone 24 -> 6 layers should give ~3x. Two candidate
explanations, and they are distinguishable:

  (a) measurement contamination -- the bench ran while the distill job had the
      box at load ~13.6/16 and set_num_threads(16) on top of that;
  (b) Amdahl -- generate_audio_stream runs the AR loop and the mimi decoder in
      two threads, so wall time is ~max(AR, mimi). mimi is byte-identical
      between the two models, so if mimi is the slower stage the backbone cut
      cannot show up at all.

This splits the two stages and times them separately, single-threaded loop, so
the backbone cut is visible in the AR number regardless of what mimi does.
"""
import copy, queue, time, json, os, sys
import torch
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import increment_steps, init_states

R = "/root/tts/TTS_training/pocket_TTS"
VOICE = "/workspace/proc_data_new/indicvoices-r/train/audio/Nepali/mdx_extra/v2/train/audios/3940649674021891_chunk_1.wav"
TEXTS = [
    "नेपाल विश्वको सबैभन्दा सुन्दर देशहरू मध्ये एक हो।",
    "आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ।",
    "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ।",
]
CFG = {"teacher_24l": f"{R}/infer/nepali_teacher_24l.yaml",
       "student_6l":  f"{R}/infer/nepali_student_6l.yaml"}

NTHREADS = int(os.environ.get("PROF_THREADS", "4"))
torch.set_num_threads(NTHREADS)
print(f"threads={NTHREADS}  loadavg={open('/proc/loadavg').read().split()[:3]}", flush=True)

res = {}
for name, cfg in CFG.items():
    m = TTSModel.load_model(config=cfg); m.to("cpu")
    SR = m.config.mimi.sample_rate
    base_state = m.get_state_for_audio_prompt(VOICE)
    rows = []
    for i, text in enumerate(TEXTS):
        state = copy.deepcopy(base_state)
        prepared = m.flow_lm.conditioner.prepare(text)
        max_gen_len = m._estimate_max_gen_len(prepared.shape[1])
        steps_per_latent = int(m.mimi.encoder_frame_rate / m.mimi.frame_rate)
        mimi_seq_len = max_gen_len * steps_per_latent

        # --- stage 1: autoregressive backbone + flow head, no decoder thread ---
        q: queue.Queue = queue.Queue()
        t0 = time.monotonic()
        m._generate(model_state=state, prepared=prepared, max_gen_len=max_gen_len,
                    frames_after_eos=3, latents_queue=q, result_queue=queue.Queue())
        latents = []
        while True:
            x = q.get()
            if x is None:
                break
            latents.append(x)
        ar_s = time.monotonic() - t0

        # --- stage 2: mimi decode of exactly those latents ---
        mimi_state = init_states(m.mimi, batch_size=1, sequence_length=mimi_seq_len)
        nsamp = 0
        t0 = time.monotonic()
        for lat in latents:
            inp = lat * m.flow_lm.emb_std + m.flow_lm.emb_mean
            frame = m.mimi.decode_from_latent(inp, mimi_state)
            increment_steps(m.mimi, mimi_state, increment=steps_per_latent)
            nsamp += frame.shape[-1]
        mimi_s = time.monotonic() - t0

        audio_s = nsamp / SR
        row = {"frames": len(latents), "audio_s": round(audio_s, 2),
               "ar_s": round(ar_s, 2), "mimi_s": round(mimi_s, 2),
               "ar_ms_per_frame": round(1000 * ar_s / max(len(latents), 1), 1),
               "mimi_ms_per_frame": round(1000 * mimi_s / max(len(latents), 1), 1),
               "ar_rtf": round(audio_s / ar_s, 2), "mimi_rtf": round(audio_s / mimi_s, 2)}
        rows.append(row)
        print(f"{name} [{i}] {row['frames']:4d} fr {row['audio_s']:5.2f}s audio | "
              f"AR {row['ar_s']:6.2f}s ({row['ar_ms_per_frame']:6.1f} ms/fr) | "
              f"mimi {row['mimi_s']:6.2f}s ({row['mimi_ms_per_frame']:6.1f} ms/fr)", flush=True)
    res[name] = rows
    ar = sum(r["ar_ms_per_frame"] for r in rows) / len(rows)
    mi = sum(r["mimi_ms_per_frame"] for r in rows) / len(rows)
    print(f"== {name}: AR {ar:.1f} ms/frame | mimi {mi:.1f} ms/frame | "
          f"pipelined floor {max(ar, mi):.1f} ms/frame vs 80 ms/frame real-time ==\n", flush=True)
    del m

t = sum(r["ar_ms_per_frame"] for r in res["teacher_24l"]) / 3
s = sum(r["ar_ms_per_frame"] for r in res["student_6l"]) / 3
print(f"backbone speedup (AR only): {t / s:.2f}x")
json.dump(res, open(f"{R}/infer/bench/stage_profile.json", "w"), indent=2)
print("DONE")

"""Generate Nepali samples from the stage-1 teacher on CPU, so this never
contends with whatever is training on the GPU."""
import subprocess, sys, os, time

R = "/root/tts/TTS_training/pocket_TTS"
CFG = f"{R}/infer/nepali_teacher_24l.yaml"
OUT = f"{R}/infer/out"
VOICES = {
    # studio read speech, cleanest prompt in the valid split
    "ivr": "/workspace/proc_data_new/indicvoices-r/train/audio/Nepali/mdx_extra/v2/train/audios/3940649674021891_chunk_1.wav",
    # spontaneous YouTube speech, the bulk of the training distribution
    "ans": "/workspace/proc_data_new/ans_snr40-50.part2of2/train/audio/automated_new_set/UCecn1j82-uxXDMCSOMYaqKw/O7yfgpoXWoM/chunk_302_4.96s.wav",
}
TEXTS = {
    "01_short":    "नमस्ते, तपाईंलाई कस्तो छ?",
    "02_news":     "प्रधानमन्त्रीले आज संसदमा नयाँ बजेट प्रस्तुत गर्नुभयो।",
    "03_long":     "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ, र यहाँ एक सय बीसभन्दा बढी भाषा बोलिन्छन्।",
    "04_question": "तिमीले भनेको कुरा मैले राम्ररी सुनेको छैन, फेरि भन्न सक्छौ?",
    "05_numeric":  "सन् दुई हजार पच्चीसमा कुल जनसंख्या तीन करोड पुग्यो।",
}
os.makedirs(OUT, exist_ok=True)
rows = []
for vk, vpath in VOICES.items():
    for tk, text in TEXTS.items():
        out = f"{OUT}/{vk}_{tk}.wav"
        t0 = time.time()
        p = subprocess.run(
            ["uv", "run", "pocket_tts", "generate", "--config", CFG, "--voice", vpath,
             "--text", text, "--output-path", out, "--device", "cpu", "--quiet"],
            cwd=f"{R}/repo", capture_output=True, text=True,
        )
        # The CLI aborts at exit (SIGABRT, "terminate called without an active
        # exception") AFTER writing a complete wav, so returncode is not a usable
        # success signal here. Judge by the artifact: a real generation is >0.5s.
        ok = False
        if os.path.exists(out):
            try:
                import soundfile as sf
                ok = sf.info(out).duration > 0.5
            except Exception:
                ok = False
        rows.append((f"{vk}_{tk}", ok, time.time() - t0))
        print(f"{'OK ' if ok else 'FAIL'} {vk}_{tk} ({time.time()-t0:.1f}s)", flush=True)
        if not ok:
            print(p.stderr[-600:], flush=True)
print("\n=== summary ===")
print(f"{sum(1 for _,ok,_ in rows if ok)}/{len(rows)} generated -> {OUT}")

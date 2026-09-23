"""Voice /workspace/v5_synth/jobs.jsonl with UNTOUCHED Kyutai 6L English.

Kyutai speaks only the English part of every job (it cannot speak Nepali: CER
0.88-0.90 on the 2x2 Nepali cells); the voice prompt is the real clip the job is
built around. Output is the raw English clip; assemble_v5.py joins it to the
real Nepali audio after QC. Run with repo/.venv, one process per shard:

    cd repo && SHARD=0 NSHARDS=4 .venv/bin/python3 ../scripts/gen_synth_v5.py

Resumable: an existing wav is skipped (same weights, same seed -> same audio).
"""
import collections, json, os, time
import scipy.io.wavfile as wav
import torch
from pocket_tts.models.tts_model import TTSModel

# Job ids are positions in jobs.jsonl, so re-running plan_v5.py re-maps them to new
# text: point OUT at a fresh dir for any run made against an older plan.
OUT = os.environ.get("OUT", "/workspace/v5_synth/audio/raw")
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
os.makedirs(OUT, exist_ok=True)

jobs = [json.loads(l) for l in open("/workspace/v5_synth/jobs.jsonl")]
jobs = [j for i, j in enumerate(jobs) if i % NSHARDS == SHARD
        and not os.path.exists(f"{OUT}/{j['id']}.wav")]
by_prompt = collections.defaultdict(list)
for j in jobs:
    by_prompt[j["real"]["path"]].append(j)

m = TTSModel.load_model(config="pocket_tts/config/english.yaml")
m.to("cuda")
SR = m.config.mimi.sample_rate
print(f"shard {SHARD}/{NSHARDS}: {len(jobs):,} jobs over {len(by_prompt):,} prompts", flush=True)

t0, done, fail = time.time(), 0, collections.Counter()
for prompt, js in by_prompt.items():
    try:
        st = m.get_state_for_audio_prompt(prompt, truncate=True)
    except Exception as e:
        fail[f"prompt:{type(e).__name__}"] += len(js)
        continue
    for j in js:
        torch.manual_seed(int(j["id"][2:]) * 7 + {"pl": 1, "co": 2, "in": 3}[j["id"][:2]])
        try:
            x = m.generate_audio(st, j["text"], copy_state=True).detach().cpu().numpy().squeeze()
        except Exception as e:
            fail[type(e).__name__] += 1
            continue
        if len(x) / SR < 0.3:
            fail["degenerate"] += 1
            continue
        wav.write(f"{OUT}/{j['id']}.wav", SR, x)
        done += 1
        if done % 500 == 0:
            print(f"  {done:,}/{len(jobs):,}  {time.time()-t0:.0f}s", flush=True)
print(f"shard {SHARD}: {done:,} ok, failures {dict(fail)} in {time.time()-t0:.0f}s", flush=True)

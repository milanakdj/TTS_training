"""Build a Parler finetune set from the PRE-VC Gemini Nepali audio.

Why this source: it is the only expressive Nepali here that is not Darjeeling-
accented. indicvoices-r Nepali is 100% West Bengal and Rasa was collected by the
same group -- its Srijana voice was rejected by ear as "a Hindi person speaking
Nepali". The user confirmed these Gemini voices sound native.

Why re-align instead of using the shipped timings: `turns` was diarized on the
VOICE-CONVERTED audio, which drifts from the source by +3.4 s on average and up
to 39 s, and only 49 of 182 files have turns/transcript_turns lining up 1:1.
So we align the true transcript against the source audio ourselves.

MMS_FA + uroman is used because the text is code-mixed Devanagari + Latin
("quadratic equation solve गर्ने"); a Devanagari-only CTC vocab cannot align it.

Captions come from `voice_direction`, which is already Parler-shaped and far
richer than Rasa's six labels ("Thoughtful but hesitant, a slight questioning
tone on 'shouldn't I subtract'").
"""
import argparse, io, json, os, re, sys
import numpy as np, soundfile as sf, torch, torchaudio
import pyarrow as pa, pyarrow.parquet as pq
import uroman as ur

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="/workspace/proc_data_new/gemini_vc/manifest_final_full_v2.jsonl")
ap.add_argument("--out", default="/workspace/milan_nepali_parler_ft/data_gem")
ap.add_argument("--min-s", type=float, default=2.0)
ap.add_argument("--max-s", type=float, default=20.0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

SRC, DST = "/projects/data/ttsteam/proc_data_new", "/workspace/proc_data_new"
# LaTeX is read aloud as words we cannot recover ("$3 \\times 2$" -> "three times two"),
# so those turns would train the model on text that does not match the audio.
LATEX = re.compile(r"\$|\\times|\\frac|\\!|\^|_\{|\\left|\\right")
PUNCT = re.compile(r"[।॥,.;:!?\"'()\[\]{}–—*<>|]")

bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(args.device).eval()
tokenizer = bundle.get_tokenizer(); aligner = bundle.get_aligner()
U = ur.Uroman()
SR_ALIGN = bundle.sample_rate

rows = []
for line in open(args.manifest):
    try: r = json.loads(line)
    except Exception: continue
    if str(r.get("language", "")).lower() in ("ne", "nepali"): rows.append(r)
if args.limit: rows = rows[:args.limit]
print(f"Nepali files: {len(rows)}", flush=True)

QUALITY = " The recording is very high quality, with the speaker's voice sounding clear and very close up."
schema = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("text", pa.string()), ("description", pa.string()), ("speaker", pa.string()),
    ("role", pa.string()), ("gender", pa.string()),
    ("duration", pa.float32()), ("id", pa.string()),
])

out_rows, n_files, n_drop_latex, n_fail = [], 0, 0, 0
for r in rows:
    src = (r.get("source_audio_filepath") or "").replace(SRC, DST)
    tt = r.get("transcript_turns") or []
    if not os.path.exists(src) or not tt: continue
    try:
        y, sr = sf.read(src, dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        # build the word stream, remembering which turn each word came from
        words, roms, owner = [], [], []
        for ti, t in enumerate(tt):
            txt = (t.get("text") or "").strip()
            if not txt or LATEX.search(txt):
                if txt: n_drop_latex += 1
                continue
            for w in txt.split():
                core = PUNCT.sub("", w)
                if not core: continue
                rom = re.sub(r"[^a-z']", "", U.romanize_string(core).lower())
                if not rom: continue
                words.append(core); roms.append(rom); owner.append(ti)
        if len(roms) < 5: continue
        wav = torch.from_numpy(y).unsqueeze(0)
        if sr != SR_ALIGN:
            wav = torchaudio.functional.resample(wav, sr, SR_ALIGN)
        with torch.inference_mode():
            emission, _ = model(wav.to(args.device))
        spans = aligner(emission[0], tokenizer(roms))
        ratio = wav.shape[1] / emission.shape[1] / SR_ALIGN
        times = [(s[0].start * ratio, s[-1].end * ratio) for s in spans]

        # group consecutive words by turn, then split long turns at the widest gaps
        by_turn = {}
        for i, ti in enumerate(owner): by_turn.setdefault(ti, []).append(i)
        for ti, idxs in by_turn.items():
            t = tt[ti]
            vd = (t.get("voice_direction") or "").strip()
            if not vd: continue
            segs = [idxs]
            changed = True
            while changed:
                changed = False; nxt = []
                for grp in segs:
                    d = times[grp[-1]][1] - times[grp[0]][0]
                    if d <= args.max_s or len(grp) < 4:
                        nxt.append(grp); continue
                    gaps = [(times[grp[k+1]][0] - times[grp[k]][1], k) for k in range(len(grp)-1)]
                    _, k = max(gaps)
                    nxt.append(grp[:k+1]); nxt.append(grp[k+1:]); changed = True
                segs = nxt
            for grp in segs:
                a, b = times[grp[0]][0], times[grp[-1]][1]
                d = b - a
                if not (args.min_s <= d <= args.max_s): continue
                seg = y[int(a*sr):int(b*sr)]
                if len(seg) < int(args.min_s*sr): continue
                role = t.get("role")
                voice = r.get("teacher_gemini_voice") if role == "teacher" else r.get("student_gemini_voice")
                gen = r.get("teacher_gemini_gender") if role == "teacher" else r.get("student_gemini_gender")
                if not voice: continue
                buf = io.BytesIO(); sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
                desc = f"{voice} speaks in Nepali. {vd.rstrip('.')}." + QUALITY
                out_rows.append({
                    "audio": {"bytes": buf.getvalue(), "path": f"{r.get('sample_id','x')}_{ti}.wav"},
                    "text": " ".join(words[i] for i in grp),
                    "description": desc, "speaker": voice, "role": role or "",
                    "gender": gen or "", "duration": float(d),
                    "id": f"{r.get('sample_id','x')}_{ti}_{grp[0]}",
                })
        n_files += 1
        if n_files % 20 == 0:
            print(f"  {n_files}/{len(rows)} files, {len(out_rows)} segments", flush=True)
    except Exception as e:
        n_fail += 1
        if n_fail <= 2:
            import traceback; traceback.print_exc(); sys.stdout.flush()

print(f"\nfiles ok {n_files}, failed {n_fail}, latex turns dropped {n_drop_latex}")
tot = sum(r["duration"] for r in out_rows)
print(f"segments {len(out_rows)}, {tot/3600:.2f} h")
import collections
print("by speaker:", collections.Counter(r["speaker"] for r in out_rows).most_common())

import random
rng = random.Random(20260905); rng.shuffle(out_rows)
nval = max(50, len(out_rows)//50)
for name, part in (("validation", out_rows[:nval]), ("train", out_rows[nval:])):
    d = os.path.join(args.out, name); os.makedirs(d, exist_ok=True)
    for i in range(0, len(part), 1000):
        pq.write_table(pa.Table.from_pylist(part[i:i+1000], schema=schema),
                       os.path.join(d, f"part-{i//1000:04d}.parquet"))
    print(f"{name}: {len(part)} rows, {sum(x['duration'] for x in part)/3600:.2f} h")
print("->", args.out)

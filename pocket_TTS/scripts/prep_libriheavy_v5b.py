"""More real English for the 15%-English v5 arms (v5m, v5m_lwf): a second, disjoint
LibriHeavy draw, sharded so several Whisper processes can share the GPU.

Same cutting/QC as prep_libriheavy_v5.py (imported logic copied verbatim below the
selection), but: no stride, up to PER_SPK recordings per speaker, and every
recording round 1 picked is excluded (round 1's selection is replayed exactly:
same manifest, same seed, same stride). Output per shard:
/workspace/v5_synth/libriheavy_b/train_v5_libriheavy_b.<shard>.jsonl

    SHARD=0 NSHARDS=4 N_REC=10400 python3 prep_libriheavy_v5b.py
"""

import collections, difflib, json, os, random, re, string, unicodedata
import jiwer
import soundfile as sf
from faster_whisper import WhisperModel

SRC = "/workspace/proc_data_new/libriheavy/merged_tts/large/manifest.jsonl"
OUT = "/workspace/v5_synth/libriheavy_b"
N_REC = int(os.environ.get("N_REC", "10400"))
PER_SPK = int(os.environ.get("PER_SPK", "4"))
SHARD, NSHARDS = int(os.environ["SHARD"]), int(os.environ["NSHARDS"])
MAX_CER = 0.08
_T = str.maketrans("", "", string.punctuation + "‘’“”–—…")
os.makedirs(f"{OUT}/clips", exist_ok=True)


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t).translate(_T).lower()).strip()


# Replay round 1's pick (prep_libriheavy_v5.py, N_CLIPS=2600) to exclude it.
rng = random.Random(9)
by_spk = collections.defaultdict(list)
for i, line in enumerate(open(SRC)):
    if i % 7:
        continue
    d = json.loads(line)
    if 15 <= d["duration"] <= 35:
        by_spk[d["speaker"]].append(d)
speakers = sorted(by_spk)
rng.shuffle(speakers)
used = {rng.choice(by_spk[s])["audio_filepath"] for s in speakers[:2600]}

rng = random.Random(10)
by_spk = collections.defaultdict(list)
for line in open(SRC):
    d = json.loads(line)
    if 15 <= d["duration"] <= 35 and d["audio_filepath"] not in used:
        by_spk[d["speaker"]].append(d)
pool = []
for s in sorted(by_spk):
    recs = by_spk[s]
    rng.shuffle(recs)
    pool += recs[:PER_SPK]
rng.shuffle(pool)
picked = pool[:N_REC][SHARD::NSHARDS]
print(f"{len(by_spk):,} speakers, pool {len(pool):,}, excluded {len(used):,} round-1 "
      f"recordings; shard {SHARD}/{NSHARDS} takes {len(picked):,}", flush=True)
rng = random.Random(100 + SHARD)   # cut-length draws

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
rows, kept, rejected = [], 0, 0
for n, d in enumerate(picked, 1):
    path = d["audio_filepath"].replace("/projects/data/ttsteam/", "/workspace/")
    segs, _ = model.transcribe(path, language="en", beam_size=5, word_timestamps=True,
                               condition_on_previous_text=False)
    words = [w for s in segs for w in s.words]
    if not words or jiwer.cer(norm(d["text"]), norm(" ".join(w.word for w in words))) > MAX_CER:
        rejected += 1
        continue
    kept += 1
    audio, sr = sf.read(path, dtype="float32")
    # Text comes from the BOOK (cased, punctuated, like what users type); Whisper
    # only supplies timings. Match the two word sequences and cut pieces only inside
    # runs where they agree word for word.
    ref = d["text"].split()
    sm = difflib.SequenceMatcher(None, [norm(t) for t in ref], [norm(w.word) for w in words],
                                 autojunk=False)
    for blk in sm.get_matching_blocks():
        i, stop = blk.a, blk.a + blk.size
        while stop - i >= 3:
            target = rng.uniform(3.0, 10.0)
            j = i + 1
            while j < stop:
                t0 = words[blk.b + (i - blk.a)].start
                if words[blk.b + (j - blk.a)].end - t0 > 11.5:
                    break
                j += 1
                if words[blk.b + (j - 1 - blk.a)].end - t0 >= target and ref[j - 1][-1:] in ".!?;:,":
                    break
            span_w = words[blk.b + (i - blk.a): blk.b + (j - blk.a)]
            dur = span_w[-1].end - span_w[0].start if span_w else 0
            if 2.0 <= dur <= 11.5 and j - i >= 3:
                a = max(0.0, span_w[0].start - 0.1)
                b = min(len(audio) / sr, span_w[-1].end + 0.15)
                out = f"{OUT}/clips/lh{SHARD}_{len(rows):06d}.wav"
                sf.write(out, audio[int(a * sr):int(b * sr)], sr)
                ws = [{"word": r, "start": round(w.start - a, 3), "end": round(w.end - a, 3)}
                      for r, w in zip(ref[i:j], span_w)]
                rows.append({"path": out, "duration": round(b - a, 3),
                             "transcript": " ".join(ref[i:j]),
                             "speaker": f"libriheavy/{d['speaker']}", "source": "v5_libriheavy",
                             "language": "en", "style": None, "words": ws})
            i = j
    if n % 200 == 0:
        print(f"  {n:,}/{len(picked):,} recordings, {kept} kept, {len(rows):,} clips", flush=True)

with open(f"{OUT}/train_v5_libriheavy_b.{SHARD}.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
h = sum(r["duration"] for r in rows) / 3600
print(f"LIBRIHEAVY_B SHARD DONE: {kept}/{len(picked)} recordings passed (rejected {rejected}), "
      f"{len(rows):,} clips, {h:.1f} h -> {OUT}/train_v5_libriheavy_b.{SHARD}.jsonl")

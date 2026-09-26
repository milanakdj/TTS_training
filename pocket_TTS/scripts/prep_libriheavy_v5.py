"""Real-English arm (L) for the v5 probe: LibriHeavy cut into short aligned clips.

Arm L swaps v5's Kyutai-synthetic "plain" clips for real read English, to test
whether clean real speech preserves Kyutai's English as well as its own output
does (arm S). LibriHeavy merged_tts clips are 15-35 s with a book transcript and no
word times, and they are 16 kHz (band-limited vs Kyutai's 24 kHz; a known
confound). One clip per speaker for voice diversity. Whisper large-v3-turbo gives
word times only; a recording is kept if its transcript agrees with the book text
(CER <= 0.08). Pieces of ~3-10 s (<= 11.5 s) are cut at the book's punctuation,
inside runs where book and Whisper agree word for word, and keep the BOOK text
(cased, punctuated). Run with synthetic_pipeline/asr_qc_service/.venv:

    N_CLIPS=2600 python3 prep_libriheavy_v5.py     # ~3.6 pieces per recording
"""
import collections, difflib, json, os, random, re, string, unicodedata
import jiwer
import soundfile as sf
from faster_whisper import WhisperModel

SRC = "/workspace/proc_data_new/libriheavy/merged_tts/large/manifest.jsonl"
OUT = "/workspace/v5_synth/libriheavy"
N_CLIPS = int(os.environ.get("N_CLIPS", "1800"))
MAX_CER = 0.08
_T = str.maketrans("", "", string.punctuation + "‘’“”–—…")
os.makedirs(f"{OUT}/clips", exist_ok=True)


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t).translate(_T).lower()).strip()


rng = random.Random(9)
by_spk = collections.defaultdict(list)
for i, line in enumerate(open(SRC)):
    if i % 7:            # stride through the 1.5 M rows; plenty of speakers survive
        continue
    d = json.loads(line)
    if 15 <= d["duration"] <= 35:
        by_spk[d["speaker"]].append(d)
speakers = sorted(by_spk)
rng.shuffle(speakers)
picked = [rng.choice(by_spk[s]) for s in speakers[:N_CLIPS]]
print(f"{len(speakers):,} speakers seen; taking one clip each from {len(picked):,}", flush=True)

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
                out = f"{OUT}/clips/lh{len(rows):06d}.wav"
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

with open(f"{OUT}/train_v5_libriheavy.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
h = sum(r["duration"] for r in rows) / 3600
print(f"LIBRIHEAVY DONE: {kept}/{len(picked)} recordings passed (rejected {rejected}), "
      f"{len(rows):,} clips, {h:.1f} h -> {OUT}/train_v5_libriheavy.jsonl")

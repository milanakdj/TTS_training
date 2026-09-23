"""QC + word-align the Kyutai English clips from gen_synth_v5.py.

Training cuts every clip at an aligned word boundary and reads the target text
FROM `words` (dataloader/loader.py), so each English clip needs word times, and
a clip whose audio does not say its text would teach the wrong mapping.
faster-whisper large-v3-turbo transcribes with word timestamps; a clip passes
when the transcript has the same word count as the Haiku text and CER <= 0.10,
and then the Haiku words (original spelling/punctuation) take Whisper's times.
Run with synthetic_pipeline/asr_qc_service/.venv:

    RAW=/workspace/v5_synth/audio/raw python3 qc_align_v5.py
"""
import json, os, re, string, unicodedata
import jiwer
from faster_whisper import WhisperModel

RAW = os.environ.get("RAW", "/workspace/v5_synth/audio/raw")
OUT = os.environ.get("QC_OUT", os.path.join(os.path.dirname(RAW), "qc.jsonl"))
MAX_CER = 0.10
_T = str.maketrans("", "", string.punctuation + "‘’“”–—…")


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t).translate(_T).lower()).strip()


jobs = {j["id"]: j for j in map(json.loads, open("/workspace/v5_synth/jobs.jsonl"))}
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT)}
todo = sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".wav") and f[:-4] not in done)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
print(f"{len(todo):,} clips to check ({len(done):,} already done)", flush=True)

npass = 0
with open(OUT, "a") as out:
    for n, jid in enumerate(todo, 1):
        ref = jobs[jid]["text"]
        segs, _ = model.transcribe(f"{RAW}/{jid}.wav", language="en", beam_size=5,
                                   word_timestamps=True, condition_on_previous_text=False)
        hw = [w for s in segs for w in s.words]
        hyp = " ".join(w.word.strip() for w in hw)
        rn, hn = norm(ref), norm(hyp)
        cer = jiwer.cer(rn, hn) if rn else 1.0
        refw = ref.split()
        ok = cer <= MAX_CER and len(refw) == len(hn.split()) == len(hw)
        row = {"id": jid, "ok": ok, "cer": round(cer, 4), "hyp": hyp}
        if ok:
            row["words"] = [{"word": r, "start": round(w.start, 3), "end": round(w.end, 3)}
                            for r, w in zip(refw, hw)]
            npass += 1
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if n % 1000 == 0:
            out.flush()
            print(f"  {n:,}/{len(todo):,}  pass {npass/n:.1%}", flush=True)
print(f"QC DONE: {npass:,}/{len(todo):,} passed ({npass/max(len(todo),1):.1%}) -> {OUT}")

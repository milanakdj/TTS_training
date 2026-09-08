#!/usr/bin/env python3
"""
screen_references.py -- automated screening gate for voice-conversion reference clips.

WHY THIS EXISTS
---------------
The v1 reference set (emotions/dataset_gemini_*) was audited (emotions/REFERENCE_AUDIT.md)
and found unusable: Gemini TTS, asked for "angry" and "happy", shouted an octave above the
speaker's natural register (median F0 407/420 Hz vs 206 Hz neutral, <2.3% of spectral energy
below 250 Hz). That broke speaker identity (CAM++ cross-emotion cosine 0.529 against a
same-speaker threshold of ~0.562) and collapsed angry and happy into each other (MFCC
centroid distance 14.0, vs 57.7 angry<->neutral).

Critically, a pitch-standard-deviation check would have MISSED all of it -- F0 sd was a flat
3.87-4.44 semitones across all four folders. MEDIAN F0 and SPEAKER EMBEDDING are the signals
that caught it. This gate is built on those, not on pitch spread.

Because these references drive ~50,000 conversions, one bad clip poisons thousands of rows,
so every clip is gated on five independent checks and every clip -- pass or fail -- is written
to the manifest so a downstream orchestrator can see the rejects.

THE GATE (per clip)
-------------------
1. median F0 in [162, 257] Hz  (+/-4 semitones around 205 Hz).
   Estimated with THREE independent estimators, none of which has a restrictive range prior:
   pyin (fmin=70, fmax=900), raw autocorrelation, and real-cepstrum peak picking. Clamping
   fmax low silently fabricates a passing value out of a subharmonic -- that exact mistake was
   made during the v1 investigation, so the range here is deliberately wide and the estimators
   are reconciled explicitly. Fraction of spectral energy below 250 Hz on voiced frames is
   recorded as a fourth, independent register signal (in-register 10-25%, shouted <3%).
2. CAM++ cosine >= 0.61 to the v2 neutral centroid, using the SAME encoder and feature
   pipeline Seed-VC actually conditions on (vc_service/server.py: energy-VAD trim, 25 s cap,
   resample to 16 kHz, kaldi fbank 80 mel dither=0, per-utterance mean subtraction).
   The centroid is built from v2 neutral clips that pass check 1, then sanity-checked against
   the 6 v1 dataset_gemini_neutral clips (audited as on-anchor keepers). If those controls do
   not score high, the new neutral set is itself off-anchor -> hard blocker, no manifest.
3. >= 60% speech after energy VAD. v1 sad clips were 44-66% speech because the prompt asked
   for sniffling.
4. Not truncated mid-decay. One v1 clip ended with its last 40 ms only -12.3 dB below speech
   level and zero trailing silence.
5. Duration in [3, 15] s.

OUTPUT
------
manifests/reference_pool_emotion_v2.parquet with columns exactly:
  emotion, path, f0_median, campplus_cos_to_neutral_centroid, speech_fraction,
  duration_s, passed, reject_reason

Read-only with respect to the pipeline tree: nothing under emotions/v2/ or
emotions/dataset_gemini_* is written or modified, and no service is touched.

Run:  vc_service/.venv/bin/python emotions/screen_references.py [--analysis]
      (--analysis additionally prints the v1-vs-v2 F0 table, the MFCC-centroid /
       LTAS-correlation separation matrices for the high-arousal emotions, and the
       sad-clip non-speech-segment comparison.)
"""
import argparse
import os
import sys

import numpy as np

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(HERE)
V2_DIR = os.path.join(HERE, os.environ.get("REF_SET_DIR", "v2"))
V1_DIRS = {
    "neutral": os.path.join(HERE, "dataset_gemini_neutral"),
    "sad": os.path.join(HERE, "dataset_gemini_sad"),
    "angry": os.path.join(HERE, "dataset_gemini_angry"),
    "happy": os.path.join(HERE, "dataset_gemini_happy"),
}
V1_NEUTRAL_CONTROL = V1_DIRS["neutral"]
SEED_VC_DIR = os.path.join(PIPELINE, "vc_service", "seed-vc")
CAMPPLUS_CKPT = os.path.join(SEED_VC_DIR, "campplus_cn_common.bin")
MANIFEST_DIR = os.path.join(PIPELINE, "manifests")
OUT_PARQUET = os.path.join(MANIFEST_DIR, "reference_pool_emotion_v2.parquet")

EMOTIONS = ["angry", "happy", "excited", "sad", "neutral"]

# --------------------------------------------------------------------------------------
# gate thresholds
# --------------------------------------------------------------------------------------
F0_TARGET_HZ = 205.0
F0_MIN_HZ = 162.0          # -4 semitones around 205
F0_MAX_HZ = 257.0          # +4 semitones around 205
CAMPPLUS_MIN_COS = 0.61
SPEECH_FRACTION_MIN = 0.60
DURATION_MIN_S = 3.0
DURATION_MAX_S = 15.0

# truncation: a clip is truncated mid-decay if its final 40 ms is still near speech level
# AND there is essentially no trailing silence.
TRUNC_TAIL_MS = 40.0
TRUNC_TAIL_DB_REL = -20.0   # louder than this (relative to speech RMS) is suspicious
TRUNC_MIN_TRAIL_SILENCE_MS = 30.0

# control sanity check: the 6 audited-good v1 neutral clips must clear this against the
# v2-neutral-derived centroid, otherwise the v2 neutral set is itself off-anchor.
CONTROL_MIN_MEAN_COS = 0.61
CONTROL_MIN_INDIV_COS = 0.55

# --------------------------------------------------------------------------------------
# analysis / measurement parameters (kept identical to REFERENCE_AUDIT.md so v1 numbers
# quoted in that report stay directly comparable)
# --------------------------------------------------------------------------------------
PYIN_FMIN, PYIN_FMAX = 70.0, 900.0
FRAME_LENGTH, HOP_LENGTH = 2048, 256
SPEECH_THRESH_DB = 35.0     # frames within 35 dB of the peak frame count as speech
VAD_TOP_DB = 30             # librosa.effects.split, same value vc_service/server.py uses
LOWBAND_HZ = 250.0
N_MFCC = 14                 # c1..c13 are used, c0 (energy) is dropped
VC_SR = 22050               # preprocess_params.sr in the Seed-VC preset server.py loads
REF_MAX_S = 25              # server.py caps the reference at sr*25 samples


# ======================================================================================
# audio helpers
# ======================================================================================
def load_audio(path, sr=None):
    import librosa
    y, got_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float64), got_sr


def frame_db(y, sr):
    """Per-frame RMS in dB plus a boolean 'this frame is speech-level' mask."""
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    loud = db > (db.max() - SPEECH_THRESH_DB)
    return db, loud


# ======================================================================================
# check 1 -- F0, three independent estimators, no restrictive range prior
# ======================================================================================
def f0_pyin(y, sr, loud_mask):
    """librosa.pyin over a deliberately WIDE range (70-900 Hz).

    A narrow fmax would let a shouted clip masquerade as in-register by locking onto a
    subharmonic, which is exactly how the v1 investigation initially fooled itself.
    """
    import librosa
    f0, voiced, _ = librosa.pyin(
        y, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=sr,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    n = min(len(f0), len(loud_mask))
    sel = voiced[:n] & loud_mask[:n] & np.isfinite(f0[:n])
    if sel.sum() < 5:
        return np.nan, 0
    return float(np.nanmedian(f0[:n][sel])), int(sel.sum())


def f0_autocorr(y, sr, fmin=60.0, fmax=1000.0, frame_ms=40.0, hop_ms=10.0):
    """Raw normalised autocorrelation peak picking on 40 ms frames.

    Fully independent of pyin: no probabilistic transition model, no viterbi, and the
    search range spans a whole female-to-shout gamut so a real 400 Hz clip cannot be
    reported as 200 Hz.
    """
    flen = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    if len(y) < flen:
        return np.nan, 0
    lag_min = max(2, int(np.floor(sr / fmax)))
    lag_max = min(flen - 1, int(np.ceil(sr / fmin)))
    if lag_max <= lag_min:
        return np.nan, 0

    frames = []
    for start in range(0, len(y) - flen + 1, hop):
        frames.append(y[start:start + flen])
    frames = np.asarray(frames)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    db = 20.0 * np.log10(rms)
    loud = db > (db.max() - SPEECH_THRESH_DB)

    ests = []
    win = np.hanning(flen)
    for fr, ok in zip(frames, loud):
        if not ok:
            continue
        x = (fr - fr.mean()) * win
        ac = np.correlate(x, x, mode="full")[flen - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lag_min:lag_max + 1]
        if seg.size < 3:
            continue
        # first prominent local maximum, not merely the global one: the global max can sit
        # at an integer multiple of the true period (an octave-down error).
        peaks = [i for i in range(1, seg.size - 1)
                 if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.35]
        if not peaks:
            continue
        best = max(peaks, key=lambda i: seg[i])
        thresh = 0.85 * seg[best]
        chosen = next((i for i in peaks if seg[i] >= thresh), best)
        ests.append(sr / float(lag_min + chosen))
    if len(ests) < 5:
        return np.nan, 0
    return float(np.median(ests)), len(ests)


def f0_cepstrum(y, sr, fmin=60.0, fmax=1000.0, frame_ms=40.0, hop_ms=10.0):
    """Real-cepstrum peak picking -- a third, spectral-domain-independent estimator."""
    flen = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    if len(y) < flen:
        return np.nan, 0
    q_min = max(2, int(np.floor(sr / fmax)))
    q_max = min(flen // 2 - 1, int(np.ceil(sr / fmin)))
    if q_max <= q_min:
        return np.nan, 0

    win = np.hanning(flen)
    ests = []
    starts = range(0, len(y) - flen + 1, hop)
    frames = np.asarray([y[s:s + flen] for s in starts])
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    db = 20.0 * np.log10(rms)
    loud = db > (db.max() - SPEECH_THRESH_DB)

    for fr, ok in zip(frames, loud):
        if not ok:
            continue
        spec = np.abs(np.fft.rfft((fr - fr.mean()) * win, n=2 * flen))
        logspec = np.log(np.maximum(spec, 1e-12))
        ceps = np.fft.irfft(logspec)
        seg = ceps[q_min:q_max + 1]
        if seg.size < 3:
            continue
        k = int(np.argmax(seg))
        # require the peak to stand out from the local cepstral floor
        if seg[k] <= np.median(seg) + 2.0 * (np.std(seg) + 1e-12) * 0.5:
            continue
        ests.append(sr / float(q_min + k))
    if len(ests) < 5:
        return np.nan, 0
    return float(np.median(ests)), len(ests)


def semitones(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return np.nan
    return 12.0 * np.log2(a / b)


def reconcile_f0(pyin_hz, ac_hz, cep_hz):
    """Reconcile three estimators into one median F0 plus a disagreement note.

    Rule: if any two estimators agree within 1.5 semitones, trust the mean of the tightest
    agreeing pair (this rejects a single estimator's octave error). If none agree, fall back
    to the median of whatever is finite and flag it loudly -- an unreconciled clip is never
    silently passed, it is reported so a human can look.
    """
    cands = {"pyin": pyin_hz, "autocorr": ac_hz, "cepstrum": cep_hz}
    finite = {k: v for k, v in cands.items() if np.isfinite(v) and v > 0}
    if not finite:
        return np.nan, "no estimator produced an F0"
    if len(finite) == 1:
        k, v = next(iter(finite.items()))
        return v, "only %s produced an F0" % k

    keys = list(finite)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = abs(semitones(finite[keys[i]], finite[keys[j]]))
            pairs.append((d, keys[i], keys[j]))
    pairs.sort()
    spread = pairs[-1][0]
    d, k1, k2 = pairs[0]
    if d <= 1.5:
        val = 0.5 * (finite[k1] + finite[k2])
        note = "" if spread <= 1.5 else (
            "estimators disagree up to %.2f st (%s); used %s+%s consensus %.1f Hz"
            % (spread, ", ".join("%s=%.1f" % (k, v) for k, v in finite.items()), k1, k2, val)
        )
        return val, note
    val = float(np.median(list(finite.values())))
    note = ("NO ESTIMATOR CONSENSUS (min pair gap %.2f st): %s -- using median %.1f Hz"
            % (d, ", ".join("%s=%.1f" % (k, v) for k, v in finite.items()), val))
    return val, note


def lowband_energy_fraction(y, sr, loud_mask):
    """Fraction of spectral energy below 250 Hz on speech-level frames.

    Independent of every pitch estimator, and the signal that most cleanly separated v1's
    shouted clips (<3%) from in-register speech (10-25%).
    """
    import librosa
    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    n = min(S.shape[1], len(loud_mask))
    S = S[:, :n][:, loud_mask[:n]]
    if S.shape[1] == 0:
        return np.nan
    total = S.sum()
    if total <= 0:
        return np.nan
    return float(S[freqs < LOWBAND_HZ, :].sum() / total)


# ======================================================================================
# check 3 -- VAD speech fraction;  check 4 -- truncation
# ======================================================================================
def vad_stats(y, sr):
    """Energy-VAD speech fraction, interior non-speech segment count, trailing silence."""
    import librosa
    if len(y) == 0:
        return 0.0, 0, 0.0
    iv = librosa.effects.split(y, top_db=VAD_TOP_DB,
                               frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    if len(iv) == 0:
        return 0.0, 0, len(y) / sr * 1000.0
    speech = int(sum(int(b) - int(a) for a, b in iv))
    frac = speech / len(y)
    interior_gaps = max(0, len(iv) - 1)
    trailing_ms = (len(y) - int(iv[-1][1])) / sr * 1000.0
    return float(frac), int(interior_gaps), float(trailing_ms)


def truncation_stats(y, sr, loud_mask):
    """Is the clip cut off mid-decay?  Returns (tail_db_rel, trailing_silence_ms, is_trunc)."""
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    n = min(len(rms), len(loud_mask))
    speech_rms = float(np.sqrt(np.mean(rms[:n][loud_mask[:n]] ** 2))) if loud_mask[:n].any() else 0.0
    ntail = max(1, int(round(sr * TRUNC_TAIL_MS / 1000.0)))
    tail_rms = float(np.sqrt(np.mean(y[-ntail:] ** 2))) if len(y) >= ntail else 0.0
    if speech_rms <= 0:
        return np.nan, 0.0, False
    tail_db_rel = 20.0 * np.log10(max(tail_rms, 1e-10) / speech_rms)
    _, _, trailing_ms = vad_stats(y, sr)
    is_trunc = (tail_db_rel > TRUNC_TAIL_DB_REL) and (trailing_ms < TRUNC_MIN_TRAIL_SILENCE_MS)
    return float(tail_db_rel), float(trailing_ms), bool(is_trunc)


# ======================================================================================
# check 2 -- CAM++ speaker embedding, exactly as vc_service/server.py conditions Seed-VC
# ======================================================================================
class CampPlusEncoder:
    """CAM++ wrapper mirroring vc_service/server.py's reference path byte for byte.

    server.py:281-339 -- librosa.load at the preset sr (22050), energy-VAD trim of the
    reference, cap at 25 s, resample to 16 kHz, torchaudio kaldi fbank (80 mel, dither=0,
    sample_frequency=16000), subtract the per-utterance feature mean, then CAMPPlus forward.
    Using the same encoder matters: it is the one Seed-VC actually conditions on, so its
    notion of "same speaker" is the one that decides whether a reference preserves identity.
    """

    def __init__(self, device="cpu"):
        import torch
        sys.path.insert(0, SEED_VC_DIR)
        from modules.campplus.DTDNN import CAMPPlus
        self.torch = torch
        self.device = torch.device(device)
        m = CAMPPlus(feat_dim=80, embedding_size=192)
        m.load_state_dict(torch.load(CAMPPLUS_CKPT, map_location="cpu"))
        m.eval().to(self.device)
        self.model = m

    def _vad_trim(self, y, sr):
        """server.py vad_trim_reference(): top_db=30, 30 ms pad, 1.0 s fallback floor."""
        import librosa
        if len(y) == 0:
            return y
        iv = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
        if len(iv) == 0:
            return y
        pad = int(round(sr * 0.03))
        merged = []
        for a, b in iv:
            a, b = max(0, int(a) - pad), min(len(y), int(b) + pad)
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        trimmed = np.concatenate([y[a:b] for a, b in merged])
        if len(trimmed) / sr < 1.0:
            return y
        return trimmed

    def embed(self, path, vad_trim=True):
        import torch
        import torchaudio
        y, sr = load_audio(path, sr=VC_SR)
        if vad_trim:
            y = self._vad_trim(y, sr)
        y = y[: sr * REF_MAX_S]
        wav = torch.tensor(y).unsqueeze(0).float().to(self.device)
        wav16 = torchaudio.functional.resample(wav, sr, 16000)
        feat = torchaudio.compliance.kaldi.fbank(
            wav16, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = self.model(feat.unsqueeze(0))
        return emb.squeeze(0).cpu().numpy().astype(np.float64)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


# ======================================================================================
# per-clip measurement
# ======================================================================================
def measure(path):
    y, sr = load_audio(path, sr=None)
    dur = len(y) / sr if sr else 0.0
    _, loud = frame_db(y, sr)

    p, np_ = f0_pyin(y, sr, loud)
    a, na = f0_autocorr(y, sr)
    c, nc = f0_cepstrum(y, sr)
    f0, f0_note = reconcile_f0(p, a, c)

    low = lowband_energy_fraction(y, sr, loud)
    frac, gaps, _ = vad_stats(y, sr)
    tail_db, trail_ms, trunc = truncation_stats(y, sr, loud)

    return dict(
        path=os.path.abspath(path), duration_s=float(dur), sr=int(sr),
        f0_pyin=p, f0_autocorr=a, f0_cepstrum=c, f0_median=f0, f0_note=f0_note,
        f0_frames=int(np_),
        lowband_frac=low, speech_fraction=frac, nonspeech_segments=gaps,
        tail_db_rel=tail_db, trailing_silence_ms=trail_ms, truncated=trunc,
    )


def clips_in(d, pattern=".wav"):
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(pattern) and os.path.getsize(os.path.join(d, f)) > 1000)


# ======================================================================================
# analysis for the separation questions (MFCC centroid distance, LTAS correlation)
# ======================================================================================
def mfcc_mean(path):
    """Mean MFCC c1..c13 over speech-level frames (c0 dropped: it is loudness, not timbre)."""
    import librosa
    y, sr = load_audio(path, sr=None)
    _, loud = frame_db(y, sr)
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                             n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    n = min(M.shape[1], len(loud))
    M = M[1:, :n][:, loud[:n]]
    return M.mean(axis=1) if M.shape[1] else np.full(N_MFCC - 1, np.nan)


def ltas(path, fmax=8000.0):
    """Long-term average spectrum in dB over speech-level frames, 0-8 kHz."""
    import librosa
    y, sr = load_audio(path, sr=None)
    _, loud = frame_db(y, sr)
    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    n = min(S.shape[1], len(loud))
    S = S[:, :n][:, loud[:n]]
    if S.shape[1] == 0:
        return None, None
    m = S.mean(axis=1)
    keep = freqs <= fmax
    return 10.0 * np.log10(np.maximum(m[keep], 1e-20)), freqs[keep]


def separation_report(groups, label):
    """MFCC-centroid distance + LTAS correlation matrices, comparable to REFERENCE_AUDIT.md."""
    names = [k for k in groups if groups[k]]
    if len(names) < 2:
        return
    cent, spread, lt = {}, {}, {}
    for k in names:
        vs = np.array([mfcc_mean(p) for p in groups[k]])
        cent[k] = vs.mean(axis=0)
        spread[k] = float(np.mean([np.linalg.norm(v - cent[k]) for v in vs]))
        ls = [ltas(p)[0] for p in groups[k]]
        ls = [x for x in ls if x is not None]
        lt[k] = np.mean(np.array(ls), axis=0) if ls else None

    print("\n  %s -- MFCC(c1-c13) centroid distance (Euclidean)" % label)
    print("    %-10s" % "" + "".join("%9s" % k[:8] for k in names))
    for i in names:
        print("    %-10s" % i + "".join(
            "%9.1f" % np.linalg.norm(cent[i] - cent[j]) if i != j else "%9s" % "-"
            for j in names))
    print("    within-emotion MFCC spread (mean clip->own centroid): " +
          ", ".join("%s %.1f" % (k, spread[k]) for k in names))

    print("\n  %s -- LTAS log-spectrum correlation (0-8 kHz)" % label)
    print("    %-10s" % "" + "".join("%9s" % k[:8] for k in names))
    for i in names:
        row = []
        for j in names:
            if i == j:
                row.append("%9s" % "-")
            elif lt[i] is None or lt[j] is None:
                row.append("%9s" % "n/a")
            else:
                n = min(len(lt[i]), len(lt[j]))
                row.append("%9.3f" % np.corrcoef(lt[i][:n], lt[j][:n])[0, 1])
        print("    %-10s" % i + "".join(row))


# ======================================================================================
# main
# ======================================================================================
# --------------------------------------------------------------------------------------
# manifest writing
# --------------------------------------------------------------------------------------
COLUMNS = ["emotion", "path", "f0_median", "campplus_cos_to_neutral_centroid",
           "speech_fraction", "duration_s", "passed", "reject_reason"]
ORCH_PY = os.path.join(PIPELINE, "orchestrator", ".venv", "bin", "python")


def write_parquet(rows, out_path):
    """Write the manifest as parquet.

    The venv that holds torch/CAM++ (vc_service/.venv) has no pyarrow, and the venv
    that has pyarrow (orchestrator/.venv) has no torch, so when pyarrow is missing we
    hand the finished rows to the orchestrator interpreter instead of failing after
    all the GPU work is already done.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pass
    else:
        _write_parquet_here(rows, out_path)
        return

    if not os.path.exists(ORCH_PY):
        raise SystemExit("no pyarrow in this interpreter and no %s to fall back to" % ORCH_PY)
    import json
    import subprocess
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        subprocess.run(
            [ORCH_PY, os.path.abspath(__file__), "--write-parquet-from", tmp, "--out", out_path],
            check=True)
    finally:
        os.unlink(tmp)


def _write_parquet_here(rows, out_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("emotion", pa.string()), ("path", pa.string()), ("f0_median", pa.float64()),
        ("campplus_cos_to_neutral_centroid", pa.float64()),
        ("speech_fraction", pa.float64()), ("duration_s", pa.float64()),
        ("passed", pa.bool_()), ("reject_reason", pa.string()),
    ])
    cols = {name: [r[name] for r in rows] for name in COLUMNS}
    pq.write_table(pa.table(cols, schema=schema), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", action="store_true",
                    help="also print v1-vs-v2 F0 table, separation matrices, sad VAD comparison")
    ap.add_argument("--out", default=OUT_PARQUET)
    ap.add_argument("--neutral-dir", default=None,
                    help="read the neutral clips from here instead of REF_SET_DIR/neutral; "
                         "use when re-screening a partial regeneration")
    ap.add_argument("--write-parquet-from", default=None,
                    help="internal: write the JSON rows at this path to --out and exit "
                         "(used to reach an interpreter that has pyarrow)")
    args = ap.parse_args()

    if args.write_parquet_from:
        import json
        with open(args.write_parquet_from, encoding="utf-8") as fh:
            _write_parquet_here(json.load(fh), args.out)
        return 0

    print("=" * 100)
    print("REFERENCE SCREENING GATE -- emotions/v2")
    print("=" * 100)
    print("gate: F0 median %.0f-%.0f Hz | CAM++ cos>=%.2f to neutral centroid | speech>=%.0f%% "
          "| not truncated | %.0f-%.0f s"
          % (F0_MIN_HZ, F0_MAX_HZ, CAMPPLUS_MIN_COS, SPEECH_FRACTION_MIN * 100,
             DURATION_MIN_S, DURATION_MAX_S))

    # --neutral-dir lets a partial re-run (e.g. only the emotions whose prompts were
    # revised) still build its centroid from the neutral set that already passed,
    # instead of needing a neutral folder regenerated alongside it.
    def emotion_dir(e):
        if e == "neutral" and args.neutral_dir:
            return args.neutral_dir
        return os.path.join(V2_DIR, e)

    v2 = {e: clips_in(emotion_dir(e)) for e in EMOTIONS}
    if args.neutral_dir:
        print("\nneutral clips (centroid + rows) taken from %s" % args.neutral_dir)
    control = clips_in(V1_NEUTRAL_CONTROL)
    print("\nfound: " + ", ".join("%s=%d" % (e, len(v2[e])) for e in EMOTIONS)
          + " | v1 neutral control=%d" % len(control))

    # ---------------- per-clip acoustic measurement ----------------
    print("\n" + "-" * 100)
    print("STEP 1 -- acoustic measurement (3 F0 estimators + <250 Hz energy + VAD + tail)")
    print("-" * 100)
    print("%-34s %6s %7s %7s %7s %8s %7s %6s %7s %6s"
          % ("clip", "dur_s", "pyin", "autoc", "ceps", "F0", "<250Hz", "spch", "tail_dB", "trail"))
    rows = {}
    for e in EMOTIONS:
        for p in v2[e]:
            m = measure(p)
            m["emotion"] = e
            rows[p] = m
            print("%-34s %6.2f %7.1f %7.1f %7.1f %8.1f %6.1f%% %5.0f%% %7.1f %5.0fms"
                  % (os.path.basename(p), m["duration_s"], m["f0_pyin"], m["f0_autocorr"],
                     m["f0_cepstrum"], m["f0_median"],
                     100 * (m["lowband_frac"] if np.isfinite(m["lowband_frac"]) else np.nan),
                     100 * m["speech_fraction"], m["tail_db_rel"], m["trailing_silence_ms"]))
            if m["f0_note"]:
                print("      ^ F0 reconciliation: %s" % m["f0_note"])

    ctl_rows = {}
    print("\n  v1 neutral control clips (audited as on-anchor keepers):")
    for p in control:
        m = measure(p)
        m["emotion"] = "v1_neutral_control"
        ctl_rows[p] = m
        print("  %-32s %6.2f %7.1f %7.1f %7.1f %8.1f %6.1f%% %5.0f%%"
              % (os.path.basename(p), m["duration_s"], m["f0_pyin"], m["f0_autocorr"],
                 m["f0_cepstrum"], m["f0_median"], 100 * m["lowband_frac"],
                 100 * m["speech_fraction"]))

    # ---------------- CAM++ embeddings ----------------
    print("\n" + "-" * 100)
    print("STEP 2 -- CAM++ embeddings (server.py path: VAD trim, 16 kHz, kaldi fbank 80 mel,")
    print("          dither=0, per-utterance mean subtraction)")
    print("-" * 100)
    enc = CampPlusEncoder("cpu")
    emb = {p: enc.embed(p) for p in list(rows) + list(ctl_rows)}
    print("  embedded %d clips (%d v2 + %d control)" % (len(emb), len(rows), len(ctl_rows)))

    # centroid from v2 neutral clips that pass check 1 (F0 in range)
    neutral_ok = [p for p in v2["neutral"]
                  if np.isfinite(rows[p]["f0_median"])
                  and F0_MIN_HZ <= rows[p]["f0_median"] <= F0_MAX_HZ]
    print("\n  v2 neutral clips passing the F0 gate, used to build the centroid: %d/%d"
          % (len(neutral_ok), len(v2["neutral"])))
    if not neutral_ok:
        print("\n  *** BLOCKER: no v2 neutral clip passes the F0 gate. There is no trustworthy")
        print("      anchor to build a centroid from. Refusing to write a manifest.")
        return 2
    centroid = unit(np.mean([unit(emb[p]) for p in neutral_ok], axis=0))

    # sanity check the centroid against the audited-good v1 neutral controls
    ctl_cos = {p: cos(emb[p], centroid) for p in ctl_rows}
    print("\n  centroid sanity check vs the %d audited-good v1 neutral clips:" % len(ctl_cos))
    for p, c in sorted(ctl_cos.items()):
        print("    %-34s cos=%.4f%s" % (os.path.basename(p), c,
                                        "" if c >= CONTROL_MIN_INDIV_COS else "   <-- LOW"))
    mean_ctl = float(np.mean(list(ctl_cos.values()))) if ctl_cos else np.nan
    print("    mean=%.4f  min=%.4f  (require mean>=%.2f and every clip>=%.2f)"
          % (mean_ctl, min(ctl_cos.values()), CONTROL_MIN_MEAN_COS, CONTROL_MIN_INDIV_COS))
    if not (mean_ctl >= CONTROL_MIN_MEAN_COS and min(ctl_cos.values()) >= CONTROL_MIN_INDIV_COS):
        print("\n  *** BLOCKER: the v1 neutral controls do NOT score high against the v2")
        print("      neutral centroid. The v2 neutral set is itself off-anchor, so the whole")
        print("      v2 pool is measured against the wrong identity. Refusing to write a")
        print("      manifest -- this needs a human decision, not a threshold tweak.")
        return 3
    print("    OK -- v2 neutral centroid agrees with the audited-good v1 neutral anchor.")

    # ---------------- apply the gate ----------------
    print("\n" + "-" * 100)
    print("STEP 3 -- applying the gate")
    print("-" * 100)
    out = []
    for e in EMOTIONS:
        for p in v2[e]:
            m = rows[p]
            c = cos(emb[p], centroid)
            reasons = []
            f0 = m["f0_median"]
            if not np.isfinite(f0):
                reasons.append("f0_unmeasurable")
            elif not (F0_MIN_HZ <= f0 <= F0_MAX_HZ):
                reasons.append("f0_out_of_register(%.1fHz not in %.0f-%.0f)"
                               % (f0, F0_MIN_HZ, F0_MAX_HZ))
            if "NO ESTIMATOR CONSENSUS" in (m["f0_note"] or ""):
                reasons.append("f0_estimators_disagree")
            if c < CAMPPLUS_MIN_COS:
                reasons.append("campplus_cos_low(%.3f<%.2f)" % (c, CAMPPLUS_MIN_COS))
            if m["speech_fraction"] < SPEECH_FRACTION_MIN:
                reasons.append("speech_fraction_low(%.2f<%.2f)"
                               % (m["speech_fraction"], SPEECH_FRACTION_MIN))
            if m["truncated"]:
                reasons.append("truncated_mid_decay(tail %.1fdB, %.0fms trailing silence)"
                               % (m["tail_db_rel"], m["trailing_silence_ms"]))
            if not (DURATION_MIN_S <= m["duration_s"] <= DURATION_MAX_S):
                reasons.append("duration_out_of_range(%.2fs)" % m["duration_s"])
            out.append(dict(
                emotion=e, path=m["path"], f0_median=float(f0) if np.isfinite(f0) else float("nan"),
                campplus_cos_to_neutral_centroid=float(c),
                speech_fraction=float(m["speech_fraction"]),
                duration_s=float(m["duration_s"]),
                passed=len(reasons) == 0, reject_reason="; ".join(reasons),
            ))

    print("\n%-12s %6s %6s %6s   %s" % ("emotion", "total", "pass", "fail", "pass rate"))
    blockers = []
    for e in EMOTIONS:
        sub = [r for r in out if r["emotion"] == e]
        npass = sum(r["passed"] for r in sub)
        print("%-12s %6d %6d %6d   %s" % (e, len(sub), npass, len(sub) - npass,
              "%.0f%%" % (100.0 * npass / len(sub)) if sub else "n/a"))
        if npass < 4:
            blockers.append((e, npass))

    rejects = [r for r in out if not r["passed"]]
    print("\nrejections (%d):" % len(rejects))
    if not rejects:
        print("  none")
    for r in rejects:
        print("  %-34s [%s] %s" % (os.path.basename(r["path"]), r["emotion"], r["reject_reason"]))

    print("\nper-clip CAM++ cosine to the v2 neutral centroid:")
    for e in EMOTIONS:
        sub = [r for r in out if r["emotion"] == e]
        if sub:
            v = [r["campplus_cos_to_neutral_centroid"] for r in sub]
            print("  %-10s n=%d  mean=%.4f  min=%.4f  max=%.4f"
                  % (e, len(v), np.mean(v), min(v), max(v)))

    # ---------------- write the manifest ----------------
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    write_parquet(out, args.out)
    print("\nwrote %s  (%d rows: %d passed, %d rejected -- rejects retained on purpose so the"
          % (args.out, len(out), sum(r["passed"] for r in out), len(rejects)))
    print("      downstream orchestrator can see them when it filters on emotion+passed)")

    if blockers:
        print("\n" + "!" * 100)
        print("BLOCKER -- fewer than 4 clips survived for: "
              + ", ".join("%s (%d)" % (e, n) for e, n in blockers))
        print("These emotions CANNOT go to production as-is.")
        print("!" * 100)

    if args.analysis:
        run_analysis(v2, out, rows, ctl_rows, emb, centroid)

    return 0


def run_analysis(v2, out, rows, ctl_rows, emb, centroid):
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Q1 -- F0 table v1 vs v2
    print("\n[Q1] median F0 per emotion, v1 vs v2")
    v1_ref = {"neutral": 206, "sad": 273, "angry": 407, "happy": 420}
    v1_meas = {}
    for e, d in V1_DIRS.items():
        ps = clips_in(d)
        vals, lows = [], []
        for p in ps:
            m = measure(p)
            if np.isfinite(m["f0_median"]):
                vals.append(m["f0_median"])
            if np.isfinite(m["lowband_frac"]):
                lows.append(m["lowband_frac"])
        v1_meas[e] = (np.median(vals) if vals else np.nan,
                      np.median(lows) if lows else np.nan, len(ps))
    print("  %-10s %12s %12s %12s %12s %12s"
          % ("emotion", "v1(report)", "v1(remeas)", "v1 <250Hz", "v2 F0", "v2 <250Hz"))
    for e in EMOTIONS:
        v2p = [rows[p] for p in v2[e]]
        f2 = np.median([m["f0_median"] for m in v2p if np.isfinite(m["f0_median"])]) if v2p else np.nan
        l2 = np.median([m["lowband_frac"] for m in v2p if np.isfinite(m["lowband_frac"])]) if v2p else np.nan
        r1 = v1_ref.get(e)
        m1 = v1_meas.get(e, (np.nan, np.nan, 0))
        print("  %-10s %12s %12s %12s %12s %12s"
              % (e,
                 "%d Hz" % r1 if r1 else "n/a (new)",
                 "%.1f Hz" % m1[0] if np.isfinite(m1[0]) else "n/a",
                 "%.1f%%" % (100 * m1[1]) if np.isfinite(m1[1]) else "n/a",
                 "%.1f Hz" % f2 if np.isfinite(f2) else "n/a",
                 "%.1f%%" % (100 * l2) if np.isfinite(l2) else "n/a"))

    # Q2 -- separation of the high-arousal emotions
    print("\n[Q2] acoustic separation. v1 baseline for reference: MFCC angry<->happy 14.0,")
    print("     angry<->neutral 57.7, within-emotion spread 22.6-27.6, LTAS angry<->happy 0.961")
    passed_by_e = {}
    for e in EMOTIONS:
        passed_by_e[e] = [r["path"] for r in out if r["emotion"] == e and r["passed"]]
    separation_report({e: clips_in(os.path.join(V2_DIR, e)) for e in EMOTIONS}, "v2 (all clips)")
    if all(passed_by_e[e] for e in EMOTIONS):
        separation_report(passed_by_e, "v2 (passing clips only)")
    separation_report({e: clips_in(d) for e, d in V1_DIRS.items()}, "v1 (recomputed here)")

    # CAM++ cross-emotion identity, the check that caught v1
    print("\n  CAM++ cross-emotion cosine (v1 failed here: 0.700 within vs 0.529 across,")
    print("  same-speaker threshold ~0.562):")
    allp = {e: clips_in(os.path.join(V2_DIR, e)) for e in EMOTIONS}
    names = [e for e in EMOTIONS if allp[e]]
    print("    %-10s" % "" + "".join("%9s" % n[:8] for n in names))
    for i in names:
        row = []
        for j in names:
            if i == j:
                vs = [cos(emb[a], emb[b]) for ai, a in enumerate(allp[i])
                      for b in allp[i][ai + 1:]]
            else:
                vs = [cos(emb[a], emb[b]) for a in allp[i] for b in allp[j]]
            row.append("%9.3f" % np.mean(vs) if vs else "%9s" % "-")
        print("    %-10s" % i + "".join(row))

    # Q3 -- did the no-crying instruction clean up sad?
    print("\n[Q3] sad: non-speech content. v1 sad was 4.0 non-speech segments/clip, 44-66% speech")
    for label, ps in (("v1 sad", clips_in(V1_DIRS["sad"])), ("v2 sad", v2["sad"])):
        ms = [rows[p] if p in rows else measure(p) for p in ps]
        if not ms:
            continue
        print("  %-8s n=%d  speech %.0f-%.0f%% (mean %.0f%%)  non-speech segments/clip mean %.1f"
              % (label, len(ms), 100 * min(m["speech_fraction"] for m in ms),
                 100 * max(m["speech_fraction"] for m in ms),
                 100 * np.mean([m["speech_fraction"] for m in ms]),
                 np.mean([m["nonspeech_segments"] for m in ms])))
        for m, p in zip(ms, ps):
            print("      %-32s speech %.0f%%  segments %d"
                  % (os.path.basename(p), 100 * m["speech_fraction"], m["nonspeech_segments"]))


if __name__ == "__main__":
    sys.exit(main())

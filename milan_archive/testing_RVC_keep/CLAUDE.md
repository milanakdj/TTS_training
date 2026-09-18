# testing_RVC — RVC voice-conversion evaluation

Evaluation of RVC (Retrieval-based Voice Conversion) for copying a target speaker's
voice onto Nepali/Hindi TTS output. **Conclusion reached 2026-09-03: RVC is the wrong
tool for this goal — see [Verdict](#verdict).**

## Layout

| path | what |
|---|---|
`RVC-WebUI/` | RVC-Project upstream @ `81eed5e`. Training + inference for every `v40k_*` / `v48k` run. |
`rvc-webui-ddPn08/` | ddPn08 fork, tested as an alternative. `infer_cli.py` is ours, not upstream. |
`testing/Generated Audio ... _join.mp3` | **The training audio for every ladder run.** |
`input_audio/source_indic_tts_audio.wav` | The conversion source clip for every render. Hindi, 16 kHz, 7.6 s. |
`compare/{40k_ladder,48k,ddpn08,lowlr}/` | 113 scored renders, one per checkpoint per precision. |
`infer_out/` | Ad-hoc one-off renders from earlier sessions. Mixed sources — not comparable. |
`testdata_10min/` | Feeds the older `sagar_10min` / `sagar_nocache` runs only. Unrelated to the ladders. |

## Training data — read this before interpreting any result

Every ladder run (`sample_1`, `sample_2`, `v48k`, and so every `v40k_ladder*`,
`v40k_lr*`) was sliced from **one** file: `testing/Generated Audio September 02, 2026 -
12_10PM_join.mp3`. Confirmed in `RVC-WebUI/logs/<run>/preprocess.log`.

- 17.9 min, 320 kbps, 48 kHz stereo — **lossy mp3 of joined TTS output**, not a real recording
- sliced to 248 segments / 15.2 min of speech → 249 filelist rows
- batch size 8 → **32 steps/epoch** in every run

`eNN` in a filename or weight name = **training epoch NN of that run**:

| tag | steps | weight file |
|---|---|---|
e05 | 160 | `assets/weights/<run>_e5_s160.pth` |
e10 | 320 | `<run>_e10_s320.pth` |
e25 | 800 | `<run>_e25_s800.pth` |
e50 | 1600 | `<run>_e50_s1600.pth` |

`v40k_ladder_fine` is a **separate 16-epoch run**, not a re-render of `v40k_ladder`.
Its `e10` and `v40k_ladder`'s `e10` are different models.

## Scoring: use ASR CER, never spectral statistics

Spectral flatness / centroid / zcr / DC-ratio **do not measure intelligibility**.
`compare/40k_ladder/fine/fine_e16_fp32.wav` reads clean on every one of them
(flatness 0.0094, centroid 1889, DC ratio 0.0001) and is at **CER 1.00** — not one
word survives. Those stats caught the 48 kHz collapse and missed the 40 kHz one
completely. `compare/48k/metrics.md` and `compare/ddpn08/metrics_raw.md` are built on
them; treat their verdicts as collapse-detection only.

Score with the synthetic pipeline's ASR QC service instead
(`/root/tts/TTS_training/synthetic_pipeline/asr_qc_service`, faster-whisper
`large-v3-turbo`, replicas on ports 8003/8013/8023/8033):

1. `POST /transcribe` the conversion **source** clip once → reference text
2. `POST /qc` each render with that as `expected_text` → per-render CER/WER

Same words go into every render, so CER measures content lost in conversion.
**Caveat:** the service hardcodes `language="ne"` but the source clip is **Hindi**. The
decode is consistent across renders so the ranking is sound, but the CER floor is not 0.

## Results — 113 renders, 2026-09-03

Learning rate decides the collapse point. Sample rate and the GPU-cache flag do not.

| run | lr | last good epoch | CER | first broken | CER |
|---|---|---|---|---|---|
`v40k_ladder` | 1e-4 | e10 | 0.069 | e15 | 0.65 → 1.00 by e25 |
`v40k_ladder_fine` | 1e-4 | e11 | 0.257 | e12 | 0.75 → 1.47 at e14 |
`v48k` | 1e-4 | **e35** | 0.030 | e40 | 0.78 → 1.00 |
`v40k_lr5e5` | 5e-5 | e15 | 0.139 | e20 | 0.83–0.97 |
`v40k_lr2e5` | 2e-5 | **no collapse through e50** | 0.030 | — | worst 0.248 at e50 |
`ddpn08` | — | e05 | 0.069 | — | gradual to 0.733 at e50, no cliff |

Best renders of all 113:

1. `compare/lowlr/v40k_lr5e5_e10_fp32.wav` — **CER 0.0198**, WER 0.045 (320 steps, ir 0)
2. `compare/lowlr/v40k_lr2e5_e05_fp32.wav` — CER 0.0297 (160 steps, ir 0)
3. `compare/48k/v48k_e35_fp32.wav` — CER 0.0297 (1120 steps, ir 0.75)

Takeaways:

- **Peak quality is always e05–e10 (160–320 steps).** Anything past e15 at lr 1e-4 is wasted compute.
- 48 kHz at lr 1e-4 delays collapse from e11 to e40 — it does not prevent it.
- lr 2e-5 never collapses in 1600 steps; lr 5e-5 gives the single best render but dies at e20.
- **fp16 vs fp32 does not affect intelligibility** (pairs track within ±0.05 both ways). fp16 only hides the DC-offset signature from the spectral stats.
- Renders in `compare/lowlr/*_fp32.wav` used index-rate 0 — `logs/v40k_lr2e5` has no `.index`, so 0 is the only setting the two lowlr runs can share. Not comparable to the 48k set's index-rate 0.75.

## Gotchas

- **`-c 1` (cache training set to GPU) collapses the generator.** loss_disc 4.26 → 0.008, loss_mel pinned at the 75 clamp, DC-constant output. Always train with `-c 0`.
- **The device rule picks fp16 on an H100 (SM 9.0)**, which NaN'd at step 600 and froze weights while still logging "Training completed: Success". `RVC_TRAINING_DTYPE` in `configs/config.py` forces fp32; the WebUI must be restarted with it set, since training is an inheriting subprocess.
- **`render_fp32.py`**: patched. `Config.__init__` runs its own argparse over `sys.argv` (`configs/config.py:189`) and rejects the script's flags — sys.argv is now hidden across `Config()` construction. `.bak` beside it.
- **`rvc-webui-ddPn08/lib/rvc/pipeline.py:243`**: patched, upstream bug. The HuBERT input `.half()` cast was gated on GPU fp16 *capability* instead of the chosen precision, so `--precision fp32` fed half inputs into float weights. Now `self.is_half and half_support`. `.bak` beside it. Upstream-reportable.
- `rvc-webui-ddPn08/infer_cli.py` defaults to `--f0-method harvest --index-rate 0`, while RVC-WebUI renders used `rmvpe` + 0.75. Cross-fork numbers carry that confound. Harvest is also not deterministic.

## Verdict

**RVC transfers timbre only. It cannot copy emotion, and no setting changes that.**

- phonetic content → HuBERT features from the **source** clip
- pitch contour, timing, stress, emotion → rmvpe/harvest F0 from the **source** clip
- the trained model contributes speaker colour and nothing else

So output expressiveness equals source expressiveness. Here both ends were flat
synthetic speech — a TTS source clip driving prosody and a lossy TTS-derived mp3
supplying timbre — so there was never any expression in the pipeline to carry across.

**Decision (user, 2026-09-03): use Seed-VC.** The goal is timbre *and* emotion carried
onto the output, which is outside RVC's design. Work continues there rather than tuning
these ladders further.

Still true under Seed-VC: emotion in any voice-conversion system comes from the source
audio being converted. Expressive output needs an expressive source clip — a real
recording, not flat TTS — and a lossless target-speaker reference.

### If RVC work resumes anyway

- Start at lr 2e-5, stop by e10 (320 steps), render fp32, score by CER.
- Replace the training reference with lossless audio (WAV/FLAC) of the real speaker; the current mp3 caps timbre fidelity regardless of hyperparameters.
- Unexplored: `fine_e01`–`e07` (weights exist in `assets/weights/`). Every run peaks at or below e10 and nothing below e08 has been rendered, so the true optimum is probably in that gap.

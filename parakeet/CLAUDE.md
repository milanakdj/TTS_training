# Parakeet → Nepali ASR

Goal: teach `nvidia/parakeet-tdt-0.6b-v3` to transcribe Nepali Devanagari.

Everything lives in `parakeet_nepali.py`. Four steps, run in order, each skips itself
if its output exists:

```
python parakeet_nepali.py smoke      # run first on any new box; see the hardware section
python parakeet_nepali.py manifests
python parakeet_nepali.py tokenizer
python parakeet_nepali.py train
```

Work dir is `~/parakeet-nepali` (audio, manifests, tokenizers, checkpoints).

**`exp_manager` runs with `resume_if_exists: True`.** Any config change you make is
ignored on the next start unless you clear the run dir first:

```bash
pgrep -af "parakeet_nepali.py"        # want nothing; pkill -f "parakeet_nepali.py train"
rm -rf ~/parakeet-nepali/exp
```

Forgetting this silently resumes the old settings from checkpoint state. One trainer
spawns ~16 dataloader workers, so `pkill` then `pgrep` again before restarting.

## The core problem

v3's SentencePiece vocab has 8192 tokens, trained on 25 European languages. It holds
**zero Devanagari**. Every Nepali transcript currently encodes to `<unk>`. The model
physically cannot represent the target text. Extending the vocab is a precondition,
not a tuning knob.

## Merge the tokenizer. Never replace it.

This is the single most important rule in this folder.

`step_tokenizer()` appends Nepali pieces **after** v3's 8192, so every original id keeps
its index. `step_train()` then copies the pretrained embedding and joint rows back by
slice after `change_vocabulary()` reallocates the layers. New Nepali rows stay random.
The encoder is never touched — acoustic features are language-agnostic.

If you instead train a fresh tokenizer and point `change_vocabulary()` at it, NeMo logs:

> "the decoder will be reinitialized"

and all 8192 pretrained rows are gone. The reference Hindi attempt in
`Hindi_GramVani_Finetune-main/` did exactly that (fresh 1024-piece BPE) and produced
garbage after 3 epochs:

```
reference: तो चलिए सुनते है नया कार्यक्रम
predicted: नमस्कार के लिए के लिए रही केेे के में को...
```

That author blamed data volume. Tokenizer replacement was the larger cause.

**Do not copy** `Hindi_GramVani_Finetune-main/tokenize_language.py`, and do not use its
`hindi_config.yaml` tokenizer block (`update_tokenizer: true` on a fresh dir).

## There is no language tag. Do not add one.

Parakeet has no `ne` tag, no `<|ne|>` token, no manifest `lang` field it reads.

- v3's model card: it "automatically detects the language of the audio and transcribes
  it without requiring additional prompting". One unified tokenizer, no per-language
  sub-tokenizers, no lang tokens.
- `nvidia/parakeet-tdt_ctc-0.6b-ja` is a **separate model** trained from scratch on
  35k hours of Japanese. Not a tagged variant of v3.
- Neither community Hindi recipe uses a tag.

The tokenizer merge **is** the whole "add a language" job. Adding `"lang": "ne"` to
manifest lines is harmless but does nothing.

## Verified NeMo facts (measured on this box, do not re-guess)

- `model.joint.joint_net` is `[activation, (Dropout if dropout>0), Linear]`. Index `2`
  is the output Linear **only** when jointnet dropout is non-zero; otherwise the list
  has length 2 and `[2]` raises IndexError. The script uses `joint_net[-1]`.
- Per-module learning rates come from **`model.cfg.optim_param_groups`** (a top-level
  model key), read by `ModelPT.setup_optimizer_param_groups()`. A key placed at
  `model.cfg.optim.param_groups` is **silently ignored** — the encoder would then train
  at the full `4e-4` and be destroyed. Unmatched params (decoder, joint) fall back to
  `cfg.optim.lr`, so only `encoder` needs an entry.
- The prediction embedding has `vocab + 1` rows; the trailing row is blank/SOS and is
  copied back with `[-1]`.
- The TDT joint output is `vocab + 1 blank + 5 durations`. Those 6 tail entries
  (`TDT_TAIL`) are pretrained and language-independent — copied back verbatim.
- `model.cfg.optim.sched` needs its own **`max_steps`**. Without it NeMo logs
  `"Scheduler will not be instantiated !"` — a warning, not an error — and trains with
  no warmup and no decay. `warmup_steps` alone is silently useless. Confirm every run:
  `grep -c "Scheduler will not be instantiated" ~/train.log` must print `0`.
- `pl.Trainer` must be constructed with **`logger=False`** and
  **`enable_checkpointing=False`**. Lightning creates both by default, `exp_manager`
  wants to own both, and it raises rather than pick a winner —
  `LoggerMisconfigurationError` then `CheckpointMisconfigurationError`, one per run.
- The HF `datasets` audio column must be cast before use:
  `ds.cast_column("audio", Audio(sampling_rate=16000))`. Without it `row["audio"]` has
  no `"array"` key at all (`KeyError: 'array'`), and the cast also resamples to the
  16 kHz NeMo's preprocessor expects. Same line the `../whisper/` scripts use.
- v3's dataset config is a **lhotse** config (`use_lhotse: true`), and four of its keys
  are wrong for a NeMo-manifest run. All four are set in `step_train`:
  - `text_field: answer` — our manifests use `text`. Leave it and every transcript
    reads as empty; the run trains on nothing and the loss curve looks plausible.
  - `max_tps: null` — lhotse's `TokenPerSecondFilter` asserts `min_tps <= max_tps`, so
    `int <= None` raises `TypeError` before step 0.
  - `use_bucketing: true` with `bucket_duration_bins: null` — lhotse then scans every
    cut to estimate bins, and batches by `bucket_batch_size`/`batch_duration` (both
    null) rather than `batch_size`. A variable batch size also breaks
    `fused_batch_size = BATCH`.
  - `max_duration: 10.0` is v3's own value; the script had raised it to 20.0. See the
    hardware section — that raise is what OOM'd.

## Training shape and why

| Setting | Value | Reason |
|---|---|---|
| encoder LR | `1e-6` | already a good multilingual acoustic model |
| decoder/joint LR | `4e-4` | Nepali rows start as noise |
| encoder freeze | first `UNFREEZE_AT` 5000 steps | garbage decoder gradients would wreck a good encoder |
| `fuse_loss_wer` | `True` | the `[B,T,U,V]` joint tensor at V≈8704 is the biggest allocation in the run |
| batch / accum | `BATCH` 2 / `ACCUM` 16 | effective batch 32; the largest that survives the unfreeze in 18 GB |
| `max_steps` | `MAX_STEPS` 30000 | **optimizer** steps, not batches. ≈5.5 epochs, ≈17 h at the measured 0.11 s/batch |
| `warmup_steps` | 1000 | was 5000 — the same as `UNFREEZE_AT`, so full LR arrived at the exact step the encoder woke up and the new rows spent the whole safe phase at a reduced rate. Run 1 sat at `val_wer=1.0` (empty output) the entire frozen phase; at 1000 the model emits Devanagari by step ~50 |
| `max_duration` | 10.0 | v3's own default. Peak memory follows the **longest** clip in a batch and grows with duration squared; mean here is 3.8 s |
| `min_duration` | 0.1 | TTS corpora carry empty/clipped rows |
| precision | `bf16-mixed` | |
| `val_check_interval` | 16000 | Lightning counts this in training **batches**, not optimizer steps. At 2000 it fired every 125 steps and cost a measured **16% of wall clock** — validation plus a 2.66 GB checkpoint write each time |
| `num_sanity_val_steps` | `0` | a fresh vocab predicts noise; the check proves nothing |
| `enable_progress_bar` | `True` | with it off Lightning prints nothing between validations, so a live run is indistinguishable from a hung one and there is no `train_step_timing` to size `MAX_STEPS` against |

### Early output looks broken. That is the expected shape.

RNNT with a fresh vocab collapses onto the most frequent piece before it learns to
condition on audio. Observed, in order:

```
step   55   पानगा बातीगा)थन्यथ पर्«थयो हुन्छउथ उन दिन भएकोार
step  475   सकोकोकोमाकोकोकोकोकोको          <- को is the most common Nepali piece
```

`val_wer=1.0` means it emits nothing scoreable. Both are normal while the encoder is
frozen — all the learning is in 436 random rows plus the joint. Judge quality only
**after** the unfreeze. `val_wer` still pinned at `1.0` several thousand steps past the
unfreeze is a real failure; before that it means nothing.

## Target hardware: NVIDIA DGX Spark (GB10)

Knobs are constants at the top of `parakeet_nepali.py`: `BATCH`, `ACCUM`, `MAX_STEPS`,
`UNFREEZE_AT`, `MEM_CAP_GB`. Keep `BATCH * ACCUM == 32`.

**Memory, measured.** 121.69 GiB addressable, one *unified* pool shared by GPU and CPU,
no separate VRAM budget. A vLLM server holds **96.5 GB** of it and is not being shut
down. At the OOM there was **1.71 GiB free of 121.69**. There is no headroom to take, so
`MEM_CAP_GB = 18` is a ceiling to stay under, never a number to raise.

vLLM also competes for *bandwidth*, not just capacity.

**The OOM cliff is at `UNFREEZE_AT`, not step 0.** Run 1 died at step 5000, the moment
the encoder unfroze, inside the RNNT loss backward (`torch.zeros_like(acts)`, 1.57 GiB).
The `MEM_CAP_GB` guard worked exactly as intended: a catchable
`torch.OutOfMemoryError`, and vLLM's 96.5 GB untouched. What fixed it:

- `max_duration` 20.0 → **10.0**. The `[B,T,U,V]` tensor grows with `T` (frames) *and*
  `U` (target tokens), and both scale with duration — so worst-case memory goes as
  **duration squared**. 20→10 is a ~4× cut, not 2×.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set at module scope in the script.
  The OOM report named fragmentation itself (362 MiB reserved but unallocated).
- If it ever OOMs again: `BATCH = 1`, `ACCUM = 32`. Same effective batch, half the
  activation memory. Do **not** raise `MEM_CAP_GB`.

**Probe the cliff in 2 minutes, not 60.** The memory question has nothing to do with
reaching step 5000, so `UNFREEZE_AT` reads from the environment:

```bash
rm -rf ~/parakeet-nepali/exp
PARAKEET_UNFREEZE_AT=20 python parakeet_nepali.py train 2>&1 | tee ~/probe.log
# let it run ~20 min past the unfreeze, Ctrl+C, then:
grep -c OutOfMemory ~/probe.log                 # want 0
rm -rf ~/parakeet-nepali/exp                    # throwaway run
```

The probe answers memory **and** live speed at once. It is not a quality test: with the
encoder unfrozen at step 20 the decoder never gets its safe head start, so its
predictions are meaningless.

**Measured throughput.** `train_step_timing`, per batch:

| Config | frozen | live (post-unfreeze) | 30000 steps × 16 accum |
|---|---|---|---|
| `max_duration = 20.0` | 0.039 s | **0.688 s** | 92 h |
| `max_duration = 10.0` | 0.039 s | **0.11 s** | **≈17 h** |

Plan against the live number, never the frozen one — the collapse at the unfreeze is
18× at 20 s and still 3× at 10 s. The 6× gain from halving `max_duration` is the same
squared-scaling effect that fixed the OOM: a smaller tensor is also faster to fill.

**Bandwidth is the real limit.** 273 GB/s, against ~2 TB/s on an A100. Compute is
plentiful; feeding it is not. This is why 30000 optimizer steps and not the 150000 an
A100 recipe would use.

## Install on aarch64 + CUDA 13 + sm_121

No container is needed and no NeMo container exists for ARM64 anyway.

- **`nemo_toolkit[asr]` does not depend on `torchaudio` at all.** Its audio deps are
  `soundfile`, `librosa`, `scipy`. So the known GB10 torchaudio-CUDA breakage does not
  touch ASR training. NeMo's `AudioToMelSpectrogramPreprocessor` uses `torch.stft`.
- **`nemo_text_processing` (the pynini/OpenFst pain) is already excluded on ARM** by an
  explicit marker in NeMo's own metadata: `"arm" not in platform_machine and "aarch" not
  in platform_machine`. It is a `tts` extra, not an `asr` one, either way.
- Everything left in the `asr` extra is pure Python or has aarch64 wheels.
- PyTorch publishes aarch64 cu130 wheels directly. `nvcr.io/nvidia/pytorch` is the
  correct base only if you want a container.

```bash
# aarch64 cu130 wheels exist -- install torch FIRST so nemo does not drag in an x86 build
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "nemo_toolkit[asr,cu13]"

export TORCH_CUDA_ARCH_LIST="12.1a"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

Installed and working on the Spark: `torch 2.12.0+cu130`, `nemo_toolkit 2.7.3`,
`numba-cuda 0.30.4`, Python 3.12, device `NVIDIA GB10 sm_121`.

**pip's dependency-conflict block on install is noise, not an error.** The shared
`training` venv also holds the Parler TTS packages, and NeMo's transitive upgrades break
them: `parler-tts` wants `transformers<=4.46.1` (venv has 4.57.3), `descript-audiotools`
wants `protobuf<5` (venv has 5.29.6). Neither parakeet nor whisper cares. It matters only
if you want to run `../training.py` again from this venv — give that its own venv.
The `NumbaPerformanceWarning: Grid size N will likely result in GPU under-utilization`
lines are also noise; the test tensors are small.

Then, before anything else:

```bash
python parakeet_nepali.py smoke
```

**The one thing that can genuinely fail is the RNNT loss.** NeMo computes it with
numba-cuda kernels compiled at runtime — hence the `cu13` extra. Nothing else in the
stack asks numba to emit code for sm_121. `step_smoke()` runs a tiny RNNT
forward+backward on the GPU precisely to catch that in seconds, rather than after the
dataset download, the wav export and the vocab merge. **It passes on sm_121** — this was
the project's main open risk and it is closed.

**Guard the memory, or you can brick the box.** Unified memory means a training OOM
takes the whole machine — including the vLLM server sharing the pool.

There is **no sudo on the Spark**, so the cgroup route below is unavailable and the cap
lives inside the process instead: `MEM_CAP_GB = 18` at the top of the script drives
`torch.cuda.set_per_process_memory_fraction()` in `step_train()`. That caps the torch
caching allocator only — cuDNN workspaces allocate outside it — but the `[B,T,U,V]`
joint tensor is the dominant allocation, so an OOM becomes a catchable
`torch.cuda.OutOfMemoryError` rather than a dead box. Confirmed in practice: the run 1
OOM left the box and vLLM healthy. Just run:

```bash
python parakeet_nepali.py train
```

If you ever do get sudo, the cgroup is the stronger guard and should replace it:

```bash
sudo systemd-run --scope -p MemoryMax=18G -p MemorySwapMax=0 \
    python parakeet_nepali.py train
```

Parakeet inference is confirmed working on a Spark
(`github.com/mARTin-B78/dgx-spark-parakeet-asr`).

## Data

- Source: `lilgoose7777/slr-combined-nepali-tts2`, first `NUM_ROWS = 177000` rows.
- Split 0.98 / 0.01 / 0.01 — transducers need very little held-out data.
- **Measured**: 173,459 train / 1,770 val / 1,771 test rows. **183.0 audio-hours**,
  mean clip **3.8 s**. `step_manifests` took 41 minutes at ~70–90 rows/s and wrote
  173k+ wav files at 16 kHz (about 28 GB — check `df -h ~` first).
- `step_manifests` prints a heartbeat every 2000 rows with rows/s, ETA and a running
  audio-hours total, and a `DONE` line per split with the final hours. That hours figure
  is the go / no-go number; no separate command needed.
- Manifests are written to `NAME.json.tmp` and renamed only after the last row, and the
  skip guard checks all three splits. An earlier version checked only that
  `train.json` *existed*, so a run that died at row 0 left an empty file that passed the
  guard forever — "already built, skipping" on nothing.
- **Under ~200 hours, expect a bad WER regardless of the recipe.** The Hindi attempt had
  ~100 hours and failed. This corpus is 183 hours, and 168 after the `max_duration`
  filter. Set expectations at "first model, clean read speech", not production ASR.
- It is **TTS** data: clean, studio, few speakers. Expect good WER on clean read speech
  and poor WER on phone or street audio.

### What `max_duration = 10.0` actually costs

| cap | clips kept | hours kept |
|---|---|---|
| 10 s | 170,207 / 173,459 (98.1%) | **168.1 / 183.0 (91.8%)** |
| 15 s | 171,792 / 173,459 (99.0%) | 173.4 / 183.0 (94.7%) |
| 20 s | 172,634 / 173,459 (99.5%) | 177.4 / 183.0 (97.0%) — this OOM'd |

8.2% of the hours for a 4× memory cut and a 6× speed-up. Clips over the cap are
**dropped**, not truncated. To buy those hours back instead of paying data, the trade is
`max_duration = 15.0` with `BATCH = 1` / `ACCUM = 32` — untested, and it needs its own
probe run. A model trained only on sub-10 s clips may transcribe long audio worse; chunk
at inference.

## Tokenizer source knob

`NE_TOKENIZER_SOURCE` at the top of the script:

- `"train"` (default) — fits fresh Nepali pieces on your own transcripts. Covers every
  character the data actually contains. **Measured: 436 pieces merged, 8192 → 8628,
  fertility 3.18 tokens/word, 0 unknown tokens.**
- `"ai4bharat"` — downloads AI4Bharat IndicVoices `ne_256`, 256 pieces, already trained
  on real Nepali. Zero cost, but **measured on this corpus it fails**: 254 pieces merged,
  fertility 3.63, and **314 unknown tokens**. Its pieces include no danda (`।`) and no
  Devanagari digits (`०-९`), and both appear in these transcripts. Kept only for
  comparison.

The step prints **fertility** (tokens per word) and asserts **zero unknown tokens**.
Do not start training unless unknowns are 0 — a token the tokenizer cannot represent is
a target the model cannot learn.

Fertility 3.18 is above the ~2.5 target. It works, but `U` in the `[B,T,U,V]` joint
tensor is target length, so every extra token per word costs both speed and memory.
Untested lever: raise `NE_VOCAB_SIZE` from 512 to 1024, delete the tokenizer dirs, re-run
the step (minutes) and compare.

Borrowing another model's *pieces* is safe; adopting a foreign tokenizer *wholesale* is
not, because v3's pretrained rows are keyed to v3's own id→piece mapping.

The merge keeps v3's `normalizer_spec` (`merged.CopyFrom(orig_proto)`). The reference
EN+Hindi script takes the auxiliary tokenizer's normalizer instead — that silently
changes how the 8192 pretrained pieces get matched, and is the one way a borrowed
tokenizer can corrupt weights you meant to preserve. New piece scores are floored below
every original score so SentencePiece never prefers them over pretrained merges.

## Running and monitoring

Training is ~17 hours. Run it under tmux; `tee` a log so you never need to attach.

```bash
tmux new-window -t parakeet -n train2
tmux send-keys -t parakeet:train2 \
  "cd <repo>/parakeet; python parakeet_nepali.py train 2>&1 | tee ~/train.log" C-m
```

`tmux send-keys` into a pane that is already running a process does nothing useful — the
keystrokes go to that process as stdin. Always start in a fresh window.

Status, without attaching:

```bash
pgrep -af "parakeet_nepali.py train"
tail -5 ~/train.log
tmux capture-pane -pt parakeet:train2 -S -5
grep -c OutOfMemory ~/train.log                          # want 0
grep -c "Scheduler will not be instantiated" ~/train.log  # want 0
grep -E "\[mem\]|\[vocab\]|\[freeze\]" ~/train.log        # want all three
grep -oE "val_wer=[0-9.]+" ~/train.log | tail -8          # must fall below 1.0
grep -A2 "WER reference" ~/train.log | tail -6            # ref vs hyp pairs
```

Two numbers on the progress bar mean different things: `it/s` is a lifetime average that
includes validation and checkpoint pauses; `train_step_timing` is the current cost of one
batch. Size the run from `train_step_timing`. The gap between them is your overhead — it
was 16% before `val_check_interval` was raised.

Don't sit and watch. Park a one-shot watcher in its own window:

```bash
tmux new-window -t parakeet -n watch
tmux send-keys -t parakeet:watch \
  "grep -m1 -E 'encoder unfrozen|OutOfMemory' <(tail -f ~/train.log); printf '\\a'; date" C-m
```

## Finishing

`step_train` saves `~/parakeet-nepali/parakeet-tdt-0.6b-v3-nepali.nemo` and then runs
`trainer.test()` on the untouched test split by itself. That `test_wer` is the honest
number — train and val WER look fine long before a vocab extension generalises.

Smoke-test the saved file before trusting it; a model that trains but will not restore is
worthless:

```bash
python3 - <<'EOF'
import json, os
import nemo.collections.asr as nemo_asr
NEMO = os.path.expanduser("~/parakeet-nepali/parakeet-tdt-0.6b-v3-nepali.nemo")
TEST = os.path.expanduser("~/parakeet-nepali/manifests/test.json")
m = nemo_asr.models.ASRModel.restore_from(NEMO, map_location="cuda"); m.eval()
rows = [json.loads(l) for _, l in zip(range(5), open(TEST, encoding="utf-8"))]
for r, h in zip(rows, m.transcribe([r["audio_filepath"] for r in rows], batch_size=1)):
    print("ref:", r["text"]); print("hyp:", getattr(h, "text", h)); print()
EOF
```

Hub push is one command; there is no model-card writer for parakeet the way there is in
`../whisper/whisper-eval.py`:

```bash
export HF_TOKEN=hf_...
huggingface-cli upload milanakdj/parakeet-tdt-0.6b-v3-nepali \
  ~/parakeet-nepali/parakeet-tdt-0.6b-v3-nepali.nemo \
  parakeet-tdt-0.6b-v3-nepali.nemo
```

## Status

Done:

1. `smoke` passes on the Spark — the numba-cuda RNNT kernel compiles for sm_121. This was
   the project's main risk.
2. `manifests` built: 183.0 audio-hours, 173,459 train rows.
3. `tokenizer` merged: 8192 → 8628, fertility 3.18, **0 unknown tokens**.
4. `train` reaches and survives the unfreeze at `BATCH = 2` with `max_duration = 10.0`,
   verified by a 20-minute `PARAKEET_UNFREEZE_AT=20` probe.

Open:

5. **No completed training run yet.** No `.nemo`, no `test_wer`.
6. Nepali text normalization is undecided. The merged vocab shows the transcripts carry
   danda (`।`), Devanagari digits, brackets, `»`, `«`, `→`, `●` and a zero-width joiner.
   Whatever is left in the text is what the model learns — and it is what the WER is
   scored against.
7. Fertility 3.18 is above the ~2.5 target. Try `NE_VOCAB_SIZE = 1024`.
8. `fuse_loss_wer=True` computes training WER every batch. Cost not isolated from the
   0.11 s/batch figure; `model.cfg.log_prediction = False` would at least stop the log
   spam.
9. Hub push and model card are not wired into the script.
10. `nvidia-parakeet-fine-tuning.ipynb` is dead scaffolding (targets v2, English, a dummy
    LibriSpeech split). Delete or ignore.

## Local reference material

- `Hindi_GramVani_Finetune-main/` — the community Hindi attempt. Useful as a **negative**
  example for the tokenizer, and for its VRAM and config numbers.
- `nvidia-parakeet-fine-tuning.ipynb` — abandoned v2 English exploration.
- `../whisper/` — the whisper-large-v3 Nepali pipeline on the same dataset and box.
  Reached **11.45% WER**, which is the number to beat.

## External links

- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/45 — Hindi recipe
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/47 — the poor-WER thread
- https://github.com/NVIDIA/NeMo/issues/13825 — same author's follow-up
- https://github.com/jeremy110/Finetune_Nemo_ASR — Chinese port
- https://github.com/furkanksl/parakeet-asr-finetuning-pipeline
- https://github.com/mARTin-B78/dgx-spark-parakeet-asr — Parakeet running on a Spark
- https://github.com/natolambert/dgx-spark-setup — ML training setup for GB10 / CUDA 13
- https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html

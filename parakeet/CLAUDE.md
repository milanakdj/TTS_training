# Parakeet → Nepali ASR

Goal: teach `nvidia/parakeet-tdt-0.6b-v3` to transcribe Nepali Devanagari.

Everything lives in `parakeet_nepali.py`. Three steps, run in order, each skips itself
if its output exists:

```
python parakeet_nepali.py smoke      # run first on any new box; see the hardware section
python parakeet_nepali.py manifests
python parakeet_nepali.py tokenizer
python parakeet_nepali.py train
```

Work dir is `~/parakeet-nepali` (audio, manifests, tokenizers, checkpoints).

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

## Verified NeMo facts (checked against NeMo `main`, do not re-guess)

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

## Training shape and why

| Setting | Value | Reason |
|---|---|---|
| encoder LR | `1e-6` | already a good multilingual acoustic model |
| decoder/joint LR | `4e-4` | Nepali rows start as noise |
| encoder freeze | first 5000 steps | garbage decoder gradients would wreck a good encoder |
| `fuse_loss_wer` | `True` | the `[B,T,U,V]` joint tensor at V≈8704 is the biggest allocation in the run |
| batch / accum | `BATCH` 2 / `ACCUM` 16 | effective batch 32; sized for ~21 GB free on the Spark |
| `max_steps` | `MAX_STEPS` 30000 | **optimizer** steps, not batches ≈ 5 epochs of 177k rows |
| precision | `bf16-mixed` | |
| `min_duration` | `0.1` | TTS corpora carry empty/clipped rows |
| `num_sanity_val_steps` | `0` | a fresh vocab predicts noise; the check proves nothing |

## Target hardware: NVIDIA DGX Spark (GB10)

The four hardware knobs are constants at the top of `parakeet_nepali.py`:
`BATCH`, `ACCUM`, `MAX_STEPS`, `UNFREEZE_AT`. Keep `BATCH * ACCUM == 32`.

**Memory.** The Spark has 128 GB of *unified* LPDDR5X and the GPU can address
effectively all of it — there is no separate VRAM budget. A vLLM inference server holds
~107 GB of that pool and is not being shut down, so the working budget is a firm
**21 GB**. That is what `BATCH = 2` is sized for. If vLLM ever goes away, `BATCH = 8`
fits with room to spare.

Note that vLLM also competes for *bandwidth*, not just capacity. Expect worse it/s than
the 273 GB/s figure suggests while it is serving.

**The OOM cliff is at `UNFREEZE_AT`, not step 0.** Reported elsewhere: batch 4 fits
~11 GB *with the encoder frozen*. Unfreezing adds encoder activations. A run that
survived 5000 steps is not proof the config fits.

**Bandwidth is the real limit.** 273 GB/s, against ~2 TB/s on an A100. Compute is
plentiful; feeding it is not. Measure it/s over the first 100 steps and set `MAX_STEPS`
to a number you will actually wait for. This is why the default is 30000 optimizer
steps and not the 150000 an A100 recipe would use.

## Install on aarch64 + CUDA 13 + sm_121

No container is needed and no NeMo container exists for ARM64 anyway. Checked against
`nemo_toolkit` 3.0.0 metadata on PyPI:

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

Then, before anything else:

```bash
python parakeet_nepali.py smoke
```

**The one thing that can genuinely fail is the RNNT loss.** NeMo computes it with
numba-cuda kernels compiled at runtime — hence the `cu13` extra. Nothing else in the
stack asks numba to emit code for sm_121. `step_smoke()` runs a tiny RNNT
forward+backward on the GPU precisely to catch that in seconds, rather than after the
dataset download, the wav export and the vocab merge.

**Guard the memory, or you can brick the box.** Unified memory means a training OOM
takes the whole machine — including the vLLM server sharing the pool. Cap the training
process instead of trusting it:

```bash
sudo systemd-run --scope -p MemoryMax=18G -p MemorySwapMax=0 \
    python parakeet_nepali.py train
```

Parakeet inference is confirmed working on a Spark
(`github.com/mARTin-B78/dgx-spark-parakeet-asr`). Fine-tuning on one is not a trodden
path — budget setup time regardless of the above.

## Data

- Source: `lilgoose7777/slr-combined-nepali-tts2`, first `NUM_ROWS = 177000` rows.
- Split 0.98 / 0.01 / 0.01 — transducers need very little held-out data.
- **Measure the hours before trusting the run.** The Hindi attempt had ~100 hours and
  failed. Under ~200 hours, expect a bad WER regardless of the recipe:
  ```
  python3 -c "import json;print(sum(json.loads(l)['duration'] for l in open('$HOME/parakeet-nepali/manifests/train.json'))/3600,'hours')"
  ```
- It is **TTS** data: clean, studio, few speakers. Expect good WER on clean read speech
  and poor WER on phone or street audio. Fine for a first model; know the ceiling.
- `step_manifests` writes 177,000 individual `.wav` files. Check free disk first.

## Tokenizer source knob

`NE_TOKENIZER_SOURCE` at the top of the script:

- `"ai4bharat"` (default) — downloads AI4Bharat IndicVoices `ne_256`, 256 pieces,
  already trained on real Nepali. Zero cost.
- `"train"` — fits fresh Nepali pieces on your own transcripts. Better fertility on
  your domain, costs a training pass.

The step prints **fertility** (tokens per word) and asserts **zero unknown tokens**.
Under ~2.5 tokens/word is fine. If the assert fires, or fertility is high, switch to
`"train"` and compare. Borrowing another model's *pieces* is safe; adopting a foreign
tokenizer *wholesale* is not, because v3's pretrained rows are keyed to v3's own
id→piece mapping.

The merge keeps v3's `normalizer_spec` (`merged.CopyFrom(orig_proto)`). The reference
EN+Hindi script takes the auxiliary tokenizer's normalizer instead — that silently
changes how the 8192 pretrained pieces get matched, and is the one way a borrowed
tokenizer can corrupt weights you meant to preserve. New piece scores are floored below
every original score so SentencePiece never prefers them over pretrained merges.

## Pending

Blockers:

1. `python parakeet_nepali.py smoke` on the Spark. Nothing else can be tested until it
   passes. The likely failure is the numba-cuda RNNT kernel on sm_121.
2. Measure training hours (command above). This is a go / no-go number.
3. Nothing has been run yet — no `~/parakeet-nepali` exists.

Open:

4. Nepali text normalization is undecided. v3 emits punctuation and capitalization.
   Transcripts likely contain `।` (danda) and possibly Devanagari digits. Whatever is
   left in the text is what the model learns.
5. No inference smoke test for the finished `.nemo`.
6. Hub push is not wired up. `../push_checkpoint_to_hub.py` exists but is unconnected.
7. `nvidia-parakeet-fine-tuning.ipynb` is dead scaffolding (targets v2, English, a dummy
   LibriSpeech split). Delete or ignore.
8. Nothing in this folder is committed to git.

## Local reference material

- `Hindi_GramVani_Finetune-main/` — the community Hindi attempt. Useful as a **negative**
  example for the tokenizer, and for its VRAM and config numbers.
- `nvidia-parakeet-fine-tuning.ipynb` — abandoned v2 English exploration.

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

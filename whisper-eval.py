#!/usr/bin/env python3
"""Benchmark one or many trained Whisper checkpoints on the held-out test split,
compare them, then write the model card for the winner to a Hub repo.

Rebuilds the exact same test split as whisper-finetune.py (same NUM_ROWS, same
empty-transcript filter, same shuffle SEED, same 80/10/10 carve). The dataset is
loaded ONCE and every checkpoint is scored on the identical clips, so the numbers
are comparable. No training, no train-split preprocessing.

The card's training facts are READ from the checkpoint's own trainer_state.json
and training_args.bin -- never retyped -- so the card cannot drift from the run.
Those two files stay local (they are excluded from the Hub upload), so point
--ckpt at the local dir to get them.

A checkpoint can be:
    a local dir            /home/x/whisper-output/large-v3/checkpoint-26550
    a Hub repo             milanakdj/whisper-large-v3-nepali-final-largev3_1
    a subfolder in a repo  milanakdj/whisper-large-v3-nepali-checkpoints/checkpoint-8850

Usage:
    # a model from the Hub (downloads it), on the split it was trained with
    python whisper-eval.py --ckpt milanakdj/whisper-medium-nepali-final \
        --num-rows 20000 --n 200 --batch 8

    # every local checkpoint, quick timing pass
    python whisper-eval.py --all --n 200

    # score several, then publish the best one's card
    python whisper-eval.py --all --n 2000 --batch 16 \
        --push-card milanakdj/whisper-large-v3-nepali-final-largev3_1

    # render the card offline and assert it is well-formed
    python whisper-eval.py --self-check

WARNING: --num-rows must match the NUM_ROWS the model was trained with. The row
slice is taken BEFORE the shuffle, so a 20000-row run and a 177000-row run have
completely different test sets. Score a 20k model against the 177k split and some
"held-out" clips were in its training data -- the WER comes out flatteringly wrong.
"""
import argparse
import glob
import json
import os
import re
import time

import evaluate
import torch
from datasets import Audio, load_dataset
from huggingface_hub import HfApi
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --- must match whisper-finetune.py ---
HF_DATASET_ID = "lilgoose7777/slr-combined-nepali-tts2"
HF_SPLIT = "train"
NUM_ROWS = 177000
SEED = 42
TRAIN_VAL_TEST_SPLIT = (0.80, 0.10, 0.10)
MODEL_VARIANT = "large-v3"  # only used for the default --output-dir path
BASE_MODEL = f"openai/whisper-{MODEL_VARIANT}"
LANGUAGE, TASK = "nepali", "transcribe"
LANG_TAG = "ne"  # the token is <|ne|>; LANGUAGE is the human name generate() takes

# d_model -> variant. large-v3 is the only one with 128 mel bins; large-v2 shares
# its d_model but uses 80. Getting this wrong means feeding a medium model 128-bin
# features (or the wrong vocab size), and the output is fluent-looking garbage --
# which is why the processor is chosen per checkpoint, from the model's own config.
VARIANT_BY_D_MODEL = {384: "tiny", 512: "base", 768: "small", 1024: "medium"}
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "whisper-output", MODEL_VARIANT
)


def step_of(ckpt):
    """Sort key / display name: the step number in 'checkpoint-26550', else 0."""
    m = re.search(r"checkpoint-(\d+)", str(ckpt))
    return int(m.group(1)) if m else 0


def local_checkpoints(output_dir):
    return sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=step_of)


def split_ckpt(ckpt):
    """A Hub path with 3+ segments means the last one is a subfolder inside the
    repo -- that is how the checkpoint backup repo is laid out. Local dirs and
    plain 'user/repo' ids pass through with no subfolder."""
    if os.path.isdir(ckpt):
        return ckpt, ""
    parts = ckpt.strip("/").split("/")
    if len(parts) >= 3:
        return "/".join(parts[:-1]), parts[-1]
    return ckpt, ""


def detect_base_model(config):
    d = getattr(config, "d_model", None)
    if d == 1280:
        return "openai/whisper-large-v3" if config.num_mel_bins == 128 \
            else "openai/whisper-large-v2"
    variant = VARIANT_BY_D_MODEL.get(d)
    if variant is None:
        raise SystemExit(f"Unrecognised Whisper size: d_model={d}, "
                         f"num_mel_bins={getattr(config, 'num_mel_bins', None)}")
    return f"openai/whisper-{variant}"


def bf16_ok():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def print_vram(tag):
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] {tag}: {free / 1e9:.1f}GB free of {total / 1e9:.1f}GB "
          f"(free is across ALL processes on this card)", flush=True)


def load_model(ckpt, token, force_tie):
    repo_or_dir, subfolder = split_ckpt(ckpt)
    cuda = torch.cuda.is_available()
    print_vram(f"before loading {os.path.basename(str(ckpt).rstrip('/'))}")
    # device_map puts each shard straight onto the GPU as it is read. Without it,
    # from_pretrained builds the whole model in CPU RAM and .to("cuda") then needs
    # the full 3.1GB free in one contiguous go on top of the copy -- twice the peak
    # for no reason, and it is what OOM'd on this shared card.
    model = WhisperForConditionalGeneration.from_pretrained(
        repo_or_dir,
        subfolder=subfolder,
        token=token,
        dtype=torch.bfloat16 if bf16_ok() else torch.float16,
        device_map="cuda" if cuda else None,
        low_cpu_mem_usage=True,
    )
    if not cuda:
        model.to("cpu")
    model.eval()
    print_vram("after load")
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    # Whisper's output head `proj_out` is weight-tied to the decoder token
    # embeddings and is therefore NOT saved in the checkpoint -- that is what the
    # "missing keys ['proj_out.weight']" line in the training log means. If the
    # saved config has tie_word_embeddings=False, from_pretrained skips the tying
    # and the head stays RANDOM: the model then emits endless nonsense and WER
    # comes out in the hundreds. Checking is two lines, so check.
    if force_tie:
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
        print("[eval] proj_out re-tied to decoder embeddings", flush=True)
    tied = torch.equal(model.proj_out.weight, model.model.decoder.embed_tokens.weight)
    print(f"[eval] tie_word_embeddings={model.config.tie_word_embeddings} "
          f"proj_out_tied={tied}", flush=True)
    if not tied:
        raise SystemExit(
            "proj_out is NOT tied to the decoder embeddings -- the output head is "
            "random and every metric is meaningless. Re-run with --force-tie, and "
            "fix tie_word_embeddings in the checkpoint's config.json."
        )

    # Per checkpoint, never global: a medium model needs 80-bin features and a
    # 51865-token vocab, large-v3 needs 128 bins and 51866. Mismatch = garbage.
    base = detect_base_model(model.config)
    processor = WhisperProcessor.from_pretrained(base, language=LANGUAGE, task=TASK)
    print(f"[eval] detected {base} (d_model={model.config.d_model}, "
          f"mel_bins={model.config.num_mel_bins})", flush=True)
    return model, processor, base


STANDARD_USAGE = """```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = "{repo_id}"
device   = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(model_id, language="{language}", task="{task}")
model     = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

# audio: 1-D float32 numpy array at 16kHz
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
with torch.inference_mode():
    ids = model.generate(inputs.input_features, max_new_tokens=225)

print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```"""

DOUBLED_PREFIX_WARNING = """
> **READ THIS FIRST.** This model does **not** work with plain
> `model.generate()`, and it does not work with `pipeline()`, faster-whisper, or
> whisper.cpp. It was trained with a doubled `<|startoftranscript|>` in the
> decoder prefix, so it must be decoded with that same prefix. The copy-paste
> function below does it. With the correct prefix the model scores the WER in the
> table; with the standard prefix it emits endless nonsense (WER above 400%).
> The weights are fine -- only the prefix convention is unusual.
"""

DOUBLED_PREFIX_USAGE = """This model needs a doubled `<|startoftranscript|>` prefix. Copy this function --
it is self-contained and needs nothing but `transformers` and `torch`.

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "{repo_id}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(MODEL_ID, language="{language}", task="{task}")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE).eval()


def transcribe(audio_batch, max_new_tokens=225):
    \"\"\"audio_batch: list of 1-D float32 numpy arrays at 16 kHz. Returns a list of
    strings. Greedy decoding with the doubled-sot prefix this model was trained on.\"\"\"
    tok = processor.tokenizer
    tid = lambda t: tok.convert_tokens_to_ids(t)
    # The doubled <|startoftranscript|> is the whole point -- one sot decodes garbage.
    prefix = [tid("<|startoftranscript|>"), tid("<|startoftranscript|>"),
              tid("<|{lang_tag}|>"), tid("<|{task}|>"), tid("<|notimestamps|>")]
    eos = tid("<|endoftext|>")

    feats = processor.feature_extractor(
        audio_batch, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE, dtype=model.dtype)

    cur = torch.tensor([prefix] * len(audio_batch), device=DEVICE)
    out, past = cur, None
    done = torch.zeros(len(audio_batch), dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        enc = model.get_encoder()(feats)
        for _ in range(max_new_tokens):
            res = model(encoder_outputs=enc, decoder_input_ids=cur,
                        past_key_values=past, use_cache=True)
            past = res.past_key_values
            nxt = res.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == eos
            if bool(done.all()):
                break
            cur = nxt[:, None]
    return processor.batch_decode(out, skip_special_tokens=True)


# one file
import librosa
audio, _ = librosa.load("clip.wav", sr=16000)
print(transcribe([audio])[0])
```

### Why the prefix is doubled

The fine-tuning collator stripped the leading `<|startoftranscript|>` from the
labels only when `labels[:, 0] == tokenizer.bos_token_id`. For Whisper
`bos_token_id` is `<|endoftext|>` (50257), never `<|startoftranscript|>` (50258),
so the strip never fired. The Trainer then prepended another one while shifting
labels into `decoder_input_ids`, and the model learned this prefix:

```
trained on :  [sot, sot, lang, task, notimestamps, w1, w2, ...]
generate() :  [sot,      lang, task, notimestamps, ...]
```

Teacher-forced loss stayed at 0.04 the whole time, so nothing looked wrong until
generation was measured. Feeding the doubled prefix realigns every position."""


SPECIAL = {"sot": "<|startoftranscript|>", "lang": "<|ne|>",
           "task": "<|transcribe|>", "nots": "<|notimestamps|>"}


def greedy_decode(model, processor, feats, max_new_tokens, n_sot):
    """Greedy decode with an explicit decoder prefix, so the number of leading
    <|startoftranscript|> tokens can be chosen.

    Why this exists: a collator bug (comparing labels[:,0] against bos_token_id
    instead of decoder_start_token_id) left the sot token in the labels, so the
    Trainer's shift produced a DOUBLED prefix and the model trained against
    [sot, sot, lang, task, notimestamps]. model.generate() always feeds the single-sot
    prefix, so such a model decodes garbage. n_sot=2 feeds the convention it was
    actually trained on. If n_sot=2 scores sanely and n_sot=1 does not, the
    checkpoint is fine and only the label alignment was wrong.
    """
    tok = processor.tokenizer
    ids = {k: tok.convert_tokens_to_ids(v) for k, v in SPECIAL.items()}
    prefix = [ids["sot"]] * n_sot + [ids["lang"], ids["task"], ids["nots"]]
    bsz = feats.shape[0]
    cur = torch.tensor([prefix] * bsz, device=model.device)
    out = cur
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    done = torch.zeros(bsz, dtype=torch.bool, device=model.device)
    past = None
    with torch.no_grad():
        enc = model.get_encoder()(feats)
        for _ in range(max_new_tokens):
            res = model(encoder_outputs=enc, decoder_input_ids=cur,
                        past_key_values=past, use_cache=True)
            past = res.past_key_values
            nxt = res.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == eos
            if bool(done.all()):
                break
            cur = nxt[:, None]
    return out


def score(model, processor, test, batch, peek, dump=None, n_sot=None):
    preds, refs = [], []
    t0 = time.time()
    for i in range(0, len(test), batch):
        rows = test[i : i + batch]
        feats = processor.feature_extractor(
            [a["array"] for a in rows["audio"]],
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(model.device, dtype=model.dtype)
        if n_sot:
            ids = greedy_decode(model, processor, feats, 225, n_sot)
        else:
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=225)
        preds += processor.batch_decode(ids, skip_special_tokens=True)
        refs += rows["text"]
        done = len(preds)
        rate = done / (time.time() - t0)
        print(
            f"[eval] {done}/{len(test)}  {rate:.1f} ex/s  "
            f"eta {(len(test) - done) / max(rate, 1e-9) / 60:.1f} min",
            flush=True,
        )

    if peek:
        print(f"\n--- first {peek} predictions ---")
        for p_, r_ in list(zip(preds, refs))[:peek]:
            print(f"  pred ({len(p_)} chars): {p_[:300]}")
            print(f"  ref  ({len(r_)} chars): {r_[:300]}\n")
        avg_p = sum(len(x) for x in preds) / len(preds)
        avg_r = sum(len(x) for x in refs) / len(refs)
        print(f"[eval] mean length: pred {avg_p:.0f} chars vs ref {avg_r:.0f} chars "
              f"({avg_p / max(avg_r, 1e-9):.1f}x)\n", flush=True)

    if dump:
        with open(dump, "w", encoding="utf-8") as f:
            f.write("pred\tref\n")
            for p_, r_ in zip(preds, refs):
                f.write(p_.replace("\t", " ") + "\t" + r_.replace("\t", " ") + "\n")
        print(f"[eval] wrote {dump}", flush=True)

    wer = 100 * evaluate.load("wer").compute(predictions=preds, references=refs)
    cer = 100 * evaluate.load("cer").compute(predictions=preds, references=refs)
    return wer, cer


def read_training_facts(ckpt_dir):
    """Pull what the run actually used out of the checkpoint. Both files are
    optional -- a repo-only checkpoint has neither, and the card says n/a."""
    facts = {}
    if not os.path.isdir(ckpt_dir):
        return facts
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            state = json.load(f)
        facts["steps"] = state.get("global_step")
        facts["best_wer"] = state.get("best_metric")
        facts["epoch"] = state.get("epoch")
        losses = [h["loss"] for h in state.get("log_history", []) if "loss" in h]
        facts["train_loss"] = losses[-1] if losses else None

    args_path = os.path.join(ckpt_dir, "training_args.bin")
    if os.path.isfile(args_path):
        try:
            facts["args"] = torch.load(args_path, weights_only=False)
        except Exception as e:  # pickle across transformers versions is fragile
            print(f"[card] couldn't read training_args.bin ({e}) -- config table "
                  f"will show n/a", flush=True)
    return facts


def comparison_table(results):
    """results: list of (ckpt, wer, cer), already scored on identical clips."""
    if len(results) < 2:
        return ""
    rows = "\n".join(
        f"| `{os.path.basename(str(c).rstrip('/'))}` | {w:.2f}% | {ce:.2f}% |"
        for c, w, ce in sorted(results, key=lambda r: r[1])
    )
    return f"""
## Checkpoint comparison

Every checkpoint scored on the same test clips, best first.

| Checkpoint | WER | CER |
|---|---|---|
{rows}

The weights in this repo are the best row.
"""


def build_model_card(repo_id, ckpt_dir, facts, wer, cer, n_scored, n_test,
                     split_sizes, results=(), base_model=BASE_MODEL,
                     num_rows=NUM_ROWS, n_sot=0):
    variant = base_model.split("whisper-")[-1]
    ckpt_repo = f"milanakdj/whisper-{variant}-nepali-checkpoints"
    a = facts.get("args")
    g = lambda name, default="n/a": getattr(a, name, default) if a is not None else default
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
    precision = "n/a"
    if a is not None:
        precision = "bf16" if a.bf16 else ("fp16" if a.fp16 else "fp32")
    lr = g("learning_rate", None)
    lr_str = "n/a" if lr in (None, "n/a") else f"{lr:.0e}"
    tb, ga = g("per_device_train_batch_size", None), g("gradient_accumulation_steps", None)
    eff = "n/a" if None in (tb, ga) or "n/a" in (tb, ga) else tb * ga
    warm = g("warmup_ratio", None)
    warm_str = "n/a" if warm in (None, "n/a") else f"{warm:.0%}"
    usage = (DOUBLED_PREFIX_USAGE.format(repo_id=repo_id, language=LANGUAGE, task=TASK,
                                         lang_tag=LANG_TAG)
             if n_sot == 2 else
             STANDARD_USAGE.format(repo_id=repo_id, language=LANGUAGE, task=TASK))
    quirk = DOUBLED_PREFIX_WARNING if n_sot == 2 else ""
    # The per-epoch numbers were produced by the Trainer's own generate() call,
    # which used the standard single-sot prefix -- so they are in the hundreds and
    # look like they contradict the test WER. Say why, or the table reads as broken.
    eval_note = ("\nThe per-epoch eval WER above is high because the training loop "
                 "decoded with the standard single-`<|startoftranscript|>` prefix. "
                 "The test WER is this model decoded correctly. Same weights.\n"
                 if n_sot == 2 else "")

    return f"""---
language:
- ne
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- nepali
- asr
- speech-recognition
- finetuned
base_model: {base_model}
datasets:
- {HF_DATASET_ID}
metrics:
- wer
- cer
---

# Whisper {variant} -- Nepali ASR (fine-tuned)

Nepali fine-tune of [`{base_model}`](https://huggingface.co/{base_model}) for `{TASK}`.
Weights are `{os.path.basename(str(ckpt_dir).rstrip("/"))}` of the run, promoted to this repo.
{quirk}
## Results

Scored with `whisper-eval.py` on a held-out test split the model never saw
during training (same dataset slice, same seed {SEED}, same 80/10/10 carve).

| Metric | Value |
|---|---|
| Test WER | {fmt(wer)}% |
| Test CER | {fmt(cer)}% |
| Test examples scored | {n_scored} of {n_test} |
| Best per-epoch eval WER | {fmt(facts.get("best_wer"))}% |
| Final train loss | {fmt(facts.get("train_loss"))} |
| Steps trained | {facts.get("steps", "n/a")} |
| Epochs completed | {fmt(facts.get("epoch"))} |
{eval_note}{comparison_table(results)}
## Training configuration

| Parameter | Value |
|---|---|
| Base model | `{base_model}` |
| Language / task | `{LANGUAGE}` / `{TASK}` |
| Epochs | {g("num_train_epochs")} |
| Learning rate | {lr_str} |
| LR schedule | linear with {warm_str} warmup ratio |
| Per-device train batch | {tb} |
| Grad accumulation steps | {ga} |
| Effective batch size | {eff} |
| Precision | {precision} |
| Gradient checkpointing | {g("gradient_checkpointing")} |
| Optimizer | {g("optim")} |
| Weight decay | {g("weight_decay")} |
| Max grad norm | {g("max_grad_norm")} |
| Generation max length | {g("generation_max_length")} |
| Eval / save strategy | per epoch (best model by WER kept) |
| Seed | {SEED} |

## Data

| Parameter | Value |
|---|---|
| Dataset | [`{HF_DATASET_ID}`](https://huggingface.co/datasets/{HF_DATASET_ID}) |
| Rows requested | {num_rows} |
| Split | {TRAIN_VAL_TEST_SPLIT[0]:.0%} / {TRAIN_VAL_TEST_SPLIT[1]:.0%} / {TRAIN_VAL_TEST_SPLIT[2]:.0%} (train/val/test) |
| Train / val / test examples | {split_sizes[0]} / {split_sizes[1]} / {split_sizes[2]} |
| Audio sampling rate | 16 kHz mono |
| Checkpoint backups | [`{ckpt_repo}`](https://huggingface.co/{ckpt_repo}) |

The training corpus is clean single-speaker studio audio, so expect degraded accuracy
on noisy real-world recordings with background noise, multiple speakers, or strong accents.

## Usage

{usage}
"""


def self_check():
    """Card must render with no trainer_state.json / training_args.bin (repo-only
    checkpoint) and with them, with and without a comparison table. Runs offline
    -- no model, no dataset, no network."""
    card = build_model_card("u/r", "/x/checkpoint-9", {}, 12.345, 4.5, 100, 9000,
                            (80, 10, 10))
    assert "n/a" in card and "WER | 12.35%" in card and "base_model:" in card
    assert "Checkpoint comparison" not in card, "no table for a single checkpoint"

    class FakeArgs:
        bf16, fp16 = True, False
        learning_rate, num_train_epochs, warmup_ratio = 1e-5, 3, 0.05
        per_device_train_batch_size, gradient_accumulation_steps = 1, 16
        gradient_checkpointing, optim, weight_decay = True, "adafactor", 0.0
        max_grad_norm, generation_max_length = 1.0, 225

    facts = {"args": FakeArgs(), "steps": 26550, "best_wer": 7.1, "epoch": 3.0,
             "train_loss": 0.0558}
    results = [("/x/checkpoint-8850", 15.0, 5.0), ("/x/checkpoint-26550", 8.2, 2.1),
               ("/x/checkpoint-17700", 9.9, 3.0)]
    card = build_model_card("u/r", "/x/checkpoint-26550", facts, 8.2, 2.1, 2000, 17700,
                            (141_000, 17_700, 17_700), results)
    assert "| Effective batch size | 16 |" in card
    assert "| Precision | bf16 |" in card
    assert "| Learning rate | 1e-05 |" in card
    assert "| LR schedule | linear with 5% warmup ratio |" in card
    assert "| Steps trained | 26550 |" in card
    assert "n/a" not in card, "every field should be filled when args are present"
    # best row must be first in the comparison table
    table = card.split("## Checkpoint comparison")[1]
    assert table.index("checkpoint-26550") < table.index("checkpoint-17700") \
        < table.index("checkpoint-8850"), "comparison table must be sorted by WER"

    # a medium card must not claim to be large-v3 anywhere
    med = build_model_card("u/r", "/x/checkpoint-9", {}, 12.3, 4.5, 100, 9000,
                           (16000, 2000, 2000),
                           base_model="openai/whisper-medium", num_rows=20000)
    assert "# Whisper medium" in med and "large-v3" not in med
    assert "| Rows requested | 20000 |" in med

    class C:
        d_model, num_mel_bins = 1024, 80
    assert detect_base_model(C) == "openai/whisper-medium"
    C.d_model, C.num_mel_bins = 1280, 128
    assert detect_base_model(C) == "openai/whisper-large-v3"
    C.num_mel_bins = 80
    assert detect_base_model(C) == "openai/whisper-large-v2"

    # a doubled-prefix card must warn, and must NOT hand out the standard snippet
    dbl = build_model_card("u/r", "/x/checkpoint-26550", facts, 8.2, 2.1, 2000, 17700,
                           (141_000, 17_700, 17_700), results, n_sot=2)
    assert "READ THIS FIRST" in dbl and "def transcribe(" in dbl
    assert "ids = model.generate(inputs.input_features" not in dbl, \
        "the standard snippet must not appear on a doubled-prefix card"
    std = build_model_card("u/r", "/x/checkpoint-26550", facts, 8.2, 2.1, 2000, 17700,
                           (141_000, 17_700, 17_700), results, n_sot=0)
    assert "READ THIS FIRST" not in std and "ids = model.generate(" in std

    assert split_ckpt("u/repo/checkpoint-10") == ("u/repo", "checkpoint-10")
    assert split_ckpt("u/repo") == ("u/repo", "")
    assert step_of("/a/b/checkpoint-8850") == 8850 and step_of("openai/whisper") == 0
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="one or more checkpoint dirs / Hub ids / repo-subfolder paths "
                         "(default: newest checkpoint-* under --output-dir)")
    ap.add_argument("--all", action="store_true",
                    help="score every checkpoint-* under --output-dir")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--num-rows", type=int, default=NUM_ROWS,
                    help=f"dataset rows the model was trained on, BEFORE the shuffle "
                         f"(default {NUM_ROWS}). The Kaggle runs used 20000 -- pass it "
                         f"or the 'held-out' clips may be ones the model trained on.")
    ap.add_argument("--n", type=int, default=2000, help="test examples to score (0 = all)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dump", default=None, help="write pred/ref pairs to this TSV "
                    "(one file per checkpoint when several are scored)")
    ap.add_argument("--push-card", default=None, metavar="REPO_ID",
                    help="after scoring, write README.md (+ processor files) to this "
                         "repo, using the BEST checkpoint's numbers")
    ap.add_argument("--card-out", default="README.md",
                    help="local path the card is written to (default: ./README.md)")
    ap.add_argument("--force-tie", action="store_true",
                    help="re-tie proj_out to the decoder embeddings after loading")
    ap.add_argument("--n-sot", type=int, default=0, choices=[0, 1, 2],
                    help="0 = normal model.generate() (default). 1 or 2 = manual greedy "
                         "decode with that many leading <|startoftranscript|> tokens. "
                         "Use 2 to test a model trained with the doubled-prefix "
                         "collator bug; compare against 1 as the control.")
    ap.add_argument("--peek", type=int, default=3,
                    help="print this many pred/ref pairs per checkpoint (0 = none)")
    ap.add_argument("--self-check", action="store_true",
                    help="render the card offline and assert it is well-formed, then exit")
    args = ap.parse_args()

    if args.self_check:
        return self_check()

    # Check the token BEFORE scoring, not after. A 401 at the end of a 40-minute
    # eval wastes the whole run for a missing environment variable.
    if args.push_card:
        tok = os.environ.get("HF_TOKEN")
        if not tok:
            raise SystemExit(
                "--push-card needs HF_TOKEN. Set it first:\n"
                "    export HF_TOKEN=hf_...\n"
                "Or drop --push-card, score the model, and upload the card by hand."
            )
        try:
            who = HfApi(token=tok).whoami()["name"]
            print(f"[hub] authenticated as {who}", flush=True)
        except Exception as e:
            raise SystemExit(
                f"HF_TOKEN was rejected before any work started ({e}).\n"
                f"Check it at https://huggingface.co/settings/tokens -- it needs "
                f"write access to {args.push_card}."
            )

    ckpts = args.ckpt or []
    if args.all:
        ckpts += local_checkpoints(args.output_dir)
    if not ckpts:
        ckpts = local_checkpoints(args.output_dir)[-1:]
    if not ckpts:
        raise SystemExit(
            f"No checkpoint found under {args.output_dir}. Pass --ckpt <dir or repo id>."
        )
    ckpts = list(dict.fromkeys(ckpts))  # de-dup, keep order
    print(f"[eval] {len(ckpts)} checkpoint(s) to score:", flush=True)
    for c in ckpts:
        print(f"         {c}", flush=True)

    token = os.environ.get("HF_TOKEN")

    # Dataset once: the load + filter + shuffle is minutes, the scoring is the
    # cheap part per checkpoint. Every checkpoint sees the identical clips.
    # --num-rows MUST match the NUM_ROWS the model was trained with. The slice is
    # taken BEFORE the shuffle, so 20000 rows and 177000 rows give completely
    # different test sets, and clips held out from a 177k run may have been
    # TRAINED ON in a 20k run. Wrong value here = leaked test set = fake WER.
    print(f"[eval] rebuilding split from the first {args.num_rows} dataset rows "
          f"(must match the NUM_ROWS this model was trained with)", flush=True)
    ds = load_dataset(HF_DATASET_ID, split=f"{HF_SPLIT}[:{args.num_rows}]", token=token)
    ds = ds.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")
    ds = ds.shuffle(seed=SEED)
    n = len(ds)
    n_train = int(n * TRAIN_VAL_TEST_SPLIT[0])
    n_val = int(n * TRAIN_VAL_TEST_SPLIT[1])
    split_sizes = (n_train, n_val, n - n_train - n_val)
    # Cross-check against the "train=... val=... test=..." line in the training log.
    # If these differ, the split moved and this WER is not held-out.
    print(f"[eval] split: train={split_sizes[0]} val={split_sizes[1]} "
          f"test={split_sizes[2]}", flush=True)

    test = ds.select(range(n_train + n_val, n))
    n_test = len(test)
    if args.n:
        test = test.select(range(min(args.n, n_test)))
    test = test.cast_column("audio", Audio(sampling_rate=16000))
    print(f"[eval] scoring {len(test)} examples at batch {args.batch}", flush=True)

    results, bases = [], {}
    for idx, ckpt in enumerate(ckpts, 1):
        print(f"\n===== [{idx}/{len(ckpts)}] {ckpt} =====", flush=True)
        dump = args.dump
        if dump and len(ckpts) > 1:
            root, ext = os.path.splitext(dump)
            dump = f"{root}.{os.path.basename(str(ckpt).rstrip('/'))}{ext}"
        try:
            model, processor, base = load_model(ckpt, token, args.force_tie)
        except torch.OutOfMemoryError as e:
            raise SystemExit(
                f"Out of GPU memory loading {ckpt}.\n"
                f"  1. `nvidia-smi` -- is the old training/eval job still running? "
                f"Kill it; it holds ~14GB.\n"
                f"  2. The vLLM server on this card holds ~99GB of 122GB.\n"
                f"  3. large-v3 in bf16 needs ~3.1GB for weights plus ~1-2GB per "
                f"batch of 4. Try --batch 2.\n"
                f"  4. CPU fallback: CUDA_VISIBLE_DEVICES= python whisper-eval.py ... "
                f"(correct, but ~50x slower).\n"
                f"Original error: {e}"
            )
        bases[ckpt] = base
        wer, cer = score(model, processor, test, args.batch, args.peek, dump,
                         n_sot=args.n_sot)
        print(f"[result] {ckpt}  WER {wer:.2f}%  CER {cer:.2f}%", flush=True)
        results.append((ckpt, wer, cer))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranked = sorted(results, key=lambda r: r[1])
    print(f"\n=== {len(ranked)} checkpoint(s) on {len(test)} test examples ===")
    print(f"{'checkpoint':<50} {'WER':>8} {'CER':>8}")
    for c, w, ce in ranked:
        print(f"{os.path.basename(str(c).rstrip('/')):<50} {w:>7.2f}% {ce:>7.2f}%")

    best_ckpt, best_wer, best_cer = ranked[0]
    print(f"\n[eval] best: {best_ckpt} at WER {best_wer:.2f}%", flush=True)

    # A WER above 100% is not a bad model, it is a broken run: the model emitted
    # more wrong words than the reference has words. Publishing that would be
    # worse than publishing nothing.
    if best_wer > 100:
        raise SystemExit(
            f"Best WER is {best_wer:.1f}%, above 100% -- the decode is broken, not "
            f"merely bad. No card written. Look at the pred/ref pairs above, and "
            f"compare against the base model: --ckpt {BASE_MODEL}"
        )

    card = build_model_card(
        args.push_card or best_ckpt, best_ckpt, read_training_facts(best_ckpt),
        best_wer, best_cer, len(test), n_test, split_sizes, results,
        base_model=bases.get(best_ckpt, BASE_MODEL), num_rows=args.num_rows,
        n_sot=args.n_sot,
    )
    with open(args.card_out, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[card] wrote {args.card_out} for {best_ckpt}")

    if args.push_card:
        api = HfApi(token=token)
        api.create_repo(args.push_card, repo_type="model", exist_ok=True)
        # Processor files first: push_to_hub writes its own stub card, so ours
        # has to land last or it gets overwritten.
        WhisperProcessor.from_pretrained(
            bases.get(best_ckpt, BASE_MODEL), language=LANGUAGE, task=TASK
        ).push_to_hub(args.push_card, token=token)
        api.upload_file(
            path_or_fileobj=args.card_out,
            path_in_repo="README.md",
            repo_id=args.push_card,
            commit_message=f"Model card -- test WER {best_wer:.2f}% on {len(test)} examples",
        )
        print(f"[card] pushed to https://huggingface.co/{args.push_card}")
        if os.path.basename(str(best_ckpt).rstrip("/")) not in ("checkpoint-26550",):
            print(f"[card] NOTE: the card describes {best_ckpt}. If that is not the "
                  f"checkpoint whose weights you uploaded to {args.push_card}, upload "
                  f"the winner's weights too or the card lies.", flush=True)


if __name__ == "__main__":
    main()

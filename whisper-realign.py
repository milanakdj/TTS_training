#!/usr/bin/env python3
"""Short corrective fine-tune: teach a checkpoint the STANDARD Whisper prefix.

Why this exists
---------------
whisper-finetune.py's collator stripped the leading <|startoftranscript|> only if
labels[:,0] == tokenizer.bos_token_id. For Whisper bos_token_id is <|endoftext|>
(50257), never <|startoftranscript|> (50258), so the strip never fired. The
Trainer then prepended another sot when shifting labels, and the model trained
against a DOUBLED prefix:

    trained on :  [sot, sot, lang, task, notimestamps, w1, w2, ...]
    generate() :  [sot,      lang, task, notimestamps, ...]

Every position is off by one, so model.generate() produced fluent garbage (WER
588% during training) while eval_loss stayed at 0.04 -- teacher forcing hid it.
Proof the weights are fine: decoding checkpoint-26550 with a doubled prefix
scores WER 8.33%, with a single prefix 465%.

This script fixes the CONVENTION, not the knowledge. It resumes from the good
weights, trains a few hundred steps with the CORRECT collator, and saves a model
that plain model.generate() handles. The encoder is frozen by default: the
misalignment is purely a decoder-side convention, and freezing halves the
optimizer/gradient memory on a shared GPU.

Cost: MAX_STEPS steps at the same speed as the original run, so ~1-3 hours for
1000 steps -- not the 80 hours of a retrain.

Usage:
    python whisper-realign.py --self-check          # collator logic, no GPU
    python whisper-realign.py \
        --ckpt /home/tarka_milan/whisper-output/large-v3/checkpoint-26550 \
        --steps 1000 --rows 8000

    # then confirm the standard prefix works (n-sot 1 = plain generate convention)
    python whisper-eval.py --ckpt ~/whisper-output/large-v3/realigned --n 32 --batch 4
"""
import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# --- must match whisper-finetune.py so we only ever touch TRAIN data ---
HF_DATASET_ID = "lilgoose7777/slr-combined-nepali-tts2"
HF_SPLIT = "train"
NUM_ROWS = 177000
SEED = 42
TRAIN_FRAC = 0.80
LANGUAGE, TASK = "nepali", "transcribe"
MAX_LABEL_LENGTH = 448


@dataclass
class Collator:
    """The corrected collator. `start` is decoder_start_token_id, NOT bos_token_id
    -- that single wrong constant is the whole bug this script exists to undo."""

    processor: Any
    start: int

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.start).all().cpu().item():
            labels = labels[:, 1:]
        assert not bool((labels[:, 0] == self.start).any()), (
            "a label row still starts with <|startoftranscript|> -- the doubled "
            "prefix bug would come straight back"
        )
        batch["labels"] = labels
        return batch


class _Batch(dict):
    """Minimal stand-in for a tokenizer BatchEncoding: dict access plus the
    .attention_mask attribute the collator reads."""

    def __init__(self, ids):
        super().__init__(input_ids=ids)
        self.attention_mask = torch.ones_like(ids)


def self_check():
    """The strip must fire with decoder_start_token_id and must NOT fire with
    bos_token_id -- the second case reproduces the original bug on purpose."""

    class FakeFE:
        def pad(self, feats, return_tensors=None):
            return {"input_features": torch.zeros(len(feats), 2)}

    class FakeTok:
        def pad(self, rows, return_tensors=None):
            return _Batch(torch.tensor([r["input_ids"] for r in rows]))

    class FakeProc:
        feature_extractor, tokenizer = FakeFE(), FakeTok()

    SOT, BOS, LANG = 50258, 50257, 50364
    rows = [{"input_features": [0.0], "labels": [SOT, LANG, 50360, 50363, 10, 11, BOS]}] * 2

    fixed = Collator(FakeProc(), SOT)(rows)
    assert fixed["labels"][0, 0].item() == LANG, "sot must be stripped"
    assert fixed["labels"].shape[1] == 6, "exactly one token removed"

    # Control: reproduce the original bug. With bos_token_id the strip never fires
    # and the sot survives into the labels -- which is exactly what trained the
    # doubled prefix. Note the in-collator guard cannot catch this on its own: it
    # compares against the same wrong constant. The real protection is main()'s
    # assert that `start == model.config.decoder_start_token_id`.
    buggy = Collator(FakeProc(), BOS)(rows)
    assert buggy["labels"][0, 0].item() == SOT, "control: wrong id must not strip"
    assert buggy["labels"].shape[1] == 7, "control: nothing removed"

    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir with the good weights")
    ap.add_argument("--out", default=None,
                    help="where to save (default: <ckpt's parent>/realigned)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=8000,
                    help="training examples to preprocess (only TRAIN split is touched)")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="low on purpose: relearn the prefix, do not relearn Nepali")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--train-encoder", action="store_true",
                    help="also train the encoder (2x the memory, no benefit here)")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        return self_check()

    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt.rstrip("/")), "realigned")
    token = os.environ.get("HF_TOKEN")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {free / 1e9:.1f}GB free of {total / 1e9:.1f}GB", flush=True)
        if free < 10e9:
            print("[gpu] WARNING: training large-v3 needs roughly 8-10GB with the "
                  "encoder frozen. Less than 10GB free will likely OOM -- wait for "
                  "the other job on this card, or lower --batch.", flush=True)

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3", language=LANGUAGE, task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.ckpt, token=token)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # required with gradient checkpointing

    start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    assert start == model.config.decoder_start_token_id, (
        f"decoder_start_token_id {model.config.decoder_start_token_id} != "
        f"<|startoftranscript|> {start}"
    )

    if not args.train_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[realign] encoder frozen; {trainable / 1e6:.0f}M trainable params",
              flush=True)

    # TRAIN split only. Same slice, same seed, same 80% cut as the original run, so
    # nothing from validation or test leaks into this.
    ds = load_dataset(HF_DATASET_ID, split=f"{HF_SPLIT}[:{NUM_ROWS}]", token=token)
    ds = ds.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")
    ds = ds.shuffle(seed=SEED)
    train = ds.select(range(int(len(ds) * TRAIN_FRAC)))
    train = train.select(range(min(args.rows, len(train))))
    train = train.cast_column("audio", Audio(sampling_rate=16000))
    print(f"[realign] preprocessing {len(train)} train examples", flush=True)

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train = train.map(prepare, remove_columns=train.column_names,
                      num_proc=max(1, (os.cpu_count() or 2) - 2))
    train = train.filter(lambda l: len(l) <= MAX_LABEL_LENGTH, input_columns=["labels"])
    print(f"[realign] {len(train)} examples after length filter", flush=True)

    trainer = Seq2SeqTrainer(
        args=Seq2SeqTrainingArguments(
            output_dir=out_dir,
            max_steps=args.steps,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            warmup_steps=min(50, args.steps // 10),
            optim="adafactor",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            dataloader_num_workers=4,
            logging_steps=25,
            save_strategy="no",  # one model at the end; nothing to rank
            report_to=["none"],
        ),
        model=model,
        train_dataset=train,
        data_collator=Collator(processor, start),
        processing_class=processor,
    )
    trainer.train()

    model.config.use_cache = True
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    print(f"\n[realign] saved to {out_dir}")
    print("[realign] now check that PLAIN generate() works (no --n-sot flag):")
    print(f"    python whisper-eval.py --ckpt {out_dir} --n 32 --batch 4")
    print("[realign] WER near 10% means the prefix is fixed. Still in the hundreds "
          "means it needs more steps.")


if __name__ == "__main__":
    main()

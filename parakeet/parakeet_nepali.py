#!/usr/bin/env python3
"""Extend nvidia/parakeet-tdt-0.6b-v3 with Nepali Devanagari, then fine-tune.

v3's unified SentencePiece vocab (8192 tokens) was trained on 25 European
languages -- Latin, Cyrillic, Greek. It contains zero Devanagari, so every
Nepali transcript currently encodes to <unk> and the model cannot represent the
target text at all. Vocabulary extension is a precondition here, not a tuning
knob.

The recipe: train a small Nepali BPE, append it to v3's tokenizer so the
original 8192 ids keep their positions, then copy the pretrained decoder/joint
weights back into those same positions after change_vocabulary() reallocates
the layers. New Nepali rows stay randomly initialised. The encoder is never
touched by any of this -- acoustic features are language-agnostic, which is why
this works at all.

Run the steps in order; each skips itself if its output already exists:

    python parakeet_nepali.py manifests
    python parakeet_nepali.py tokenizer
    python parakeet_nepali.py train
"""

import json
import os
import sys

WORK = os.path.expanduser("~/parakeet-nepali")
MANIFEST_DIR = os.path.join(WORK, "manifests")
AUDIO_DIR = os.path.join(WORK, "audio")
NE_TOKENIZER_DIR = os.path.join(WORK, "tokenizer-ne")  # Nepali-only, 512 pieces
MERGED_TOKENIZER_DIR = os.path.join(WORK, "tokenizer-merged")  # 8192 + Nepali
EXP_DIR = os.path.join(WORK, "exp")

BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
HF_DATASET_ID = "lilgoose7777/slr-combined-nepali-tts2"
NUM_ROWS = 177000
SPLIT = (0.98, 0.01, 0.01)  # transducers need very little held-out data
SEED = 42

# Where the Nepali pieces come from. Borrowing another model's tokenizer is fine
# -- a SentencePiece model is just pieces + scores, and the merge below is what
# assigns them ids. What you cannot do is adopt a foreign tokenizer wholesale:
# v3's 8192 pretrained embedding rows are keyed to v3's own id->piece mapping, so
# replacing that mapping points every pretrained row at the wrong token.
#   "ai4bharat" -- AI4Bharat IndicVoices ne_256, 256 BPE pieces, already trained
#                  on real Nepali. Zero cost, and the same source the reference
#                  EN+Hindi port used.
#   "train"     -- fit fresh pieces on your own transcripts. Better fertility on
#                  your domain, at the cost of a training pass.
# Run the `tokenizer` step both ways and compare the fertility numbers it prints.
NE_TOKENIZER_SOURCE = "ai4bharat"
NE_VOCAB_SIZE = 512  # only used when NE_TOKENIZER_SOURCE == "train"
AI4BHARAT_NE_URL = (
    "https://raw.githubusercontent.com/AI4Bharat/IndicVoices/master"
    "/artifacts/tokenizers/ne_256/tokenizer_spe_bpe_v256/tokenizer.model"
)

# v3's TDT joint emits vocab + 1 blank + 5 durations. Those 6 tail entries are
# pretrained and language-independent, so they get copied back verbatim.
TDT_TAIL = 6


# --------------------------------------------------------------------------
def step_manifests():
    """HF dataset -> wav files + NeMo line-delimited JSON manifests."""
    import soundfile as sf
    from datasets import load_dataset

    if os.path.exists(os.path.join(MANIFEST_DIR, "train.json")):
        print("[manifests] already built, skipping")
        return

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    ds = load_dataset(HF_DATASET_ID, split=f"train[:{NUM_ROWS}]")
    ds = ds.train_test_split(test_size=1 - SPLIT[0], seed=SEED)
    holdout = ds["test"].train_test_split(test_size=0.5, seed=SEED)
    splits = {"train": ds["train"], "val": holdout["train"], "test": holdout["test"]}

    for name, split in splits.items():
        path = os.path.join(MANIFEST_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            for i, row in enumerate(split):
                audio, text = row["audio"], row["text"].strip()
                if not text:
                    continue
                wav = os.path.join(AUDIO_DIR, f"{name}-{i:07d}.wav")
                if not os.path.exists(wav):
                    sf.write(wav, audio["array"], audio["sampling_rate"])
                f.write(
                    json.dumps(
                        {
                            "audio_filepath": wav,
                            "duration": len(audio["array"]) / audio["sampling_rate"],
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[manifests] {name}: {path}")


# --------------------------------------------------------------------------
def step_tokenizer():
    """Train a Nepali BPE, then append it to v3's vocab without reordering it."""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
    import nemo.collections.asr as nemo_asr

    if os.path.exists(os.path.join(MERGED_TOKENIZER_DIR, "tokenizer.model")):
        print("[tokenizer] already merged, skipping")
        return

    os.makedirs(NE_TOKENIZER_DIR, exist_ok=True)
    os.makedirs(MERGED_TOKENIZER_DIR, exist_ok=True)

    # 1. Nepali transcripts -> flat text -> BPE
    corpus = os.path.join(WORK, "ne_text.txt")
    if not os.path.exists(corpus):
        with open(corpus, "w", encoding="utf-8") as out:
            for line in open(os.path.join(MANIFEST_DIR, "train.json"), encoding="utf-8"):
                out.write(json.loads(line)["text"] + "\n")

    ne_model_path = os.path.join(NE_TOKENIZER_DIR, "tokenizer.model")
    if NE_TOKENIZER_SOURCE == "ai4bharat":
        import urllib.request

        if not os.path.exists(ne_model_path):
            urllib.request.urlretrieve(AI4BHARAT_NE_URL, ne_model_path)
        print(f"[tokenizer] borrowed AI4Bharat ne_256 -> {ne_model_path}")
    else:
        spm.SentencePieceTrainer.train(
            input=corpus,
            model_prefix=os.path.join(NE_TOKENIZER_DIR, "tokenizer"),
            vocab_size=NE_VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,  # Devanagari is a small closed set -- cover it all
            bos_id=-1,
            eos_id=-1,
            unk_id=0,
            pad_id=-1,
        )

    # 2. Pull v3's own tokenizer straight out of the loaded model. Reading it from
    #    the .nemo archive means guessing at a hashed filename inside the cache.
    base = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL, map_location="cpu")
    orig_path = os.path.join(WORK, "orig_tokenizer.model")
    with open(orig_path, "wb") as f:
        f.write(base.tokenizer.tokenizer.serialized_model_proto())
    print(f"[tokenizer] base vocab: {base.tokenizer.vocab_size}")
    del base

    # 3. Merge: every original piece keeps its index, Nepali pieces append after.
    #    That index stability is the whole trick -- it is what lets the pretrained
    #    embedding and joint rows be copied back by slice in step_train().
    orig_proto, ne_proto = sp_pb2.ModelProto(), sp_pb2.ModelProto()
    orig_proto.ParseFromString(open(orig_path, "rb").read())
    ne_proto.ParseFromString(open(ne_model_path, "rb").read())

    # CopyFrom the base, so the merged model keeps v3's normalizer_spec and
    # model_type. The reference EN+Hindi script takes the AUXILIARY tokenizer's
    # normalizer instead -- that silently changes how the 8192 pretrained pieces
    # get matched, which is the one way a borrowed tokenizer can corrupt weights
    # you meant to preserve.
    merged = sp_pb2.ModelProto()
    merged.CopyFrom(orig_proto)
    seen = {p.piece for p in orig_proto.pieces}
    # Scores are a monotonically decreasing rank. New pieces must sort below every
    # original one or SentencePiece will prefer them over the pretrained merges.
    floor = min(p.score for p in orig_proto.pieces)
    added = 0
    for piece in ne_proto.pieces:
        if piece.type != sp_pb2.ModelProto.SentencePiece.NORMAL or piece.piece in seen:
            continue  # skip <unk> and anything the base vocab already had
        new = merged.pieces.add()
        new.piece, new.score, new.type = piece.piece, floor + piece.score, piece.type
        seen.add(piece.piece)
        added += 1
    merged.trainer_spec.vocab_size = len(merged.pieces)

    out_model = os.path.join(MERGED_TOKENIZER_DIR, "tokenizer.model")
    with open(out_model, "wb") as f:
        f.write(merged.SerializeToString())
    with open(os.path.join(MERGED_TOKENIZER_DIR, "tokenizer.vocab"), "w", encoding="utf-8") as f:
        for p in merged.pieces:
            f.write(f"{p.piece}\t{p.score}\n")
    with open(os.path.join(MERGED_TOKENIZER_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
        for p in merged.pieces:
            f.write(p.piece + "\n")

    sp = spm.SentencePieceProcessor(model_file=out_model)
    print(f"[tokenizer] merged {len(orig_proto.pieces)} + {added} = {sp.get_piece_size()}")

    # Fertility: tokens per word on real transcripts. This is the number that
    # decides borrow-vs-train -- a tokenizer fitted elsewhere may segment your
    # corpus badly, and every extra token per word is a longer target sequence,
    # a slower step, and more for the decoder to get right. Under ~2.5 is fine;
    # much above that, rerun with NE_TOKENIZER_SOURCE = "train" and compare.
    with open(os.path.join(MANIFEST_DIR, "train.json"), encoding="utf-8") as f:
        lines = [json.loads(l)["text"] for _, l in zip(range(2000), f)]
    words = sum(len(t.split()) for t in lines)
    tokens = sum(len(sp.encode(t)) for t in lines)
    unks = sum(sp.encode(t).count(sp.unk_id()) for t in lines)
    print(f"[tokenizer] fertility: {tokens / max(words, 1):.2f} tokens/word over {words} words")
    print(f"[tokenizer] unknown tokens: {unks}")

    probe = lines[0] if lines else "नेपाली भाषा"
    assert sp.decode(sp.encode(probe)) == probe, "Devanagari does not round-trip"
    assert unks == 0, f"{unks} unknown tokens remain -- Nepali pieces did not merge in"


# --------------------------------------------------------------------------
def step_train():
    import lightning.pytorch as pl
    import torch
    from omegaconf import open_dict
    import nemo.collections.asr as nemo_asr
    from nemo.utils.exp_manager import exp_manager

    model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL)
    prev_vocab = model.tokenizer.vocab_size

    # Hold references before change_vocabulary() swaps the layers out. Everything
    # here is either vocab-shaped (needs slice-copying) or vocab-independent but
    # reallocated anyway (needs reattaching wholesale).
    old_embed = model.decoder.prediction.embed
    old_dec_rnn = model.decoder.prediction.dec_rnn
    old_joint_pred, old_joint_enc = model.joint.pred, model.joint.enc
    old_joint_out = model.joint.joint_net[2]

    model.change_vocabulary(
        new_tokenizer_dir=MERGED_TOKENIZER_DIR,
        new_tokenizer_type="bpe",
    )
    print(f"[vocab] {prev_vocab} -> {model.tokenizer.vocab_size}")

    with torch.no_grad():
        # Prediction embedding: original ids by slice, plus the trailing blank/SOS row.
        model.decoder.prediction.embed.weight[:prev_vocab] = old_embed.weight[:prev_vocab]
        model.decoder.prediction.embed.weight[-1] = old_embed.weight[-1]
        # These three carry no vocab dimension -- reattach the pretrained modules.
        model.decoder.prediction.dec_rnn = old_dec_rnn
        model.joint.pred, model.joint.enc = old_joint_pred, old_joint_enc
        # Joint output: original ids, then the blank + 5 duration logits at the tail.
        for attr in ("weight", "bias"):
            new_p, old_p = getattr(model.joint.joint_net[2], attr), getattr(old_joint_out, attr)
            new_p[:prev_vocab] = old_p[:prev_vocab]
            new_p[-TDT_TAIL:] = old_p[-TDT_TAIL:]
    del old_embed, old_dec_rnn, old_joint_pred, old_joint_enc, old_joint_out

    with open_dict(model.cfg):
        for key, name, shuffle in (
            ("train_ds", "train", True),
            ("validation_ds", "val", False),
        ):
            ds = model.cfg[key]
            ds.manifest_filepath = os.path.join(MANIFEST_DIR, f"{name}.json")
            ds.sample_rate = 16000
            ds.batch_size = 4
            ds.shuffle = shuffle
            ds.num_workers = 8
            ds.pin_memory = True
            ds.max_duration = 20.0
            ds.is_tarred = False

        # The RNNT joint materialises a [B, T, U, V] tensor -- at V=8704 that is the
        # single largest allocation in the run, far bigger than the weights. Fusing
        # the loss into the joint computes it in fused_batch_size slices instead.
        model.cfg.joint.fuse_loss_wer = True
        model.cfg.joint.fused_batch_size = 4

        # Encoder is already a good multilingual acoustic model; the decoder/joint
        # rows for Nepali are noise. Same LR for both would wreck the encoder long
        # before the new rows learn anything.
        model.cfg.optim.name = "adamw"
        model.cfg.optim.lr = 4e-4
        model.cfg.optim.weight_decay = 1e-3
        model.cfg.optim.sched.warmup_steps = 5000
        model.cfg.optim.sched.min_lr = 1e-6
        model.cfg.optim.param_groups = {
            "encoder": {"lr": 1e-6},
            "decoder": {"lr": 4e-4},
            "joint": {"lr": 4e-4},
        }

    model.setup_training_data(model.cfg.train_ds)
    model.setup_multiple_validation_data(model.cfg.validation_ds)
    model.setup_optimization(model.cfg.optim)

    class UnfreezeEncoder(pl.Callback):
        """Encoder stays frozen until the new vocab rows stop emitting noise.
        Unfreezing early lets garbage decoder gradients flow back into a good
        encoder; unfreezing late is just slower. 5k steps is the reported knee."""

        def __init__(self, unfreeze_at):
            self.unfreeze_at, self.done = unfreeze_at, False

        def on_train_start(self, trainer, pl_module):
            pl_module.encoder.freeze()
            print(f"[freeze] encoder frozen until step {self.unfreeze_at}", flush=True)

        def on_train_batch_start(self, trainer, pl_module, *_):
            if not self.done and trainer.global_step >= self.unfreeze_at:
                pl_module.encoder.unfreeze()
                self.done = True
                print(f"[freeze] encoder unfrozen at {trainer.global_step}", flush=True)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_steps=150000,
        accumulate_grad_batches=8,  # effective batch 32
        precision="bf16-mixed",
        val_check_interval=2000,
        limit_val_batches=100,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        callbacks=[UnfreezeEncoder(5000)],
    )
    exp_manager(
        trainer,
        {
            "exp_dir": EXP_DIR,
            "name": "parakeet-tdt-0.6b-v3-nepali",
            "resume_if_exists": True,
            "resume_ignore_no_checkpoint": True,
            "checkpoint_callback_params": {"save_top_k": 2, "monitor": "val_wer", "mode": "min"},
        },
    )
    trainer.fit(model)
    model.save_to(os.path.join(WORK, "parakeet-tdt-0.6b-v3-nepali.nemo"))


if __name__ == "__main__":
    steps = {"manifests": step_manifests, "tokenizer": step_tokenizer, "train": step_train}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(steps)}}}")
    os.makedirs(WORK, exist_ok=True)
    steps[sys.argv[1]]()

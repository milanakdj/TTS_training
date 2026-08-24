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

    python parakeet_nepali.py smoke      # run this first on any new box
    python parakeet_nepali.py manifests
    python parakeet_nepali.py tokenizer
    python parakeet_nepali.py train
"""

import json
import os
import sys
import time

# The OOM report named fragmentation directly ("361.96 MiB reserved but unallocated").
# Free to set, and it must land before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
# Measured on this corpus: "ai4bharat" (256 pieces) left 314 unknown tokens and
# fertility 3.63. Its pieces cover no danda (।) and no Devanagari digits, both of which
# are in these transcripts. Fitting on our own text covers every character present.
NE_TOKENIZER_SOURCE = "train"
NE_VOCAB_SIZE = 512  # only used when NE_TOKENIZER_SOURCE == "train"
AI4BHARAT_NE_URL = (
    "https://raw.githubusercontent.com/AI4Bharat/IndicVoices/master"
    "/artifacts/tokenizers/ne_256/tokenizer_spe_bpe_v256/tokenizer.model"
)

# v3's TDT joint emits vocab + 1 blank + 5 durations. Those 6 tail entries are
# pretrained and language-independent, so they get copied back verbatim.
TDT_TAIL = 6

# The only knobs the hardware forces. Tuned for a DGX Spark with ~21 GB of the 128 GB
# unified pool actually free. BATCH * ACCUM is the effective batch -- keep the product
# at 32 when you change either. Free the rest of the pool and BATCH = 8 fits easily.
BATCH = 2
ACCUM = 16
# Optimizer steps, not batches. Spark's 273 GB/s bandwidth makes this the wall-clock
# knob: measure it/s over the first 100 steps, then set a number you will actually wait
# for. 30k steps is ~5 epochs of 177k rows at effective batch 32.
MAX_STEPS = 15000
# Overridable so the OOM cliff can be probed directly. The cliff is the unfreeze, not
# step 5000, and nothing about the memory question needs 5000 steps of warmup first:
#   PARAKEET_UNFREEZE_AT=20 python parakeet_nepali.py train
# answers it in two minutes instead of an hour. Throwaway run -- rm -rf the exp dir after.
UNFREEZE_AT = int(os.environ.get("PARAKEET_UNFREEZE_AT", 5000))
# The Spark has one 128 GB pool shared by GPU and CPU, and a vLLM server holds ~107 GB
# of it. An uncapped training OOM takes the whole box down, vLLM included. With no sudo
# there is no systemd-run cgroup, so the cap goes inside the process instead.
MEM_CAP_GB = 18


# --------------------------------------------------------------------------
def step_smoke():
    """Prove the stack works before spending a day on data.

    On aarch64 + CUDA 13 + sm_121 the fragile part is not NeMo itself -- it is the
    RNNT loss, whose kernels numba-cuda compiles at runtime for whatever arch it
    finds. If sm_121 is unsupported this fails here in seconds instead of after the
    dataset download, the wav export and the vocab merge.
    """
    import torch

    print(f"[smoke] torch {torch.__version__}, cuda {torch.version.cuda}")
    assert torch.cuda.is_available(), "no CUDA device"
    print(f"[smoke] device: {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability()))}")

    import soundfile  # step_manifests writes every wav through this

    from nemo.collections.asr.losses.rnnt import RNNTLoss

    n_classes = 16  # blank is id n_classes, so the logit dim is n_classes + 1
    loss_fn = RNNTLoss(num_classes=n_classes)
    B, T, U = 2, 8, 4
    logits = torch.randn(B, T, U + 1, n_classes + 1, device="cuda", requires_grad=True)
    loss = loss_fn(
        log_probs=logits,
        targets=torch.randint(1, n_classes, (B, U), dtype=torch.int32, device="cuda"),
        input_lengths=torch.full((B,), T, dtype=torch.int32, device="cuda"),
        target_lengths=torch.full((B,), U, dtype=torch.int32, device="cuda"),
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all(), "RNNT loss produced NaN/Inf"
    print(f"[smoke] rnnt loss forward+backward on gpu: {float(loss):.4f}")

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL, map_location="cpu")
    print(f"[smoke] {BASE_MODEL} loaded, vocab {model.tokenizer.vocab_size}")
    print("[smoke] ok")


# --------------------------------------------------------------------------
def step_manifests():
    """HF dataset -> wav files + NeMo line-delimited JSON manifests."""
    import soundfile as sf
    from datasets import Audio, load_dataset

    names = ("train", "val", "test")
    if all(os.path.exists(os.path.join(MANIFEST_DIR, f"{n}.json")) for n in names):
        print("[manifests] already built, skipping")
        return

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    ds = load_dataset(HF_DATASET_ID, split=f"train[:{NUM_ROWS}]")
    # Without this the audio column stays undecoded and row["audio"] has no "array"
    # key at all. It also resamples to the 16 kHz NeMo's preprocessor expects, so the
    # wavs on disk need no second pass. Same line the whisper scripts use.
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.train_test_split(test_size=1 - SPLIT[0], seed=SEED)
    holdout = ds["test"].train_test_split(test_size=0.5, seed=SEED)
    splits = {"train": ds["train"], "val": holdout["train"], "test": holdout["test"]}

    for name, split in splits.items():
        path = os.path.join(MANIFEST_DIR, f"{name}.json")
        total, kept, secs, t0 = len(split), 0, 0.0, time.time()
        print(f"[manifests] {name}: {total} rows -> {path}", flush=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for i, row in enumerate(split):
                audio, text = row["audio"], row["text"].strip()
                if not text:
                    continue
                wav = os.path.join(AUDIO_DIR, f"{name}-{i:07d}.wav")
                if not os.path.exists(wav):
                    sf.write(wav, audio["array"], audio["sampling_rate"])
                dur = len(audio["array"]) / audio["sampling_rate"]
                kept, secs = kept + 1, secs + dur
                f.write(
                    json.dumps(
                        {
                            "audio_filepath": wav,
                            "duration": dur,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # 177k rows is hours of wav writing. Without a heartbeat there is no
                # way to tell a slow run from a hung one, and no ETA to plan around.
                if (i + 1) % 2000 == 0:
                    rate = (i + 1) / max(time.time() - t0, 1e-9)
                    print(f"[manifests] {name} {i + 1}/{total}  {rate:.0f} rows/s  "
                          f"eta {(total - i - 1) / rate / 60:.0f} min  "
                          f"{secs / 3600:.1f} audio-hours", flush=True)
        # Audio hours is the go/no-go number for this whole run: under ~200 hours of
        # training data, the recipe does not matter and the WER comes out bad.
        assert kept, f"{name}: wrote 0 rows -- the dataset load produced nothing"
        os.replace(tmp, path)
        print(f"[manifests] {name}: DONE {kept}/{total} rows kept, "
              f"{secs / 3600:.1f} audio-hours, "
              f"{(time.time() - t0) / 60:.0f} min -> {path}", flush=True)


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

    # ponytail: caps the torch caching allocator only -- cuDNN workspaces and other
    # libraries allocate outside it. That is fine here because the [B,T,U,V] joint
    # tensor is the dominant allocation by far, and this turns a box-killing OOM into
    # a catchable torch.cuda.OutOfMemoryError. Swap for a cgroup if you ever get sudo.
    if torch.cuda.is_available():
        total_gb = torch.cuda.mem_get_info()[1] / 1e9
        torch.cuda.set_per_process_memory_fraction(min(MEM_CAP_GB / total_gb, 1.0))
        print(f"[mem] capped at {MEM_CAP_GB} GB of {total_gb:.0f} GB pool", flush=True)

    model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL)
    prev_vocab = model.tokenizer.vocab_size

    # Hold references before change_vocabulary() swaps the layers out. Everything
    # here is either vocab-shaped (needs slice-copying) or vocab-independent but
    # reallocated anyway (needs reattaching wholesale).
    old_embed = model.decoder.prediction.embed
    old_dec_rnn = model.decoder.prediction.dec_rnn
    old_joint_pred, old_joint_enc = model.joint.pred, model.joint.enc
    # joint_net is [activation, (dropout if >0), Linear] -- index 2 only lands on the
    # Linear when jointnet dropout happens to be non-zero. [-1] is always the output.
    old_joint_out = model.joint.joint_net[-1]

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
            new_p, old_p = getattr(model.joint.joint_net[-1], attr), getattr(old_joint_out, attr)
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
            # v3 ships text_field="answer" (its own manifests used that key) and
            # step_manifests writes "text". Leave it and lhotse reads every
            # transcript as empty -- the run trains on nothing and looks fine.
            ds.text_field = "text"
            ds.batch_size = BATCH
            ds.shuffle = shuffle
            ds.num_workers = 8
            ds.pin_memory = True
            # The [B,T,U,V] joint tensor scales with T, so peak memory is set by the
            # LONGEST clip in a batch, not the mean. This corpus averages 3.8 s, so
            # dropping 20 -> 10 loses very little data and halves the worst case. v3's
            # own config shipped 10.0.
            ds.max_duration = 10.0
            ds.min_duration = 0.1  # TTS corpora carry empty/clipped rows; they blow up the loss
            ds.is_tarred = False
            # v3 leaves max_tps null and lhotse's TokenPerSecondFilter asserts
            # min_tps <= max_tps -- int vs None is a TypeError before step 0. Set both
            # wide enough that nothing is filtered.
            ds.min_tps = 0
            ds.max_tps = 1000
            # Bucketing with bucket_duration_bins=null makes lhotse scan every cut to
            # estimate bins, and it batches by bucket_batch_size / batch_duration (both
            # null) instead of batch_size -- which would also break the fused loss
            # below, since fused_batch_size=BATCH assumes a fixed batch size.
            ds.use_bucketing = False

        # The RNNT joint materialises a [B, T, U, V] tensor -- at V=8704 that is the
        # single largest allocation in the run, far bigger than the weights. Fusing
        # the loss into the joint computes it in fused_batch_size slices instead.
        # ponytail: batch 4 is the reported fit for ~11 GB with the encoder frozen.
        # Unfreezing adds encoder activations and can OOM there, not at step 0 -- so a
        # run that survived 5000 steps is not proof the config fits.
        model.cfg.joint.fuse_loss_wer = True
        model.cfg.joint.fused_batch_size = BATCH

        # Encoder is already a good multilingual acoustic model; the decoder/joint
        # rows for Nepali are noise. Same LR for both would wreck the encoder long
        # before the new rows learn anything.
        model.cfg.optim.name = "adamw"
        model.cfg.optim.lr = 4e-4
        model.cfg.optim.weight_decay = 1e-3
        # Was 5000, the same as UNFREEZE_AT -- so the new Nepali rows spent the entire
        # frozen phase at a reduced LR, and full LR arrived at the same step as the live
        # encoder. Warm up fast, let the decoder learn while the encoder is safe.
        model.cfg.optim.sched.warmup_steps = 1000
        # Without this NeMo logs "Scheduler will not be instantiated !" and carries on
        # with NO warmup and NO decay -- the decoder hits the full 4e-4 at step 1 on
        # random Nepali rows. A warning, not an error, so it is easy to miss.
        model.cfg.optim.sched.max_steps = MAX_STEPS
        model.cfg.optim.sched.min_lr = 1e-6
        # NeMo reads per-module LRs from cfg.optim_param_groups (a top-level model key),
        # NOT from cfg.optim.param_groups -- a stray key under optim is silently dropped
        # and the encoder would then train at the full 4e-4.
        model.cfg.optim_param_groups = {"encoder": {"lr": 1e-6}}

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
        # exp_manager owns BOTH the logger and the checkpoint callback. Lightning
        # creates each by default, and exp_manager errors rather than pick a winner.
        logger=False,
        enable_checkpointing=False,

        max_steps=MAX_STEPS,
        accumulate_grad_batches=ACCUM,
        precision="bf16-mixed",
        # Lightning counts this in training BATCHES, not optimizer steps. At 2000 it
        # fired every 125 steps: measured 16% of wall clock lost to validation plus a
        # 2.66 GB checkpoint write each time. 16000 batches = every 1000 optimizer
        # steps, so 15 checkpoints across the run -- enough to rank.
        val_check_interval=16000,
        limit_val_batches=100,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        # On. With it off Lightning prints nothing between validations, so there is no
        # way to tell a live run from a hung one and no it/s to size MAX_STEPS against
        # -- and it/s is the number this whole run has to be planned around.
        enable_progress_bar=True,
        num_sanity_val_steps=0,  # a fresh vocab predicts noise; the sanity pass proves nothing
        callbacks=[UnfreezeEncoder(UNFREEZE_AT)],
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

    # Held-out WER. train/val WER during a vocab extension look fine long before the
    # model generalises, so the untouched test split is the only honest number.
    test_cfg = model.cfg.validation_ds.copy()
    with open_dict(test_cfg):
        test_cfg.manifest_filepath = os.path.join(MANIFEST_DIR, "test.json")
    model.setup_multiple_test_data(test_cfg)
    trainer.test(model)


if __name__ == "__main__":
    steps = {
        "smoke": step_smoke,
        "manifests": step_manifests,
        "tokenizer": step_tokenizer,
        "train": step_train,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(steps)}}}")
    os.makedirs(WORK, exist_ok=True)
    steps[sys.argv[1]]()

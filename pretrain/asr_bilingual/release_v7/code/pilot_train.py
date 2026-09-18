"""Phase-3 pilot: a small CTC model from random init, per PRETRAINING_FROM_SCRATCH.md.

Step 9 calls phase 3 "the real decision point": a 2-3 day pilot that says whether
the recipe converges at all, for ~5% of the total cost, and it must not be skipped
or scaled past on faith. So this script trains the *pilot* and stops. It does not
roll on into the 120M hybrid -- that decision needs the pilot's numbers and a human.

Architecture comes from the parakeet-tdt_ctc-110m config as a known-good
FastConformer template, then is cut down and stripped to CTC-only:
  * CTC-only removes the B x T x U x V joint tensor entirely, which is where most
    of the memory and a large share of step time goes (Step 2).
  * CTC converges faster and more stably from random init than RNN-T (Step 7.1).
  * d_model/n_layers are reduced to land near the ~32M the pilot calls for.

Weights are RANDOM. The template supplies topology only -- restore_from is never
called, which is the whole point of pretraining rather than finetuning.
"""
import argparse, json, os, sys, time
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager

HERE = os.path.dirname(os.path.abspath(__file__))
TMPL = "/workspace/asr_pretrain_v2/tmpl/model_config.yaml"
TOK = "/workspace/asr_pretrain_v2/tokenizer"   # tokenizer.model + .vocab + vocab.txt


def build_cfg(a):
    """CTC-only config, assembled from the parakeet template's proven blocks.

    This is a real EncDecCTCModelBPE, not a hybrid with ctc_loss_weight=1.0.
    The hybrid computes the transducer loss unconditionally
    (hybrid_rnnt_ctc_models.py:456 applies the CTC branch *after* loss_value is
    already computed), so weighting RNN-T to zero would still have paid the whole
    B x T x U x V joint cost -- which is the exact cost Step 2 says to remove.
    """
    import sentencepiece as _spm
    t = OmegaConf.load(TMPL)
    # getattr, not a.tokenizer_dir: diag_batches.py and grad_agree.py build a cfg
    # from their own argument namespaces and must keep working.
    tok = getattr(a, "tokenizer_dir", None) or TOK
    V = _spm.SentencePieceProcessor(
        model_file=os.path.join(tok, "tokenizer.model")).get_piece_size()

    enc = OmegaConf.to_container(t.encoder, resolve=True)
    enc["d_model"] = a.d_model
    enc["n_layers"] = a.n_layers
    enc["n_heads"] = a.n_heads
    enc["subsampling_factor"] = getattr(a, "subsampling", 8)   # FastConformer default 8x
    enc["feat_in"] = t.preprocessor.features

    spec = OmegaConf.to_container(t.spec_augment, resolve=True)
    # SpecAugment on from the start (Step 7.4) -- from scratch on this much data
    # overfits without it.
    spec["freq_masks"], spec["time_masks"] = a.freq_masks, a.time_masks

    cfg = OmegaConf.create({
        "sample_rate": 16000,
        "compute_eval_loss": True,
        # mean_batch (NeMo's default) averages the *total* CTC loss per
        # utterance, so a bucket of 20 s clips carries ~5x the loss and ~5x the
        # gradient of a bucket of 4 s clips -- grad_agree measured 230 vs 1253.
        # mean_volume divides by the token count instead, which makes every
        # bucket contribute in proportion to what it actually teaches.
        "ctc_reduction": getattr(a, "ctc_reduction", "mean_batch"),
        "log_prediction": False,
        # Our BPE: 2.39 tok/word against the shipped 5.12. One of the four valid
        # reasons to pretrain at all, and free from scratch (Step 3).
        "tokenizer": {"dir": tok, "type": "bpe"},
        "preprocessor": OmegaConf.to_container(t.preprocessor, resolve=True),
        "spec_augment": spec,
        "encoder": enc,
        "decoder": {
            "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
            "feat_in": a.d_model,
            "num_classes": V,
            "vocabulary": [f"<{i}>" for i in range(V)],  # replaced from the BPE
        },
        "optim": {
            "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98],
            "weight_decay": 1e-3,
            # CosineAnnealing, not NoamAnnealing. Step 6 offers either, but Noam
            # rescales the configured lr by d_model^-0.5 * min(step^-0.5,
            # step*warmup^-1.5): at d_model=256, warmup=5000 that turns lr 2.5e-3
            # into an effective *peak* of 2.2e-6, ~1100x too small. The first
            # pilot ran all 35,294 steps that way and sat at val_wer 0.9994.
            # Cosine consumes lr directly, so the number here is the number used.
            # max_steps must be explicit -- lhotse datasets have no __len__.
            "sched": {"name": "CosineAnnealing", "warmup_steps": a.warmup,
                      "min_lr": 1e-6, "max_steps": a.max_steps},
        },
    })
    with open_dict(cfg):
        for ds, man, shuf in (("train_ds", a.train_manifest, True),
                              ("validation_ds", a.val_manifest, False)):
            cfg[ds] = OmegaConf.create({
                "manifest_filepath": man, "sample_rate": 16000,
                "use_lhotse": True, "use_bucketing": True, "num_buckets": 30,
                "shuffle": shuf, "batch_duration": a.batch_duration,
                # Validation loads in-process. With 16 train workers plus 16 val
                # workers the val loop deadlocked at step 2000: GPU fell to 0%,
                # every val worker sat at 0% CPU, and the run sat there for 40
                # minutes. The val set is 2,400 clips -- workers buy nothing.
                "num_workers": (a.workers if ds == "train_ds" else 0),
                "max_duration": a.max_duration,
                "min_duration": 0.3, "defer_setup": False,
                "is_tarred": bool(a.tarred),
                "tarred_audio_filepaths": a.tarred,
                "shard_manifests": False, "shuffle_n": 2048,
            })
            if a.bucket_bins and os.path.exists(a.bucket_bins):
                cfg[ds].bucket_duration_bins = json.load(open(a.bucket_bins))
    if a.input_cfg:
        # Weighted multi-source sampling replaces the flat manifest for training
        # only; validation stays a single bilingual file.
        with open_dict(cfg):
            cfg.train_ds.input_cfg = OmegaConf.load(a.input_cfg)
            cfg.train_ds.manifest_filepath = None
    return cfg


def main():
    p = argparse.ArgumentParser()
    M = "/workspace/asr_pretrain_v2/manifests"
    p.add_argument("--train-manifest", default=f"{M}/train_mix.jsonl")
    p.add_argument("--val-manifest", default=f"{M}/val_mix.jsonl")
    p.add_argument("--tarred", default=None, help="tarred_audio_filepaths glob")
    p.add_argument("--bucket-bins",
                   default="/workspace/asr_pretrain_v2/manifests/bucket_duration_bins.json",
                   help="precomputed lhotse bucket boundaries; without these the "
                        "sampler estimates them by scanning the multiplexed "
                        "sources at startup and effectively hangs")
    p.add_argument("--input-cfg", default=None,
                   help="lhotse input_cfg yaml with per-bucket sampling weights; "
                        "overrides --train-manifest when given")
    p.add_argument("--exp-dir", default="/workspace/asr_pretrain_v2/ckpt")
    p.add_argument("--name", default="pilot_ctc_32m")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--batch-duration", type=float, default=600)
    # Step 6 says max_duration 20, sized for an RNN-T whose joint is B x T x U x V.
    # CTC has no U dimension at all -- the loss is B x T x V -- so a 60 s clip
    # costs linearly in T, not quadratically. Raising the cap is what makes the
    # 30-59 s OOV corpus usable at all; at 20 s lhotse silently drops every row
    # of it. Bucketing keeps the padding waste down.
    p.add_argument("--max-duration", type=float, default=60)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.5e-3)
    # Step 6 specifies warmup 15000, sized for the full multi-100k-step run.
    # The pilot is ~35k steps total, where 15k would be 43% of the run spent
    # warming up. 5k keeps the "long warmup, 2k will diverge" property at pilot
    # scale; raise it back to 15000 for phase 4.
    p.add_argument("--warmup", type=int, default=5000)
    # The pilot is denominated in audio hours seen, not steps (Step 1: the
    # schedule is hours-of-audio, and the dataloader feeds the same bytes/s
    # whatever the model size).
    p.add_argument("--hours-seen", type=float, default=5000)
    p.add_argument("--audio-per-step", type=float, default=None,
                   help="s of audio per step; measured by bench, not guessed")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--limit-val-batches", type=int, default=20,
                   help="bound the val loop; unbounded validation is what there "
                        "is no reason to pay for every 2000 steps")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="0 disables clipping. CTC loss here is ~300 and the grad "
                        "norm runs 12-800, so the old clip_val=1.0 was scaling "
                        "every update down by 1-2 orders of magnitude.")
    p.add_argument("--freq-masks", type=int, default=2)
    p.add_argument("--time-masks", type=int, default=10)
    # batch_duration 600 gives ~50 utterances a step. NeMo's own conformer
    # recipes reach lr ~2e-3 at a global batch of 1,000-2,000 utterances across
    # 16-64 GPUs; at 1/40th of that batch the same lr is 40x too hot, which is
    # what every collapsed run here looks like. Accumulation buys the batch back
    # without needing the GPUs.
    p.add_argument("--accum", type=int, default=1,
                   help="accumulate_grad_batches; effective batch is "
                        "batch_duration * accum")
    # 8x at a 10 ms hop is 12.5 encoder frames/s. diag_batches measures T/U at
    # median 3.04 but p05 1.82, and CTC needs one frame per token plus one per
    # repeat -- the bottom of that distribution has almost no alignment slack,
    # which is where a from-scratch model has to find its first alignment.
    # 4x doubles T for ~2x the encoder cost.
    # parakeet-tdt_ctc-110m, whose config supplies this topology, was trained
    # with 1,024 pieces. A from-scratch CTC head learns every class from random
    # init with no pretrained decoder to inherit.
    p.add_argument("--restore-from", default=None,
                   help="full .nemo to continue from, with a fresh schedule")
    p.add_argument("--init-encoder", default=None,
                   help=".nemo whose encoder weights seed this run (control arm)")
    p.add_argument("--tokenizer-dir", default=TOK)
    p.add_argument("--ctc-reduction", default="mean_batch",
                   choices=["mean_batch", "mean_volume"])
    p.add_argument("--subsampling", type=int, default=8)
    p.add_argument("--precision", default="bf16-mixed",
                   help="Lightning precision. CTC's forward-backward is a "
                        "log-sum-exp over hundreds of frames; 32 is the control "
                        "arm for any suspicion that reduced precision is what "
                        "keeps the model at the blank prior.")
    p.add_argument("--params-only", action="store_true")
    a = p.parse_args()

    if a.max_steps is None:
        aps = (a.audio_per_step or a.batch_duration * 0.85) * a.accum
        a.max_steps = int(a.hours_seen * 3600 / aps)
    cfg = build_cfg(a)

    if a.restore_from:
        # Continue a run with a fresh (short) schedule. Used to anneal a
        # mid-flight checkpoint down to min_lr so it can be compared with arms
        # that finished their cosine: a model measured at 87% of peak lr is not
        # comparable to one measured at 0.2%.
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            a.restore_from, map_location="cpu")
        model.cfg.train_ds, model.cfg.validation_ds = cfg.train_ds, cfg.validation_ds
        model.cfg.optim = cfg.optim
        print(f"[restore] full model from {a.restore_from}")
    else:
        model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg, trainer=None)
    if a.init_encoder:
        # Control arm, not a change of plan: load ONLY the encoder from a trained
        # model and leave the CTC head random. If that descends where random init
        # does not, the training loop, the data path and the loss are all sound
        # and the wall is specific to starting the encoder from noise. Requires
        # --d-model 512 --n-layers 17 --n-heads 8 to match the donor.
        src = nemo_asr.models.ASRModel.restore_from(a.init_encoder, map_location="cpu")
        sd = {k[len("encoder."):]: v for k, v in src.state_dict().items()
              if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
        print(f"[init] encoder from {a.init_encoder}: loaded {len(sd)} tensors, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing or unexpected:
            print(f"[init] missing[:5] {list(missing)[:5]}")
            print(f"[init] unexpected[:5] {list(unexpected)[:5]}")
        del src
    n = sum(x.numel() for x in model.parameters())
    enc = sum(x.numel() for x in model.encoder.parameters())
    print(f"[params] total {n/1e6:.1f}M  encoder {enc/1e6:.1f}M "
          f"(d_model={a.d_model} layers={a.n_layers})")
    print(f"[plan] {a.hours_seen:.0f} h seen -> {a.max_steps} optimizer steps, "
          f"lr {a.lr} warmup {a.warmup}, max_duration {a.max_duration}s, "
          f"batch {a.batch_duration}s x accum {a.accum} "
          f"= {a.batch_duration*a.accum:.0f}s of audio per update")
    # Print the peak LR the scheduler will actually reach. A scheduler that
    # silently rescales the configured value cost one whole pilot run.
    sch = cfg.optim.sched
    if sch.name == "CosineAnnealing":
        peak = a.lr
    elif sch.name == "NoamAnnealing":
        peak = a.lr * (a.d_model ** -0.5) * (a.warmup ** -0.5)
    else:
        peak = float("nan")
    print(f"[lr] scheduler {sch.name}: effective peak lr = {peak:.3e}")
    if peak < a.lr / 10:
        print(f"[lr] WARNING: peak is {a.lr/peak:.0f}x below the configured lr")
    if a.params_only:
        return

    trainer = pl.Trainer(devices=1, accelerator="gpu", max_steps=a.max_steps,
                         max_epochs=-1, precision=a.precision,
                         log_every_n_steps=50, val_check_interval=a.val_every,
                         accumulate_grad_batches=a.accum, gradient_clip_val=a.grad_clip,
                         limit_val_batches=a.limit_val_batches,
                         enable_progress_bar=True,
                         # exp_manager owns both of these and errors if Lightning
                         # made them first.
                         logger=False, enable_checkpointing=False)
    exp_manager(trainer, OmegaConf.create(
        {"exp_dir": a.exp_dir, "name": a.name, "create_checkpoint_callback": True,
         # every_n_epochs must be 0: the run never reaches an epoch boundary, so
         # the default of 1 means no checkpoint is ever written.
         "checkpoint_callback_params": {"save_top_k": 3, "monitor": "val_wer",
                                        "mode": "min", "always_save_nemo": True,
                                        "every_n_epochs": 0,
                                        "every_n_train_steps": a.val_every}}))
    model.set_trainer(trainer)
    model.setup_training_data(cfg.train_ds)
    model.setup_validation_data(cfg.validation_ds)
    trainer.fit(model)
    # Lhotse leaves worker threads alive, so a normal return hangs at interpreter
    # shutdown. The first pilot finished training at 02:29 and then sat there
    # until 07:59 holding the queue open behind it. exp_manager has already
    # written the checkpoints and the .nemo by this point.
    print("[done] training complete, exiting", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

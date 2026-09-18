"""Finetune nemotron-3.5-asr-streaming-0.6b for Nepali (+ English replay).

Baseline measured before training: WER 1.440 / CER 0.878 on our 600-clip Nepali
gold test -- the ne-NP prompt slot exists but was never trained, so this is
teaching a language from scratch, not adapting one.

Tokenizer is left alone on purpose. It tokenises Nepali at 5.12 tok/word (near
character level) vs 2.39 for a proper Indic BPE, which inflates the RNN-T target
length ~2.1x. Paying that is still cheaper than replacing the vocabulary, which
would reinitialise the joint + prediction net and force the English head to be
relearned. Extending the vocab is the v2 lever.

  python train.py --bench          # measure steps/s, print epoch estimate, exit
  python train.py                  # train
"""
import argparse, json, os, time
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager

BASE = ("/workspace/hf_cache/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/"
        "snapshots/ea30d66debe3740a08b573244286791d423d6b3e/"
        "nemotron-3.5-asr-streaming-0.6b.nemo")
M = "/root/tts/TTS_training/pretrain/asr_bilingual/manifests"


def build(a):
    model = nemo_asr.models.ASRModel.restore_from(BASE, map_location="cpu")
    cfg = model.cfg
    with open_dict(cfg):
        # prompt_mode=langID is carried per-row by these manifests.  Without that
        # field the lhotse prompt dataset falls back to default_prompt_mode="unified"
        # and routes 50% of every batch through the language-agnostic `auto` prompt
        # (index 101).  In a 90.6%-Nepali mixture that taught `auto` to mean
        # "Devanagari" and starved the en-US slot (~4.7% of steps), so English audio
        # came back phonetically transliterated into Devanagari: en_test normalised
        # WER 0.109 (base) -> 0.963 at 113k steps.  See logs/eval_en_*.json.
        for ds, man, shuf in (("train_ds", a.train_manifest, True),
                              ("validation_ds", a.val_manifest, False)):
            d = cfg[ds]
            d.manifest_filepath = man
            d.is_tarred = False            # plain files; tar only if reads bottleneck
            d.tarred_audio_filepaths = None
            d.shard_manifests = False
            d.use_lhotse = True
            d.use_bucketing = True
            d.shuffle = shuf
            d.batch_duration = a.batch_duration
            d.num_workers = a.workers
            d.max_duration = 20
            d.min_duration = 0.3
            d.defer_setup = False
        # The pretraining schedule is Noam lr=0.5 warmup=10k, i.e. peak ~1.5e-4.
        # Finetuning gets an explicit cosine at a comparable peak instead, so the
        # LR does not depend on a step count we are not reproducing.
        # fused_batch_size=2 (the shipped value) runs the joint two samples at a
        # time. With V=13,088 that keeps memory down but serialises the step.
        if a.fused_batch: cfg.joint.fused_batch_size = a.fused_batch
        cfg.optim.name = "adamw"
        cfg.optim.lr = a.lr
        cfg.optim.weight_decay = 1e-3
        # max_steps must be explicit: prepare_lr_scheduler otherwise derives it via
        # len(train_dataloader.dataset), which an iterable lhotse dataset lacks.
        cfg.optim.sched = OmegaConf.create(
            {"name": "CosineAnnealing", "warmup_steps": a.warmup, "min_lr": 1e-6,
             "max_steps": a.max_steps})
    model.cfg = cfg
    model.setup_training_data(cfg.train_ds)
    model.setup_validation_data(cfg.validation_ds)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", default=f"{M}/train_mix_langid.jsonl")
    p.add_argument("--val-manifest", default=f"{M}/ne_val_langid.jsonl")
    p.add_argument("--resume-dir", default=None,
                   help="existing exp_manager run dir to resume into, e.g. "
                        "ckpt/nemotron_ne_en/2026-09-12_10-08-33; picks up *-last.ckpt "
                        "from <dir>/checkpoints and keeps writing there")
    p.add_argument("--batch-duration", type=float, default=200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    # 6 epochs ~= 34.5 h. The --bench path badly underestimates throughput: it
    # times 40 steps from a cold start, which are dominated by warprnnt numba JIT
    # compiling each of the 30 bucket shapes plus CUDA graph capture. It reported
    # 0.35 it/s (39x realtime); measured steady state is 3.37 it/s (384x).
    #
    # Lhotse batches by duration, so there is no fixed steps-per-epoch and the
    # dataset has no __len__. The run is driven by max_steps instead, converted
    # from epochs using the benched 114 s of audio per step:
    #   2207.1 h * 3600 / 114 = 69.7k steps/epoch -> 139k for 2 epochs.
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--audio-per-step", type=float, default=114.0)
    p.add_argument("--corpus-hours", type=float, default=2207.1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--val-every", type=int, default=10000)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--bench-steps", type=int, default=40)
    p.add_argument("--fused-batch", type=int, default=None)
    a = p.parse_args()
    if a.max_steps is None:
        a.max_steps = int(a.epochs * a.corpus_hours * 3600 / a.audio_per_step)
    print(f"[plan] {a.epochs} epochs over {a.corpus_hours:.0f} h "
          f"= {a.max_steps} steps; val every {a.val_every}")

    model = build(a)
    if a.bench:
        # training_step touches self._optimizer and self.log, so it only works
        # inside a real Trainer. Benchmark the actual training path rather than a
        # hand-rolled loop that measures something subtly different.
        class Bench(pl.Callback):
            def __init__(s, warmup=5):
                s.warmup, s.t0, s.audio, s.n = warmup, None, 0.0, 0
            def on_train_batch_end(s, tr, pl_module, out, batch, idx):
                if idx == s.warmup:
                    torch.cuda.synchronize(); s.t0 = time.time(); s.audio = 0.0; s.n = 0
                if s.t0 is not None:
                    s.audio += float(batch[1].sum()) / 16000; s.n += 1
        cb = Bench()
        tr = pl.Trainer(devices=1, accelerator="gpu", max_steps=a.bench_steps,
                        precision="bf16-mixed", limit_val_batches=0,
                        num_sanity_val_steps=0, enable_checkpointing=False,
                        logger=False, callbacks=[cb], enable_progress_bar=False)
        model.set_trainer(tr)
        tr.fit(model)
        torch.cuda.synchronize()
        el = time.time() - cb.t0
        rt = cb.audio / el
        h_total = 2207.1
        print(f"\n== bench: {cb.n} steps in {el:.1f}s = {cb.n/el:.2f} it/s")
        print(f"   audio/step {cb.audio/max(cb.n,1):.0f}s   throughput {rt:.0f}x realtime")
        print(f"   peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        print(f"   EPOCH over {h_total:.0f} h: {h_total*3600/rt/3600:.2f} h")
        return

    # exp_manager owns both the logger and the checkpoint callback, and errors out
    # if Lightning already made either. Both defaults must be turned off here.
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_steps=a.max_steps,
                         max_epochs=-1,
                         precision="bf16-mixed", log_every_n_steps=50,
                         val_check_interval=a.val_every, accumulate_grad_batches=1,
                         gradient_clip_val=1.0, enable_progress_bar=True,
                         logger=False, enable_checkpointing=False)
    em = {"exp_dir": "/root/tts/TTS_training/pretrain/asr_bilingual/ckpt", "name": "nemotron_ne_en",
          "create_checkpoint_callback": True}
    if a.resume_dir:
        # explicit_log_dir overrides exp_dir/name/version, so the resumed run keeps
        # writing into the original timestamped dir instead of opening a new one.
        # resume_if_exists then sets trainer ckpt_path from <dir>/checkpoints/*-last.ckpt,
        # which restores global_step, optimizer state and the cosine schedule position.
        em["explicit_log_dir"] = a.resume_dir
        em["resume_if_exists"] = True
        em["resume_ignore_no_checkpoint"] = False
    exp_manager(trainer, OmegaConf.create(
        {**em,
         # every_n_epochs must go to 0: the run never reaches an epoch boundary, so
         # the default of 1 would mean no checkpoint is ever written. Saving on the
         # same cadence as validation keeps the monitored val_wer fresh.
         "checkpoint_callback_params": {"save_top_k": 3, "monitor": "val_wer",
                                        "mode": "min", "always_save_nemo": True,
                                        "every_n_epochs": 0,
                                        "every_n_train_steps": a.val_every}}))
    model.set_trainer(trainer)
    trainer.fit(model)


if __name__ == "__main__":
    main()

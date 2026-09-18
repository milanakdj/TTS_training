"""What the training loop is actually fed: target lengths, text, frames per batch.

Five runs have now plateaued at val_wer ~1.0 with a flat CTC loss, so this walks
the real dataloader (same input_cfg, bins and caps as run_v5) and reports what
reaches the loss -- an empty or near-empty transcript is free loss and drags the
model to all-blank, which is exactly the observed failure.
"""
import argparse, sys, os, collections
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_train
import nemo.collections.asr as nemo_asr

p = argparse.ArgumentParser()
M = "/workspace/asr_pretrain_v2/manifests"
p.add_argument("--train-manifest", default=f"{M}/train_mix.jsonl")
p.add_argument("--val-manifest", default=f"{M}/val_mix.jsonl")
p.add_argument("--input-cfg", default=f"{M}/train_input_cfg.yaml")
p.add_argument("--bucket-bins", default=f"{M}/bucket_duration_bins.json")
p.add_argument("--tarred", default=None)
p.add_argument("--batch-duration", type=float, default=600)
p.add_argument("--max-duration", type=float, default=20)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--batches", type=int, default=30)
p.add_argument("--d-model", type=int, default=256)
p.add_argument("--n-layers", type=int, default=16)
p.add_argument("--n-heads", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=15000)
p.add_argument("--max-steps", type=int, default=100000)
p.add_argument("--freq-masks", type=int, default=2)
p.add_argument("--time-masks", type=int, default=10)
a = p.parse_args()

cfg = pilot_train.build_cfg(a)
model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg, trainer=None)
model.setup_training_data(cfg.train_ds)
dl = model._train_dl

n_utt = n_empty = 0
tot_audio = 0.0
tlens, ulens = [], []
short = []
for i, batch in enumerate(dl):
    if i >= a.batches:
        break
    sig, sig_len, tgt, tgt_len = batch[0], batch[1], batch[2], batch[3]
    tot_audio += float(sig_len.sum()) / 16000
    for j in range(sig_len.shape[0]):
        n_utt += 1
        u = int(tgt_len[j])
        t = int(sig_len[j]) / 16000 * 100 / 8      # encoder frames, 8x subsampling
        ulens.append(u); tlens.append(t)
        if u == 0:
            n_empty += 1
        elif t / u < 1.0:
            short.append((t, u))
print(f"\n[batches] {a.batches}  utts {n_utt}  audio {tot_audio:.0f}s "
      f"({tot_audio/a.batches:.0f}s per batch, {n_utt/a.batches:.1f} utts per batch)")
print(f"[targets] empty {n_empty} ({100*n_empty/max(n_utt,1):.2f}%)  "
      f"T<U {len(short)} ({100*len(short)/max(n_utt,1):.2f}%)")
ul = sorted(ulens); tl = sorted(tlens)
q = lambda v, f: v[int(f*(len(v)-1))]
print(f"[U tokens] p05 {q(ul,.05)}  med {q(ul,.5)}  p95 {q(ul,.95)}")
print(f"[T frames] p05 {q(tl,.05):.0f}  med {q(tl,.5):.0f}  p95 {q(tl,.95):.0f}")
r = sorted(t/u for t, u in zip(tlens, ulens) if u)
print(f"[T/U]      p05 {q(r,.05):.2f}  med {q(r,.5):.2f}")

# decode a few targets back through the tokenizer: the text the loss is fitting
b = next(iter(dl))
tgt, tgt_len = b[2], b[3]
print("\n[targets decoded] first 3 rows of a fresh batch:")
for j in range(min(3, tgt.shape[0])):
    ids = tgt[j, :int(tgt_len[j])].tolist()
    print(f"  U={len(ids):3d}  {model.tokenizer.ids_to_text(ids)[:90]!r}")

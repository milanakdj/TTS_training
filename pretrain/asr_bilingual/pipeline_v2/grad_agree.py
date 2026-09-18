"""Do two different batches want the same thing?

The bisect leaves two live explanations for a loss that never moves on real data
while 64 fixed utterances memorise fine:

  (a) the corpus gives no consistent gradient -- batches pull in unrelated
      directions and cancel, in which case no learning rate rescues it;
  (b) the corpus is fine and the update size is wasting the signal.

Those are distinguishable without training: take K batches, compute the full
gradient of each, and look at the pairwise cosine similarity. Independent batches
drawn from a learnable corpus agree weakly but *positively* (typically 0.02-0.2
early in training); a corpus with no learnable signal sits at zero within the
noise floor, which for a d-dimensional random vector is ~1/sqrt(d) = 2e-4 here.

Reported per parameter group as well, because a gradient that agrees in the
decoder but not the encoder means something different from one that agrees
nowhere.
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_train
import nemo.collections.asr as nemo_asr

p = argparse.ArgumentParser()
M = "/workspace/asr_pretrain_v2/manifests"
p.add_argument("--train-manifest", default=f"{M}/train_mix.jsonl")
p.add_argument("--val-manifest", default=f"{M}/probe_val.jsonl")
p.add_argument("--input-cfg", default=None)
p.add_argument("--bucket-bins", default=f"{M}/bucket_duration_bins.json")
p.add_argument("--restore", default=None, help=".nemo to load instead of random init")
p.add_argument("--tarred", default=None)
p.add_argument("--batches", type=int, default=6)
p.add_argument("--batch-duration", type=float, default=240)
p.add_argument("--max-duration", type=float, default=20)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--d-model", type=int, default=256)
p.add_argument("--n-layers", type=int, default=16)
p.add_argument("--n-heads", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--max-steps", type=int, default=1000)
p.add_argument("--freq-masks", type=int, default=0)
p.add_argument("--time-masks", type=int, default=0)
a = p.parse_args()

cfg = pilot_train.build_cfg(a)
if a.restore:
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(a.restore, map_location="cuda")
    model.cfg.train_ds = cfg.train_ds
else:
    model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg, trainer=None)
model = model.cuda().train()
model.setup_training_data(cfg.train_ds)
dl = model._train_dl

groups = {"encoder.pre_encode": [], "encoder.layers": [], "decoder": []}
for n, _ in model.named_parameters():
    for g in groups:
        if n.startswith(g):
            groups[g].append(n)
            break

grads = []
for i, batch in enumerate(dl):
    if i >= a.batches:
        break
    sig, sl, tgt, tl = (x.cuda() for x in batch[:4])
    model.zero_grad(set_to_none=True)
    logp, enc_len, _ = model(input_signal=sig, input_signal_length=sl)
    loss = model.loss(log_probs=logp, targets=tgt, input_lengths=enc_len, target_lengths=tl)
    loss.backward()
    g = {n: p.grad.detach().float().flatten().clone()
         for n, p in model.named_parameters() if p.grad is not None}
    grads.append(g)
    print(f"[batch {i}] utts {sig.shape[0]:4d}  loss {float(loss):8.1f}  "
          f"|g| {torch.cat(list(g.values())).norm():8.2f}", flush=True)

def cos(ga, gb, names):
    x = torch.cat([ga[n] for n in names]); y = torch.cat([gb[n] for n in names])
    return float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-12))

allnames = list(grads[0].keys())
d = sum(grads[0][n].numel() for n in allnames)
print(f"\n{d} params; noise floor for unrelated gradients ~{d**-0.5:.2e}")
print(f"\n{'pair':10} {'all':>9} " + " ".join(f"{g.split('.')[-1]:>12}" for g in groups))
vals = []
for i in range(len(grads)):
    for j in range(i + 1, len(grads)):
        c = cos(grads[i], grads[j], allnames)
        vals.append(c)
        print(f"{i}-{j:<8} {c:9.4f} " +
              " ".join(f"{cos(grads[i], grads[j], groups[g]):12.4f}" for g in groups))
import statistics
print(f"\nmean pairwise cosine over {len(vals)} pairs: {statistics.mean(vals):.4f}")

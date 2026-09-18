"""Is the collapsed model at a real minimum, or is the optimiser failing to move?

At the blank solution the loss is flat for 197,299 steps while gradients from
independent batches still agree at cosine 0.93. Those two facts are only
compatible in two ways, and a line search separates them without training:

  * average the gradient over several batches, then evaluate the loss at
    w - eta*g for a range of eta on *different* batches;
  * if some eta reduces the loss materially, descent is available and the
    optimiser (lr, Adam state, clipping) is what is failing to take it;
  * if no eta helps, the all-blank point is a genuine attractor for this
    model/recipe and no learning-rate search will ever escape it.

Losses are reported per target token so the batches are comparable.
"""
import argparse, os, sys, copy
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_train
import nemo.collections.asr as nemo_asr

p = argparse.ArgumentParser()
M = "/workspace/asr_pretrain_v2/manifests"
p.add_argument("--restore", default=None, help=".nemo to probe; random init if unset")
p.add_argument("--train-manifest", default=f"{M}/train_mix.jsonl")
p.add_argument("--val-manifest", default=f"{M}/probe_val.jsonl")
p.add_argument("--input-cfg", default=None)
p.add_argument("--bucket-bins", default=f"{M}/bucket_duration_bins.json")
p.add_argument("--tarred", default=None)
p.add_argument("--grad-batches", type=int, default=4)
p.add_argument("--eval-batches", type=int, default=3)
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

batches = []
for i, b in enumerate(model._train_dl):
    if i >= a.grad_batches + a.eval_batches:
        break
    batches.append(tuple(x.cuda() for x in b[:4]))
gb, eb = batches[:a.grad_batches], batches[a.grad_batches:]

def loss_per_token(bs):
    tot = tok = 0.0
    with torch.no_grad():
        for sig, sl, tgt, tl in bs:
            logp, elen, _ = model(input_signal=sig, input_signal_length=sl)
            l = model.loss(log_probs=logp, targets=tgt, input_lengths=elen, target_lengths=tl)
            tot += float(l) * sig.shape[0]
            tok += float(tl.sum())
    return tot / tok

# mean gradient over the gradient batches
model.zero_grad(set_to_none=True)
for sig, sl, tgt, tl in gb:
    logp, elen, _ = model(input_signal=sig, input_signal_length=sl)
    l = model.loss(log_probs=logp, targets=tgt, input_lengths=elen, target_lengths=tl)
    (l / len(gb)).backward()
g = {n: q.grad.detach().clone() for n, q in model.named_parameters() if q.grad is not None}
gnorm = torch.cat([v.flatten() for v in g.values()]).norm()
w0 = {n: q.detach().clone() for n, q in model.named_parameters()}

base_tr, base_ev = loss_per_token(gb), loss_per_token(eb)
print(f"|grad| {gnorm:.2f}   loss/token: grad-batches {base_tr:.4f}  eval-batches {base_ev:.4f}")
print(f"\n{'eta':>10} {'|step|':>10} {'grad-batch':>12} {'eval-batch':>12}")
for eta in (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    with torch.no_grad():
        for n, q in model.named_parameters():
            if n in g:
                q.copy_(w0[n] - eta * g[n])
    tr, ev = loss_per_token(gb), loss_per_token(eb)
    print(f"{eta:10.1e} {eta*float(gnorm):10.4f} {tr:12.4f} {ev:12.4f}"
          + ("   <-- descends" if ev < base_ev - 1e-3 else ""))
with torch.no_grad():
    for n, q in model.named_parameters():
        q.copy_(w0[n])

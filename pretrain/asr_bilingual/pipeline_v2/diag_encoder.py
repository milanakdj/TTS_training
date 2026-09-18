"""Does the encoder actually use the audio, or is the memorise test a fake pass?

mem64 drove 64 utterances to 0.45 loss/token and was read as proof that the model
and loss are wired correctly. That reading has a hole: 64 clips of 4-9 s can be
told apart by duration and gross energy alone, so a model that ignores the audio
content entirely can still emit the right transcript for each one. This checks
the three things that distinguish those cases:

  1. loss/token on the training set (the number the memorise test reported);
  2. loss/token with the audio rotated by one against the text -- if the encoder
     is really reading the audio this must blow up to the prior, and if it stays
     low the model is keying on something other than the speech;
  3. pairwise cosine between mean-pooled encoder outputs for different clips --
     near 1.0 means the encoder emits the same representation whatever it is fed,
     which is the collapse that would make CTC hopeless no matter the recipe.
"""
import argparse, json
import torch, soundfile as sf, librosa, numpy as np
import nemo.collections.asr as nemo_asr

p = argparse.ArgumentParser()
p.add_argument("nemo")
p.add_argument("--manifest", default="/workspace/asr_pretrain_v2/manifests/mem64.jsonl")
p.add_argument("--n", type=int, default=64)
p.add_argument("--batch", type=int, default=8)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.manifest)][:a.n]
m = nemo_asr.models.EncDecCTCModelBPE.restore_from(a.nemo, map_location="cuda").eval().cuda()

def audio(r):
    x, sr = sf.read(r["audio_filepath"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x

wavs = [audio(r) for r in rows]

def score(texts):
    tot_loss = tot_tok = 0.0
    with torch.no_grad():
        for i in range(0, len(rows), a.batch):
            ws, ts = wavs[i:i + a.batch], texts[i:i + a.batch]
            L = max(len(w) for w in ws)
            sig = torch.zeros(len(ws), L)
            for j, w in enumerate(ws):
                sig[j, :len(w)] = torch.from_numpy(w)
            sl = torch.tensor([len(w) for w in ws], device="cuda")
            ids = [m.tokenizer.text_to_ids(t) for t in ts]
            U = max(len(t) for t in ids)
            tt = torch.zeros(len(ids), U, dtype=torch.long, device="cuda")
            for j, t in enumerate(ids):
                tt[j, :len(t)] = torch.tensor(t, device="cuda")
            tl = torch.tensor([len(t) for t in ids], device="cuda")
            logp, elen, _ = m(input_signal=sig.cuda(), input_signal_length=sl)
            loss = m.loss(log_probs=logp, targets=tt, input_lengths=elen, target_lengths=tl)
            tot_loss += float(loss) * len(ws)
            tot_tok += float(tl.sum())
    return tot_loss / tot_tok

texts = [r["text"] for r in rows]
print(f"loss/token, audio paired with its own text : {score(texts):7.3f}")
print(f"loss/token, audio rotated against the text : {score(texts[1:] + texts[:1]):7.3f}")
print("   (prior for a 4,000-piece vocabulary is ln 4001 = 8.294)")

embs = []
with torch.no_grad():
    for w in wavs[:16]:
        sig = torch.from_numpy(w)[None].cuda()
        sl = torch.tensor([len(w)], device="cuda")
        proc, plen = m.preprocessor(input_signal=sig, length=sl)
        enc, elen = m.encoder(audio_signal=proc, length=plen)
        embs.append(enc[0, :, :int(elen[0])].mean(-1).float())
E = torch.stack(embs)
E = E / E.norm(dim=-1, keepdim=True)
C = (E @ E.T)
off = C[~torch.eye(len(E), dtype=bool, device=C.device)]
print(f"\nencoder mean-pooled cosine between different clips: "
      f"mean {off.mean():.4f}  min {off.min():.4f}  max {off.max():.4f}")
print(f"per-frame std of encoder output within a clip: "
      f"{torch.stack([e.std() for e in embs]).mean():.4f}")

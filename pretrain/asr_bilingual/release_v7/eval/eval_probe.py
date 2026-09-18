"""Fixed-set scoring for a .nemo checkpoint: per-token CTC loss, WER, blank rate.

The logged train_loss is not comparable across steps -- lhotse buckets vary the
token count per batch by an order of magnitude, so the scalar swings 40-1300 on a
model that is doing nothing. Everything here is measured on one fixed set of
utterances and normalised per target token, so two checkpoints can actually be
compared, and a random-init model is scored alongside as the do-nothing baseline.
"""
import argparse, json, math, sys
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate

p = argparse.ArgumentParser()
p.add_argument("nemo", nargs="+")
p.add_argument("--manifest", default="/workspace/asr_pretrain_v2/manifests/probe_val.jsonl")
p.add_argument("--n", type=int, default=200)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--device", default="cuda")
a = p.parse_args()

rows = [json.loads(l) for l in open(a.manifest)][:a.n]
refs = [r["text"] for r in rows]
paths = [r["audio_filepath"] for r in rows]

print(f"{'checkpoint':34} {'loss/token':>10} {'WER':>7} {'blank%':>7} {'emptyhyp':>9}")
for path in a.nemo:
    m = nemo_asr.models.EncDecCTCModelBPE.restore_from(path, map_location=a.device)
    m.eval().to(a.device)
    tot_loss = tot_tok = 0.0
    blank_frames = all_frames = 0
    with torch.no_grad():
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            sigs, lens = [], []
            import soundfile as sf, librosa, numpy as np
            for r in chunk:
                x, sr = sf.read(r["audio_filepath"], dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != 16000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                sigs.append(torch.from_numpy(x)); lens.append(len(x))
            L = max(lens)
            sig = torch.zeros(len(sigs), L)
            for j, s in enumerate(sigs):
                sig[j, :len(s)] = s
            sig, sl = sig.to(a.device), torch.tensor(lens, device=a.device)
            tgt = [m.tokenizer.text_to_ids(r["text"]) for r in chunk]
            U = max(len(t) for t in tgt)
            tt = torch.zeros(len(tgt), U, dtype=torch.long, device=a.device)
            for j, t in enumerate(tgt):
                tt[j, :len(t)] = torch.tensor(t, device=a.device)
            tl = torch.tensor([len(t) for t in tgt], device=a.device)
            logp, enc_len, _ = m(input_signal=sig, input_signal_length=sl)
            loss = m.loss(log_probs=logp, targets=tt, input_lengths=enc_len,
                          target_lengths=tl)
            # m.loss reduces over the batch; recover the total to normalise by tokens
            tot_loss += float(loss) * len(chunk)
            tot_tok += float(tl.sum())
            pred = logp.argmax(-1)
            blank = m.decoder.num_classes_with_blank - 1
            for j in range(len(chunk)):
                n = int(enc_len[j])
                all_frames += n
                blank_frames += int((pred[j, :n] == blank).sum())
    hyps = m.transcribe(paths, batch_size=a.batch, verbose=False)
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]
    wer = word_error_rate(hypotheses=hyps, references=refs)
    empty = sum(1 for h in hyps if not h.strip())
    name = path.split("/ckpt")[-1][-33:]
    print(f"{name:34} {tot_loss/tot_tok:10.3f} {wer:7.3f} "
          f"{100*blank_frames/max(all_frames,1):7.2f} {empty:6d}/{len(hyps)}")
    del m
    torch.cuda.empty_cache()

"""Throughput of millions-scale FastConformer-CTC on this H100, bf16.

Measures the real fwd+bwd+step path (GPU mel -> encoder -> CTC) on synthetic
audio so the number is compute-only; the lhotse dataloader already benched at
7248x realtime (logs/bench_dl_w8.log) so it is not the wall at these sizes.
"""
import time, torch, torch.nn as nn
from nemo.collections.asr.modules import ConformerEncoder, ConvASRDecoder
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.losses.ctc import CTCLoss

SR, VOCAB = 16000, 4000
CFGS = [
    ("nano  d128 L 8 b256", 128,  8, 4, 256),
    ("nano  d128 L 8 b512", 128,  8, 4, 512),
    ("tiny  d176 L 8 b512", 176,  8, 4, 512),
    ("small d176 L16 b384", 176, 16, 4, 384),
]

def build(d, L, H):
    pre = AudioToMelSpectrogramPreprocessor(features=80, window_size=0.025,
                                            window_stride=0.01, normalize="per_feature").cuda()
    enc = ConformerEncoder(feat_in=80, n_layers=L, d_model=d, n_heads=H,
                           ff_expansion_factor=4, self_attention_model="rel_pos",
                           conv_kernel_size=9, subsampling="dw_striding",
                           subsampling_factor=8, subsampling_conv_channels=256,
                           pos_emb_max_len=5000).cuda()
    dec = ConvASRDecoder(feat_in=d, num_classes=VOCAB).cuda()
    return pre, enc, dec

def run(name, d, L, H, bsz=64, sec=15, steps=25, warm=8):
    pre, enc, dec = build(d, L, H)
    params = sum(p.numel() for m in (enc, dec) for p in m.parameters())
    loss_fn = CTCLoss(num_classes=VOCAB - 1, zero_infinity=True).cuda()
    opt = torch.optim.AdamW([p for m in (enc, dec) for p in m.parameters()], lr=1e-3)
    audio = torch.randn(bsz, SR * sec, device="cuda")
    alen = torch.full((bsz,), SR * sec, device="cuda", dtype=torch.long)
    # ~2.4 BPE tok/s of Nepali speech at 4k vocab -> 36 targets for 15 s
    tgt = torch.randint(1, VOCAB - 1, (bsz, 36), device="cuda")
    tlen = torch.full((bsz,), 36, device="cuda", dtype=torch.long)
    t0 = None
    for i in range(steps + warm):
        if i == warm:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f, flen = pre(input_signal=audio, length=alen)
            e, elen = enc(audio_signal=f, length=flen)
            lp = dec(encoder_output=e)
            loss = loss_fn(log_probs=lp, targets=tgt,
                           input_lengths=elen, target_lengths=tlen)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    el = time.time() - t0
    rt = steps * bsz * sec / el
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"{name}  {params/1e6:6.1f}M  {steps/el:5.2f} it/s  {rt:7.0f}x RT  "
          f"{mem:4.1f}GB  epoch(2500h)={2500/ (rt/3600):8.0f}s = {2500*3600/rt/3600:5.2f} h")
    del pre, enc, dec, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    for c in CFGS:
        try: run(*c)
        except Exception as ex: print(c[0], "FAILED", type(ex).__name__, str(ex)[:200])

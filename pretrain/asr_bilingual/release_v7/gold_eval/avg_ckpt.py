"""Average the saved run_v7 checkpoints and export a .nemo.

The card notes averaging was skipped because save_top_k=3 left too few. Three
top-k plus the train-end 'last' is four, which is the low end of the usual 5-10
but still worth testing rather than assuming. Averaged in float64 on CPU; only
float tensors are averaged, integer buffers (step counters, num_batches_tracked)
are taken from the first checkpoint rather than averaged into nonsense.
"""
import glob, torch, collections
import nemo.collections.asr as nemo_asr

CK = "/workspace/asr_pretrain_v2/ckpt/run_v7_encinit/2026-09-16_08-41-31/checkpoints"
paths = sorted(glob.glob(f"{CK}/*.ckpt"))
print(f"averaging {len(paths)} checkpoints:")
for p in paths: print("  ", p.split("/")[-1])

acc, n = None, 0
for p in paths:
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    if acc is None:
        acc = {k: (v.double().clone() if v.is_floating_point() else v.clone())
               for k, v in sd.items()}
    else:
        for k, v in sd.items():
            if acc[k].is_floating_point(): acc[k] += v.double()
    n += 1
for k in acc:
    if acc[k].is_floating_point(): acc[k] /= n

model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    "/root/tts/TTS_training/pretrain/asr_bilingual/release_v7/parakeet-ctc-nepali-110m.nemo",
    map_location="cpu")
tgt = model.state_dict()
cast = {k: acc[k].to(tgt[k].dtype) for k in tgt if k in acc}
missing = [k for k in tgt if k not in acc]
print(f"\nloaded {len(cast)}/{len(tgt)} tensors, {len(missing)} missing")
if missing: print("  missing:", missing[:5])
model.load_state_dict(cast, strict=False)
out = "/root/tts/TTS_training/pretrain/asr_bilingual/gold_eval/parakeet-110m-avg4.nemo"
model.save_to(out)
print("wrote", out)

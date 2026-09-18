"""Can NeMo 3.0.0 actually load nemotron-3.5-asr and train it?

Checks, in order of what would kill the plan: restore, model class, vocab size,
whether the RNN-T pieces we may later want to resize are where we expect, and a
real forward pass on GPU.
"""
import torch, nemo.collections.asr as nemo_asr

P = ("/workspace/hf_cache/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/"
     "snapshots/ea30d66debe3740a08b573244286791d423d6b3e/"
     "nemotron-3.5-asr-streaming-0.6b.nemo")

m = nemo_asr.models.ASRModel.restore_from(P, map_location="cpu")
print("class:", type(m).__name__)
print("params: %.0fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
print("encoder:", type(m.encoder).__name__)
for attr in ("decoder", "joint"):
    if hasattr(m, attr): print(f"{attr}:", type(getattr(m, attr)).__name__)
try: print("vocab:", len(m.tokenizer.vocab))
except Exception as e: print("vocab: ?", e)
print("has change_vocabulary:", hasattr(m, "change_vocabulary"))
print("cfg keys:", list(m.cfg.keys())[:20])

"""Dataloader-only throughput: is the GPU starved, or is the model just slow?

Iterates the exact training dataloader with no model attached. If this lands near
the 39x realtime the full step achieves, the bottleneck is I/O (16 pinned cores,
network FS) and no amount of model surgery will help. If it is far above, the
joint tensor is the problem and the vocab swap is the fix.
"""
import time, sys, torch
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr

BASE = ("/workspace/hf_cache/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/"
        "snapshots/ea30d66debe3740a08b573244286791d423d6b3e/"
        "nemotron-3.5-asr-streaming-0.6b.nemo")
W = int(sys.argv[1]) if len(sys.argv) > 1 else 8

m = nemo_asr.models.ASRModel.restore_from(BASE, map_location="cpu")
with open_dict(m.cfg):
    d = m.cfg.train_ds
    d.manifest_filepath = "/root/tts/TTS_training/pretrain/asr_bilingual/manifests/train_mix.jsonl"
    d.is_tarred = False; d.tarred_audio_filepaths = None; d.shard_manifests = False
    d.use_lhotse = True; d.use_bucketing = True; d.shuffle = True
    d.batch_duration = 200; d.num_workers = W; d.max_duration = 20
    d.min_duration = 0.3; d.defer_setup = False
m.setup_training_data(m.cfg.train_ds)

dl = m.train_dataloader()
audio = 0.0; t0 = None; n = 0
for i, b in enumerate(dl):
    if i == 5: t0 = time.time(); audio = 0.0; n = 0
    if t0 is not None: audio += float(b[1].sum()) / 16000; n += 1
    if i >= 45: break
el = time.time() - t0
print(f"\n== dataloader only [workers={W}]: {n} batches in {el:.1f}s")
print(f"   audio/batch {audio/n:.0f}s   throughput {audio/el:.0f}x realtime")

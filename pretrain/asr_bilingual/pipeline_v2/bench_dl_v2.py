"""Step 1: the dataloader ceiling. The doc says this has never been run and that
it decides whether the plan is 3 weeks or 3 months.

The schedule is denominated in audio hours seen, so the dataloader feeds the same
bytes/s whether the model is 600M or 32M. If this lands near the model's step
rate, the run is I/O bound and shrinking the model buys nothing.

No model is attached -- this is the pilot's own dataloader config, built from the
same template, so the number transfers directly to the pilot.
"""
import argparse, json, os, sys, time
from omegaconf import OmegaConf
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from lhotse.dataset import AudioSamples

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="/workspace/asr_pretrain_v2/manifests/train_mix.jsonl")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--batch-duration", type=float, default=600)
ap.add_argument("--max-duration", type=float, default=60)
ap.add_argument("--tarred", default=None)
ap.add_argument("--bucket-bins",
                default="/workspace/asr_pretrain_v2/manifests/bucket_duration_bins.json",
                help="without these the sampler estimates bins by scanning the "
                     "sources, which can stall the whole queue")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--batches", type=int, default=120,
                help="measure past the warmup; a 40-step window from cold is "
                     "what made the last bench underestimate by 10x")
a = ap.parse_args()

cfg = OmegaConf.create({
    "manifest_filepath": a.manifest, "use_lhotse": True, "use_bucketing": True,
    "num_buckets": 30, "batch_duration": a.batch_duration, "shuffle": True,
    "num_workers": a.workers, "max_duration": a.max_duration, "min_duration": 0.3,
    "is_tarred": bool(a.tarred), "tarred_audio_filepaths": a.tarred,
    "shard_manifests": False, "sample_rate": 16000, "shuffle_n": 2048,
})
if a.bucket_bins and os.path.exists(a.bucket_bins):
    cfg.bucket_duration_bins = json.load(open(a.bucket_bins))

class _Load:
    """Actually decode the audio.

    A passthrough dataset that just returns the CutSet never touches the files --
    it measures the sampler and reports ~10,000x realtime, which is meaningless
    for a question about I/O. AudioSamples is what NeMo's own dataset uses, so
    this times the same reads the real training loop performs.
    """
    def __init__(self): self.load = AudioSamples(fault_tolerant=True)
    def __getitem__(self, cuts):
        audio, audio_lens, cuts = self.load(cuts)
        return audio, audio_lens, cuts

dl = get_lhotse_dataloader_from_config(cfg, global_rank=0, world_size=1,
                                       dataset=_Load())
audio, t0, n = 0.0, None, 0
for i, (_wav, wav_lens, _cuts) in enumerate(dl):
    if i == a.warmup:
        t0 = time.time(); audio = 0.0; n = 0
    if t0 is not None:
        audio += float(wav_lens.sum()) / 16000; n += 1
    if i >= a.batches: break
el = time.time() - t0
print(f"\n== dataloader only [workers={a.workers}, tarred={bool(a.tarred)}]: "
      f"{n} batches in {el:.1f}s")
print(f"   audio/batch {audio/max(n,1):.0f}s   throughput {audio/el:.0f}x realtime")
print(f"   => one pass over 4,500 h would take {4500*3600/(audio/el)/3600:.1f} h")
# Breaking out of a lhotse dataloader leaves its worker threads alive, so a normal
# return hangs at interpreter shutdown. As a queue stage that means the driver
# blocks forever *after* printing its result. Leave immediately instead.
sys.stdout.flush()
os._exit(0)

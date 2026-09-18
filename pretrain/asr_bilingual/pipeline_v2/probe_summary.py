"""Loss trajectory per ablation arm, straight from the tfevents."""
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/asr_pretrain_v2/ckpt_probe"
print(f"{'arm':14} {'steps':>6} " + " ".join(f"{p:>7}" for p in
      ("s50", "s300", "s600", "s1000", "s1500", "s2000", "s2500", "s2999")) + "   val_wer")
for arm in sorted(os.listdir(root)):
    for ev in sorted(glob.glob(os.path.join(root, arm, "*", "events.out.tfevents.*"))):
        ea = EventAccumulator(ev, size_guidance={'scalars': 100000}); ea.Reload()
        t = ea.Tags()['scalars']
        if 'train_loss' not in t:
            continue
        tl = {x.step: x.value for x in ea.Scalars('train_loss')}
        ks = sorted(tl)
        def near(s):
            c = min(ks, key=lambda k: abs(k - s))
            return f"{tl[c]:7.1f}" if abs(c - s) < 200 else "      -"
        vw = [f"{x.value:.4f}" for x in ea.Scalars('val_wer')] if 'val_wer' in t else []
        print(f"{arm:14} {max(ks):>6} " + " ".join(near(s) for s in
              (50, 300, 600, 1000, 1500, 2000, 2500, 2999)) + "   " + " ".join(vw))

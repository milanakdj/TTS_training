#!/usr/bin/env bash
# One serial driver for the remaining probes, most informative first.
#
# initenc and arch110m are a matched pair: identical 110M topology, differing
# only in whether the encoder starts from parakeet's trained weights or from
# noise. That pair answers whether this training loop works at all, so it runs
# before the arms that only explore hyperparameters.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until grep -q "ladder done" /workspace/asr_pretrain_v2/logs/probe_ladder.log 2>/dev/null; do sleep 30; done
bash "$HERE/probe_initenc.sh"
bash "$HERE/probe_arch.sh"
echo "[queue] all probes done"

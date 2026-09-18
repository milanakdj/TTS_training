#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until grep -q "arch+vocab probe done" /workspace/asr_pretrain_v2/logs/probe_arch.log 2>/dev/null; do sleep 30; done
exec bash "$HERE/probe_batch3.sh"

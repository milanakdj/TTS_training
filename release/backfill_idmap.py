#!/usr/bin/env python3
"""Rebuild id_map entries that pack_reserve.py's tail drain failed to write.

The packer wrote id_map inside the batch loop but not in the final partial drain,
so the last <=16 rows of each finished source went to the Hub with no way back from
their opaque id to a source path. `id` is h16(path#offset) -- a pure function of the
manifest row -- so the mapping is fully recoverable without touching audio.

This replays the packer's filter chain in the same source order (archive exclusion,
then cross-source chunk dedupe) so the ids it derives are the ids the packer used,
then emits only the ones missing from the existing map.

Output goes to a SEPARATE file so it cannot interleave with the running packer's
appends. Merge when the run is done.

  python3 backfill_idmap.py --sources r3,r4,r2
"""
import argparse, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_reserve import SOURCES, ROOT, NE, resolve, archived_paths, h16, NEW_IDMAP

OUT = "/root/tts/TTS_training/release/id_map_reserve_backfill.jsonl"

def main(a):
    want = set(a.sources.split(",")) if a.sources else {c for c, *_ in SOURCES}
    have = set()
    if os.path.isfile(NEW_IDMAP):
        for l in open(NEW_IDMAP, errors="replace"):
            try: have.add(json.loads(l)["id"])
            except Exception: pass
    print(f"[init] existing id_map entries: {len(have):,}")
    arch = archived_paths()
    seen = set()          # replays the packer's cross-source chunk dedupe
    out = open(OUT, "w")
    total_new = 0
    # The dedupe replay only needs sources at or before the last wanted one, since
    # `seen` is populated in SOURCES order. Stopping there avoids stat-ing 86k r6
    # paths on weka just to backfill an earlier source.
    last = max(i for i, (c, *_) in enumerate(SOURCES) if c in want)
    for idx, (code, dirname, mf, _h) in enumerate(SOURCES):
        if idx > last: break
        path = f"{ROOT}/{dirname}/{mf}"
        if not os.path.isfile(path): continue
        cand = new = 0
        for line in open(path, errors="replace"):
            if not NE.search(line): continue
            try: d = json.loads(line)
            except Exception: continue
            src = resolve(d.get("audio_filepath"))
            if not src: continue
            text = d.get("text") or d.get("normalised_text") or d.get("unnormalised_text")
            if not text or not str(text).strip(): continue
            if float(d.get("duration") or 0) <= 0: continue
            off = d.get("offset") if d.get("offset") is not None else d.get("chunk_start_s")
            uid = h16(f"{src}#{off if off is not None else 0}")
            chunk_id = d.get("id") or d.get("key") or d.get("sample_id")
            chunk_id = str(chunk_id) if chunk_id else None
            if src in arch: continue
            if uid in seen or (chunk_id and chunk_id in seen): continue
            seen.add(uid)
            if chunk_id: seen.add(chunk_id)
            cand += 1
            if code in want and uid not in have:
                out.write(json.dumps({"id": uid, "src": code, "path": src,
                                      "off": off, "backfilled": True}) + "\n")
                new += 1
        if code in want:
            print(f"  {code}: {cand:,} candidate rows, {new:,} missing from id_map")
            total_new += new
    out.close()
    print(f"[done] wrote {total_new:,} backfill entries -> {OUT}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--sources")
    main(p.parse_args())

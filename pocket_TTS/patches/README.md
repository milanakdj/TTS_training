# Our changes to the upstream pocket-tts checkout

`pocket_TTS/repo/` is Kyutai's `kyutai-labs/pocket-tts` checkout. It is git-ignored,
so the edits we made to its training code only existed on this box until now.
This patch is `git diff` of that checkout against the commit in `UPSTREAM_BASE`.

It adds:
- LwF: \`lwf_weight\` times MSE against a frozen, untouched copy of the base model (v5)
- the tokenizer graft: resizes the text embedding for an extended vocabulary, with
  separate backbone and embedding learning rates and frozen inherited rows (v3/v4)
- per-language validation sets, which log as \`valid [ne]\` and \`valid [en]\`
- cu126 torch wheels in \`pyproject.toml\` and \`uv.lock\`

Depth distillation (\`distill_teacher_*\`) is upstream code and isn't part of this patch.

Restore it:

```bash
git clone https://github.com/kyutai-labs/pocket-tts.git pocket_TTS/repo
cd pocket_TTS/repo && git checkout $(cat ../patches/UPSTREAM_BASE)
git apply ../patches/pocket_tts_training.patch
```

Our training configs live in `pocket_TTS/configs/`; copy them into `repo/training/configs/`.

# Training configs

These are the two configs that produced the shipped models. They live in
`repo/training/configs/` when running, but `repo/` is an upstream pocket-tts
checkout with its own git history and is not tracked here — so the copies in this
directory are the tracked source of truth.

| file | stage | output |
|---|---|---|
| `nepali_finetune.yaml` | 1 — finetune the 24-layer English release on Nepali | `runs/nepali_teacher_24l/` |
| `nepali_distill.yaml` | 2 — depth-distil the teacher into a 6-layer student | `/workspace/nepali_student_6l/` |

To use them, copy back into `repo/training/configs/` and launch via
`scripts/train_teacher.sh` / `scripts/train_student.sh`. See
[`../docs/ADR.md`](../docs/ADR.md) for why each setting is what it is.

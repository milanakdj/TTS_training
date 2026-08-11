#!/usr/bin/env python3
"""Renders whisper-finetune.py's build_model_card() against stub state.

The card is a ~90-line f-string that only executes after training finishes, so a
typo in it costs a full run to discover. This execs just that one function with
fake globals -- no torch, no GPU, no dataset -- and checks it renders.

    python test_model_card.py
"""

import ast
from pathlib import Path

SRC = Path(__file__).with_name("whisper-finetune.py")


class _Args:  # stands in for Seq2SeqTrainingArguments
    bf16, fp16 = True, False
    warmup_ratio, weight_decay, max_grad_norm = 0.05, 0.0, 1.0
    gradient_checkpointing, optim, generation_max_length = True, "adamw_torch", 225


def load_build_model_card():
    """Exec only the build_model_card def, with every global it reads stubbed."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_model_card"
    )
    ns = {
        "MODEL_NAME": "openai/whisper-medium",
        "MODEL_VARIANT": "medium",
        "HF_DATASET_ID": "lilgoose7777/slr-combined-nepali-tts2",
        "CKPT_REPO_ID": "milanakdj/whisper-medium-nepali-checkpoints",
        "LANGUAGE": "nepali",
        "TASK": "transcribe",
        "NUM_EPOCHS": 3,
        "LEARNING_RATE": 1e-5,
        "PER_DEVICE_TRAIN_BATCH_SIZE": 16,
        "PER_DEVICE_EVAL_BATCH_SIZE": 16,
        "GRADIENT_ACCUMULATION_STEPS": 1,
        "MAX_LABEL_LENGTH": 448,
        "NUM_ROWS": 177000,
        "TRAIN_VAL_TEST_SPLIT": (0.80, 0.10, 0.10),
        "SEED": 42,
        "training_args": _Args(),
        "vectorized_datasets": {
            "train": [0] * 141600,
            "validation": [0] * 17700,
            "test": [0] * 17700,
        },
    }
    exec(compile(ast.Module([fn], []), "<card>", "exec"), ns)
    return ns["build_model_card"]


def main():
    build = load_build_model_card()

    card = build("milanakdj/whisper-medium-nepali-final", wer=21.5, cer=6.25, steps=3429, best_wer=22.1)
    assert card.startswith("---\n"), "card must open with YAML frontmatter"
    assert card.count("\n---\n") >= 1, "frontmatter must be closed"
    assert "base_model: openai/whisper-medium" in card
    assert "| Test WER | 21.50% |" in card
    assert "| Best eval WER | 22.10% |" in card
    assert "| Steps trained | 3429 |" in card
    assert "| Effective batch size | 16 |" in card
    assert "| Precision | bf16 |" in card
    assert "141600 / 17700 / 17700" in card
    assert "80% / 10% / 10%" in card
    assert "milanakdj/whisper-medium-nepali-final" in card  # usage snippet model_id

    # cell 16's path: promoted checkpoint, no test metrics -- must not crash on None
    partial = build("milanakdj/whisper-medium-nepali-final", steps=3429)
    assert "| Test WER | n/a% |" in partial
    assert "| Best eval WER | n/a% |" in partial

    # every {placeholder} resolved -- a surviving brace means a broken f-string
    for c in (card, partial):
        assert "{" not in c, "unrendered placeholder in card"

    print(f"OK -- card renders ({len(card)} chars, {len(card.splitlines())} lines)")


if __name__ == "__main__":
    main()

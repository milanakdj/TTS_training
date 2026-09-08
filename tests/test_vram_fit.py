#!/usr/bin/env python3
"""Does the current whisper-finetune.py config fit in the free VRAM?

Answers in ~1 minute instead of 25 minutes of preprocessing followed by an OOM.
Reads the real config out of whisper-finetune.py (no duplicated numbers), builds
the model, and runs a few training steps on synthetic batches shaped exactly like
the real ones -- Whisper's encoder input is fixed-size (3000 frames), so a fake
batch has the same memory profile as a real one.

    python test_vram_fit.py            # use the config as written
    python test_vram_fit.py --batch 8  # try a different per-device batch

Reports peak VRAM and whether it fits, plus whether adamw_bnb_8bit actually
loads on this machine.
"""

import argparse
import ast
import os
import sys
from pathlib import Path

SRC = Path(__file__).with_name("whisper-finetune.py")
STEPS = 3  # step 1 allocates optimizer state; later steps show the steady-state peak


def _is_config_only(node):
    """True for a plain assignment, or an if-block whose branches only assign.

    The point is to pick up the `if MODEL_VARIANT == "medium":` batch-size block
    while refusing to execute anything with side effects -- the resume block in
    the same file would otherwise start downloading checkpoints from the Hub.
    """
    if isinstance(node, ast.Assign):
        return True
    if isinstance(node, ast.If):
        return all(_is_config_only(n) for n in [*node.body, *node.orelse])
    return False


def load_config():
    """Exec only the config assignments from whisper-finetune.py, so this test
    and the trainer can never disagree about the batch size."""
    os.environ.setdefault("HF_TOKEN", "dummy-not-used")  # config reads it at import
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    ns = {"os": os}
    for node in tree.body:
        if not _is_config_only(node):
            continue
        try:
            exec(compile(ast.Module([node], []), "<config>", "exec"), ns)
        except Exception:
            continue  # depends on something we skipped; not a training knob
    return ns


def main():
    import torch
    from transformers import WhisperForConditionalGeneration

    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=cfg["PER_DEVICE_TRAIN_BATCH_SIZE"])
    ap.add_argument("--optim", default=cfg["OPTIM"])
    ap.add_argument("--variant", default=cfg["MODEL_VARIANT"])
    ap.add_argument(
        "--grad-checkpointing",
        action="store_true",
        default=cfg["GRADIENT_CHECKPOINTING"],
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device -- nothing to measure")

    free_b, total_b = torch.cuda.mem_get_info()
    budget_gb = free_b / 1e9
    print(
        f"[gpu] {torch.cuda.get_device_name(0)}: {budget_gb:.1f}GB free / "
        f"{total_b / 1e9:.1f}GB total ({(total_b - free_b) / 1e9:.1f}GB held by others)"
    )
    print(
        f"[cfg] whisper-{args.variant} | batch={args.batch} | optim={args.optim} | "
        f"grad_checkpointing={args.grad_checkpointing}"
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{args.variant}"
    ).cuda()
    if args.grad_checkpointing:
        # Same non-reentrant path the training args request.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    model.config.use_cache = False

    # Match the trainer's optimizer, since its state is a large fixed cost.
    if args.optim == "adamw_bnb_8bit":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
        except Exception as e:
            sys.exit(
                f"\nadamw_bnb_8bit is unavailable here: {type(e).__name__}: {e}\n"
                f"Set OPTIM = 'adafactor' in whisper-finetune.py and re-run."
            )
    elif args.optim == "adafactor":
        from transformers.optimization import Adafactor

        optimizer = Adafactor(model.parameters(), lr=1e-5, relative_step=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Whisper's encoder input is always 3000 frames, so shape -- not content --
    # determines memory. Labels at MAX_LABEL_LENGTH is the worst case.
    n_mels = model.config.num_mel_bins
    features = torch.randn(args.batch, n_mels, 3000, device="cuda")
    labels = torch.randint(
        0, model.config.vocab_size, (args.batch, cfg["MAX_LABEL_LENGTH"]), device="cuda"
    )

    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(STEPS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=features, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(
                f"[step {step + 1}/{STEPS}] loss={loss.item():.3f} "
                f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                flush=True,
            )
    except torch.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1e9
        sys.exit(
            f"\nOOM at batch={args.batch} (peak {peak:.1f}GB of {budget_gb:.1f}GB free)."
            f"\nHalve PER_DEVICE_TRAIN_BATCH_SIZE and double GRADIENT_ACCUMULATION_STEPS."
        )

    peak = torch.cuda.max_memory_allocated() / 1e9
    headroom = budget_gb - peak
    print(f"\nFITS: peak {peak:.1f}GB of {budget_gb:.1f}GB free ({headroom:.1f}GB spare)")
    if headroom < 2:
        print("  Headroom is thin -- a long batch could still OOM. Consider one step down.")
    elif headroom > peak * 0.5:
        print(
            f"  Room to spare: try --batch {args.batch * 2} and halve "
            f"GRADIENT_ACCUMULATION_STEPS to train faster at the same effective batch."
        )


if __name__ == "__main__":
    main()

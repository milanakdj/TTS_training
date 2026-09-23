#!/usr/bin/env python3
"""Nepali+English Pocket-TTS v4 24-layer TEACHER -- runnable example.

This is the unguided teacher and is not the shippable model: use
milanakdj/pocket-tts-nepali-en-6l-v4 (the student) instead. This script exists
so the checkpoint is at least loadable and inspectable, e.g. before
re-distilling a new student from training/checkpoint_00250000.pt.
"""
import sys
import time

import scipy.io.wavfile as wav
import torch
from huggingface_hub import hf_hub_download
from pocket_tts.models.tts_model import TTSModel

from ne_frontend import assert_clean, normalize

CONFIG = "hf://milanakdj/pocket-tts-nepali-en-24l-teacher-v4/config.yaml"
DEFAULT_TEXT = "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ।"

# see the student's inference.py for why this is needed explicitly
_TOKENIZER = hf_hub_download(
    repo_id="milanakdj/pocket-tts-nepali-en-24l-teacher-v4",
    filename="tokenizer/ne_en_9682.model",
)

torch.set_num_threads(4)


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else None
    text = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEXT
    if voice is None:
        sys.exit(f"usage: {sys.argv[0]} <voice_prompt.wav> [text]")

    model = TTSModel.load_model(config=CONFIG)
    model.to("cpu")
    sr = model.config.mimi.sample_rate

    state = model.get_state_for_audio_prompt(voice)
    # keep_latin=True -- see the student's inference.py for why this matters
    # for this tokenizer (default transliterates English into Devanagari).
    text = normalize(text, keep_latin=True)
    assert_clean(text, model=_TOKENIZER)

    t0 = time.monotonic()
    audio = model.generate_audio(state, text, copy_state=True)
    elapsed = time.monotonic() - t0

    samples = audio.detach().cpu().numpy().squeeze()
    wav.write("out.wav", sr, samples)
    secs = len(samples) / sr
    print(f"spoke: {text}")
    print(f"wrote out.wav -- {secs:.2f}s of audio in {elapsed:.2f}s "
          f"({secs / elapsed:.2f}x real-time on {torch.get_num_threads()} threads)")


if __name__ == "__main__":
    main()

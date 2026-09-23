#!/usr/bin/env python3
"""Nepali+English Pocket-TTS v4 (6-layer distilled bilingual student) -- runnable
CPU example.

    pip install pocket-tts
    python inference.py my_voice.wav "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ।"
    python inference.py my_voice.wav "Hello, how are you today?"

`my_voice.wav` is 3-5 seconds of the voice you want, mono. You supply it: no
reference audio ships with this repo, because the training corpus contains real
people who did not consent to having their voices redistributed as cloning prompts.

READ THIS BEFORE YOU COPY THE load_model CALL
---------------------------------------------
`eos_threshold=EOS_THRESHOLD` is not optional on this model. pocket-tts defaults
to -4.0; at that value this checkpoint's ENGLISH generations end early (English
is only 5% of the training mix and its EOS head is the one that undercalibrated
-- Nepali is insensitive to this knob across the whole range tested). The value
below was chosen on 40 held-out utterances (20 Nepali + 20 English) disjoint from
the evaluation set, scored on transcribed content (WER), not generated duration
-- duration is a documented lying proxy for this model family. See the model card.
"""
import sys
import time

import os

import scipy.io.wavfile as wav
import torch
from huggingface_hub import hf_hub_download
from pocket_tts.models.tts_model import TTSModel

from ne_frontend import assert_clean, normalize

CONFIG = "hf://milanakdj/pocket-tts-nepali-en-6l-v4/config.yaml"
EOS_THRESHOLD = 0.0
DEFAULT_TEXT = "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ।"

# ne_frontend.py's assert_clean() defaults to checking against v2's OLD
# nepali_bpe4000.model (no byte fallback) -- wrong tokenizer for this repo and
# not shipped alongside it. This one (ne_en_9682, WITH byte fallback) is what
# actually backs these weights; point assert_clean at it explicitly instead of
# letting it fall through to a path that only exists on the training box.
_TOKENIZER = hf_hub_download(
    repo_id="milanakdj/pocket-tts-nepali-en-6l-v4",
    filename="tokenizer/ne_en_9682.model",
)

# Pin the thread count. Letting torch grab every core on a machine that is doing
# anything else makes all the threads contend, and the wall time then measures
# contention rather than compute.
torch.set_num_threads(4)


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else None
    text = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEXT
    if voice is None:
        sys.exit(f"usage: {sys.argv[0]} <voice_prompt.wav> [text]")

    model = TTSModel.load_model(config=CONFIG, eos_threshold=EOS_THRESHOLD)
    model.to("cpu")
    sr = model.config.mimi.sample_rate

    # Extracting the prompt state is the slow part and it is reusable: do it once
    # per voice, then synthesize any number of sentences (either language) from
    # the same state.
    state = model.get_state_for_audio_prompt(voice)

    # keep_latin=True is required here. normalize() defaults to keep_latin=False,
    # which TRANSLITERATES English into Devanagari phonetics ("Hello" ->
    # "हेल्लो") -- correct for v2's tokenizer (no byte fallback, so Latin script
    # would otherwise become <unk>), wrong for this one. ne_en_9682 encodes
    # Latin script natively, which is the entire point of this bilingual model;
    # keep_latin=True keeps English words as English words instead of spelling
    # them out phonetically in Nepali script.
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

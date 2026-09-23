#!/usr/bin/env python3
"""Nepali Pocket-TTS v3 (6-layer distilled student) -- runnable CPU example.

    pip install pocket-tts
    python inference.py my_voice.wav "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ।"

`my_voice.wav` is 3-5 seconds of the voice you want, mono. You supply it: no
reference audio ships with this repo, because the training corpus contains real
people who did not consent to having their voices redistributed as cloning prompts.

READ THIS BEFORE YOU COPY THE load_model CALL
---------------------------------------------
`eos_threshold=EOS_THRESHOLD` is not optional on this model. pocket-tts defaults
to -4.0; at that value this checkpoint ends the utterance after a few frames and
you get roughly a quarter of the speech you asked for, with no error and no
warning. The threshold is not expressible in config.yaml (the Config schema is
strict and has no such field), so it has to be passed here, every time.

Why: the student's backbone was distilled onto the teacher's cfg-2.0-combined
activations while its EOS head stayed frozen at the teacher's weights, so the
head reads a distribution it was not calibrated for and fires early. The value
below was chosen on 40 held-out utterances that are disjoint from the evaluation
set. See the model card.
"""
import sys
import time

import scipy.io.wavfile as wav
import torch
from pocket_tts.models.tts_model import TTSModel

from ne_frontend import assert_clean, normalize

CONFIG = "hf://milanakdj/pocket-tts-nepali-6l-v3/config.yaml"
EOS_THRESHOLD = 0.0
DEFAULT_TEXT = "नेपाल हिमाल, पहाड र तराई गरी तीन भौगोलिक क्षेत्रमा विभाजित छ।"

# Pin the thread count. Letting torch grab every core on a machine that is doing
# anything else makes all the threads contend, and the wall time then measures
# contention rather than compute -- that mismeasurement hid this model's entire
# speed advantage over its teacher until the count was pinned.
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
    # per voice, then synthesize any number of sentences from the same state.
    state = model.get_state_for_audio_prompt(voice)

    # v3's tokenizer has byte fallback, so unnormalized digits and Latin words no
    # longer become <unk> and get deleted -- they get read literally instead.
    # normalize() still verbalizes numbers into Nepali words, which is the only
    # way to get "२०२४" spoken as a year rather than as four digits.
    text = normalize(text)
    assert_clean(text)

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

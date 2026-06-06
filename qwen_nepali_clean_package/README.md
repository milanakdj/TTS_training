# Qwen3-TTS Nepali Fine-Tuning Package

Goal: prepare everything needed for Milan to run a clean Qwen3-TTS Nepali training experiment on a stronger GPU.

## Why This Exists

Indic Parler is currently ahead for Nepali because it officially supports Nepali. Qwen3-TTS is stronger/newer, but Nepali is not officially supported, so we should test it carefully instead of doing blind large training.

The first target is:

```text
clean dataset -> Qwen codec tokens -> train without padding crash -> generate Nepali test samples
```

Only after that should we scale.

## Files

- `MILAN_RUNBOOK.md`: main file to send to Milan. It has the GPU commands.
- `qwen_1_7b_config.json`: recommended first-run config.
- `test_sentences_nepali.txt`: fixed Nepali sentences for comparing Qwen vs Indic.
- `inspect_dataset_speakers.py`: finds good speakers in a dataset.
- `prepare_small_dataset.py`: creates Qwen `train_raw.jsonl` and clean 24 kHz WAV files.
- `check_codec_jsonl.py`: checks/filters Qwen codec token output.
- `generate_qwen_samples.py`: generates WAV samples from the trained Qwen checkpoint.
- `run_qwen_1_7b_gpu.sh`: one-script GPU runner after setup.
- `SEND_TO_MILAN_MESSAGE.txt`: short message you can send to Milan.

## Data Rule

For the first Qwen run, use:

```text
Titung/nepali-tts-tagged-combined
```

It is already processed/tagged for fine-tuning, has 10k rows, audio at 24kHz, and includes useful acoustic fields like SNR, PESQ, phonemes, speaking rate, noise/reverb labels, and `text_description`.

If possible, still prefer one clean speaker/ref audio.

Bad first run:

```text
random mixed speakers + random reference audio
```

Better first run:

```text
one speaker + one clean reference audio + exact transcripts
```

Reason: Qwen fine-tuning is single-speaker focused, so mixed speakers can create unstable or scratchy output.

## Your Part

You only need:

1. Hugging Face access/token for gated datasets.
2. Milan/Kiran confirmation of which dataset/speaker should be used.
3. Milan's GPU to run the heavy training.

Everything else is prepared as scripts and commands.

## Suggested First Message To Milan

```text
I updated the Qwen3-TTS 1.7B package with Titung/nepali-tts-tagged-combined as the primary dataset. It is already processed/tagged for fine-tuning, so the plan is to run a small clean Qwen run first, verify codec/padding and sample quality, then compare the same sentences against Indic before scaling.
```

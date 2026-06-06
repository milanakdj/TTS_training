import argparse
from pathlib import Path


def load_sentences(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--speaker", default="nepali_speaker")
    parser.add_argument("--sentences", default="qwen_nepali_clean/test_sentences_nepali.txt")
    parser.add_argument("--output-dir", default="qwen_nepali_clean/outputs/generated_samples")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=args.attn,
    )

    for index, sentence in enumerate(load_sentences(args.sentences), start=1):
        wavs, sr = model.generate_custom_voice(
            text=sentence,
            speaker=args.speaker,
            max_new_tokens=args.max_new_tokens,
        )
        out_path = output_dir / f"qwen_nepali_{index:02d}.wav"
        sf.write(out_path, wavs[0], sr)
        print(f"{out_path}\t{sentence}")


if __name__ == "__main__":
    main()

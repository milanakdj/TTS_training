"""Build tokenizer_v3/ne_en_9682.model -- Kyutai's English vocab, EXTENDED for Nepali.

Why a graft and not a fresh joint BPE. The v2 tokenizer (nepali_bpe4000) was trained
from scratch on Nepali, which threw away two properties the Kyutai base shipped with:
`byte_fallback` (so English, digits and `-` became <unk>, and the model deletes <unk>
rather than mispronouncing it) and any relationship to the pretrained text embedding.

Keeping ids 0..3999 byte-identical to Kyutai's tokenizer means row i of the
pretrained `flow_lm.conditioner.embed.weight` still means piece i, so English
survives the finetune as *inherited weights* rather than as something the model must
relearn from replay data. Nepali pieces are appended at 4000+ and are the only rows
that need fresh initialization -- see build_manifest_v3.py and nepali_finetune_v3.yaml.

Measured on 40,000 held-in transcripts:

    nepali_bpe4000  38.7 tok/utt, 662 utterances containing <unk>
    ne_en_9682      34.8 tok/utt, 0 utterances containing <unk>

So the grafted tokenizer is also ~10% cheaper per utterance than the one it replaces.

    python build_tokenizer_v3.py            # writes tokenizer_v3/
"""
import os
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

OUT = "/root/tts/TTS_training/pocket_TTS/tokenizer_v3"
NE_TEXT = "/workspace/tok_v3/ne_text.txt"          # one transcript per line
KYUTAI = ("hf://kyutai/pocket-tts-without-voice-cloning/"
          "languages/english_2026-04_24l/tokenizer.model"
          "@e81d79e8194ad4c7ce879c87a4258ef20cbf2487")


def harvest(path=f"{OUT}/ne_only_6k.model"):
    """A Nepali-only BPE, used purely as a source of good Nepali pieces."""
    if os.path.exists(path):
        return path
    spm.SentencePieceTrainer.train(
        input=NE_TEXT, model_prefix=path[:-6], vocab_size=6000, model_type="bpe",
        character_coverage=1.0, byte_fallback=True, split_digits=True,
        num_threads=8, input_sentence_size=400000, shuffle_input_sentence=True)
    return path


def graft(kyutai_model, ne_model, out=f"{OUT}/ne_en_9682.model"):
    base = pb.ModelProto(); base.ParseFromString(open(kyutai_model, "rb").read())
    ne = pb.ModelProto(); ne.ParseFromString(open(ne_model, "rb").read())
    have = {p.piece for p in base.pieces}
    lo = min(p.score for p in base.pieces if p.type == 1)

    # Single Devanagari characters FIRST. Without them ४-९ and the danda fall to
    # byte fallback at three tokens each -- encodable, but needlessly expensive.
    singles = [chr(c) for c in range(0x0900, 0x0980)] + ["‌", "‍"]
    order = ([c for c in singles if c not in have]
             + [p.piece for p in ne.pieces if p.type == 1 and p.piece not in have])

    for i, piece in enumerate(order):
        q = base.pieces.add()
        q.piece, q.type = piece, 1
        q.score = lo - 1.0 - i * 1e-3      # strictly below every inherited piece
    base.trainer_spec.vocab_size = len(base.pieces)
    open(out, "wb").write(base.SerializeToString())
    with open(out.replace(".model", ".vocab"), "w") as fh:
        for p in base.pieces:
            fh.write(f"{p.piece}\t{p.score}\n")
    return out, len(base.pieces)


def verify(kyutai_model, out):
    """The two invariants that make embedding inheritance legal. Do not skip."""
    old = spm.SentencePieceProcessor(model_file=kyutai_model)
    new = spm.SentencePieceProcessor(model_file=out)
    n = old.get_piece_size()
    assert all(old.id_to_piece(i) == new.id_to_piece(i) for i in range(n)), \
        "id drift in the inherited range -- the pretrained embedding no longer applies"
    probes = ["Foodmandu costs 45 rupees", "COVID-19 vaccine",
              "The quick brown fox jumps over the lazy dog."]
    assert all(old.encode(t) == new.encode(t) for t in probes), "English encoding changed"
    for t in ["सन् २०२४ मा ४५ जना विद्यार्थी थिए।", "मैले Foodmandu बाट 45 रुपैयाँमा अर्डर गरें।"]:
        assert 0 not in new.encode(t), t
    print(f"verified: ids 0..{n-1} inherited, English unchanged, no <unk>")


if __name__ == "__main__":
    from pocket_tts.utils.utils import download_if_necessary
    os.makedirs(OUT, exist_ok=True)
    ky = download_if_necessary(KYUTAI)
    out, size = graft(ky, harvest(), )
    print(f"wrote {out}  vocab={size}")
    verify(ky, out)

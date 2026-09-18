"""Build an alternative SentencePiece BPE for the from-scratch runs.

The shipped recipe uses a 4,000-piece BPE. `parakeet-tdt_ctc-110m`, whose config
supplies this run's topology, was trained with 1,024 -- and a from-scratch CTC
head has to learn every one of those classes from random init, with no pretrained
decoder to inherit. Smaller vocabulary trades alignment slack (more tokens per
word, so lower T/U) for a much easier classification problem.

Writes tokenizer.model + tokenizer.vocab + vocab.txt, which is the trio NeMo's
`tokenizer: {dir: ..., type: bpe}` expects; vocab.txt is not produced by
sentencepiece and is generated from the model here.
"""
import argparse, json, os, random
import sentencepiece as spm

p = argparse.ArgumentParser()
p.add_argument("--manifest", default="/workspace/asr_pretrain_v2/manifests/train_mix.jsonl")
p.add_argument("--out", default="/workspace/asr_pretrain_v2/tokenizer_1k")
p.add_argument("--vocab-size", type=int, default=1024)
p.add_argument("--sample", type=int, default=400000)
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
txt = os.path.join(a.out, "corpus.txt")
random.seed(0)
n = 0
with open(a.manifest) as f, open(txt, "w") as g:
    for i, l in enumerate(f):
        if i % 3:
            continue
        g.write(json.loads(l)["text"].strip() + "\n")
        n += 1
        if n >= a.sample:
            break
print(f"[corpus] {n} lines -> {txt}")

spm.SentencePieceTrainer.train(
    input=txt, model_prefix=os.path.join(a.out, "tokenizer"),
    vocab_size=a.vocab_size, model_type="bpe",
    character_coverage=1.0,          # two scripts; do not drop rare Devanagari
    bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,
    input_sentence_size=a.sample, shuffle_input_sentence=True,
    num_threads=16,
)
sp = spm.SentencePieceProcessor(model_file=os.path.join(a.out, "tokenizer.model"))
with open(os.path.join(a.out, "vocab.txt"), "w") as g:
    for i in range(sp.get_piece_size()):
        g.write(sp.id_to_piece(i) + "\n")
print(f"[done] {sp.get_piece_size()} pieces in {a.out}")

"""Text normalisation to the tokenizer's actual character set.

Preflight measured a 3.8% UNK rate on the raw manifests. The BPE at
`/workspace/asr_pretrain_v2/tokenizer` was built from a lowercased,
digit-free, punctuation-light corpus, so uppercase Latin, every digit (ASCII and
Devanagari) and most punctuation all map to <unk>. Training on that teaches the
model to emit <unk> as a target -- PRETRAINING_FROM_SCRATCH.md Step 3: "those
currently become trainable <unk> targets and there is no reason to inherit that."

The fix is the doc's: normalise the text, not the tokenizer. Order matters --
digits are verbalised *before* the UNK sweep, or they would simply be deleted and
the model would be trained to skip every number it hears.

Nepali numbers use the validated verbalizer from the OOV work (South Asian scale,
0-99 as a table because Nepali is irregular there). English numbers get a local
speller; num2words is not installed and this needs no dependency.
"""
import re, os, sys, unicodedata, importlib.util

_VN = "/workspace/oov_distill/verbalize_ne.py"
_ne = None
if os.path.exists(_VN):
    _s = importlib.util.spec_from_file_location("verbalize_ne", _VN)
    _ne = importlib.util.module_from_spec(_s); _s.loader.exec_module(_ne)

DEV_DIGITS = "०१२३४५६७८९"
_D2A = {d: str(i) for i, d in enumerate(DEV_DIGITS)}
_ZW = dict.fromkeys(map(ord, "‌‍­﻿"), None)
_PUNCT_MAP = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
              "―": "-", "…": " ", " ": " "}

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]


def _en_int(n):
    if n < 20: return _EN_ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _EN_TENS[t] + (" " + _EN_ONES[r] if r else "")
    for v, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"),
                    (100, "hundred")):
        if n >= v:
            h, r = divmod(n, v)
            return f"{_en_int(h)} {name}" + (f" {_en_int(r)}" if r else "")
    return _EN_ONES[n]


def _en_number(a):
    """Long runs are read digit-by-digit, as phone numbers and IDs actually are."""
    if "." in a:                      # the regex also matches decimals
        i, _, f = a.partition(".")
        head = _en_number(i) if i else "zero"
        return head + " point " + " ".join(_EN_ONES[int(c)] for c in f)
    if len(a) >= 7 or (len(a) > 1 and a[0] == "0"):
        return " ".join(_EN_ONES[int(c)] for c in a)
    n = int(a)
    return _en_int(n) if n < 10**12 else " ".join(_EN_ONES[int(c)] for c in a)


def _say(a, lang):
    if lang == "ne-NP" and _ne is not None:
        r, _k = _ne.verbalize(a)
        if r: return r
    if lang == "ne-NP":                     # verbalizer unavailable: digit-by-digit
        return " ".join(DEV_DIGITS[int(c)] for c in a)
    return _en_number(a)


_NUM = re.compile(r"\d+(?:\.\d+)?")


class Normalizer:
    def __init__(self, sp):
        self.sp, self.unk, self._ok = sp, sp.unk_id(), {}

    def _char_ok(self, ch):
        """Cache per character. Encoded in context so the word-boundary marker
        does not make an otherwise-fine character look unknown."""
        v = self._ok.get(ch)
        if v is None:
            v = not any(i == self.unk for i in self.sp.encode("क" + ch))
            self._ok[ch] = v
        return v

    def __call__(self, t, lang):
        t = unicodedata.normalize("NFC", t).translate(_ZW)
        for k, v in _PUNCT_MAP.items(): t = t.replace(k, v)
        t = "".join(_D2A.get(c, c) for c in t)          # Devanagari -> ASCII digits
        t = _NUM.sub(lambda m: " " + _say(m.group(0), lang) + " ", t)
        t = t.lower()
        # Whatever the tokenizer still cannot represent becomes a space rather
        # than an <unk> target.
        t = "".join(c if (c == " " or self._char_ok(c)) else " " for c in t)
        return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(
        model_file="/workspace/asr_pretrain_v2/tokenizer/tokenizer.model")
    nz = Normalizer(sp)
    for lang, s in [("ne-NP", "काठमाडौं । २०७५ सालमा ADB ले ९२ वटा (परियोजना) ‘सम्झौता’ ."),
                    ("en-US", "Cosine theta is adjacent side by hypotenuse, 25 times."),
                    ("en-US", "Call 9841234567 on 12 March 2019 — OK?")]:
        o = nz(s, lang)
        ids = sp.encode(o)
        print(f"{lang} {s!r}\n   -> {o!r}\n   unk={sum(1 for i in ids if i==sp.unk_id())}")

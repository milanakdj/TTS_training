"""Text frontend for Nepali pocket-TTS: make every input character representable.

WHY THIS EXISTS
    nepali_bpe4000.model has no byte_fallback. Anything outside its 4,000 pieces
    encodes to a single <unk> (id 0), and the model does not mispronounce <unk> --
    it either deletes the word or truncates the rest of the utterance. Measured on
    the 6L student: "मैले 45 रुपैयाँ तिरें।" produced 0.80 s of audio reading only
    "मैले". So the fix belongs here, before the tokenizer, not in more training.

WHAT IS NOT REPRESENTABLE (probed against the .model, see SAFE below)
    - every Latin letter                    -> <unk>
    - every ASCII digit                     -> <unk>
    - Devanagari digits ४ ५ ६ ७ ८ ९         -> <unk>   (only ० १ २ ३ are in vocab)
    - the ASCII hyphen                      -> <unk>   (so "COVID-19" fails twice)

normalize() is total: its output is guaranteed <unk>-free, and assert_clean()
re-encodes to prove it. Prefer failing loudly here over silent deletion downstream.
"""
import json, re, sys, unicodedata

# ---------------------------------------------------------------- numbers ---
# Vendored from /workspace/oov_distill/verbalize_ne.py so this module is
# self-contained and can travel with the weights the way the tokenizer does.
# Nepali uses the South Asian scale (हजार 10^3, लाख 10^5, करोड 10^7) and 0-99 is
# irregular -- it is a lookup table, not a composition rule. That is the part
# naive verbalizers get wrong.
_DIG = "०१२३४५६७८९"
_D2A = {d: str(i) for i, d in enumerate(_DIG)}
ONES = ["शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ"]
TENS = {
 10:'दस',11:'एघार',12:'बाह्र',13:'तेह्र',14:'चौध',15:'पन्ध्र',16:'सोह्र',17:'सत्र',18:'अठार',19:'उन्नाइस',
 20:'बीस',21:'एक्काइस',22:'बाइस',23:'तेइस',24:'चौबिस',25:'पच्चिस',26:'छब्बिस',27:'सत्ताइस',28:'अठ्ठाइस',29:'उनन्तिस',
 30:'तीस',31:'एकतिस',32:'बत्तिस',33:'तेत्तिस',34:'चौँतिस',35:'पैँतिस',36:'छत्तिस',37:'सैँतिस',38:'अठतिस',39:'उनन्चालिस',
 40:'चालिस',41:'एकचालिस',42:'बयालिस',43:'त्रिचालिस',44:'चवालिस',45:'पैँतालिस',46:'छयालिस',47:'सच्चालिस',48:'अठचालिस',49:'उनन्चास',
 50:'पचास',51:'एकाउन्न',52:'बाउन्न',53:'त्रिपन्न',54:'चवन्न',55:'पचपन्न',56:'छपन्न',57:'सन्ताउन्न',58:'अन्ठाउन्न',59:'उनान्साठी',
 60:'साठी',61:'एकसट्ठी',62:'बयसट्ठी',63:'त्रिसट्ठी',64:'चौंसट्ठी',65:'पैंसट्ठी',66:'छयसट्ठी',67:'सतसट्ठी',68:'अठसट्ठी',69:'उनन्सत्तरी',
 70:'सत्तरी',71:'एकहत्तर',72:'बहत्तर',73:'त्रिहत्तर',74:'चौहत्तर',75:'पचहत्तर',76:'छयहत्तर',77:'सतहत्तर',78:'अठहत्तर',79:'उनासी',
 80:'असी',81:'एकासी',82:'बयासी',83:'त्रियासी',84:'चौरासी',85:'पचासी',86:'छयासी',87:'सतासी',88:'अठासी',89:'उनान्नब्बे',
 90:'नब्बे',91:'एकानब्बे',92:'बयानब्बे',93:'त्रियानब्बे',94:'चौरानब्बे',95:'पन्चानब्बे',96:'छयानब्बे',97:'सन्तानब्बे',98:'अन्ठानब्बे',99:'उनान्सय',
}
SCALE = [(10**7, "करोड"), (10**5, "लाख"), (1000, "हजार"), (100, "सय")]


def to_ascii(s):
    return "".join(_D2A.get(c, c) for c in s)


def say_int(n):
    if n < 10:
        return ONES[n]
    if n < 100:
        return TENS[n]
    for v, name in SCALE:
        if n >= v:
            head, rest = divmod(n, v)
            out = f"{say_int(head)} {name}"
            return out if not rest else f"{out} {say_int(rest)}"
    return ONES[n]


def say_digits(s):
    """Digit-by-digit -- how phone numbers and long IDs are actually read."""
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


# Every character the tokenizer can encode without emitting <unk>. Probed against
# the .model, not assumed -- the punctuation half of this was wrong when guessed:
# ( ) ; - and the typographic quotes all encode to <unk>, ZWJ does too, and of the
# two danda forms only । survives (॥ does not).
SAFE = set(
    "ँंःअआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरऱलळवशषसह़ऽािीुूृॆेैॉोौ्"
    "क़ख़ग़ज़ड़ढ़फ़य़ॠ।०१२३"
) | set("!\"',.:? \t\n…\u200c")

# v3 only: ne_en_9682 inherits Kyutai's byte_fallback, so Latin is representable
# and must survive the final guard rather than being stripped out of existence.
SAFE_LATIN = SAFE | set("abcdefghijklmnopqrstuvwxyz"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Latin letters spelled the way a Nepali speaker says them, for acronyms.
LETTER = {
    "a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ", "g": "जी",
    "h": "एच", "i": "आई", "j": "जे", "k": "के", "l": "एल", "m": "एम", "n": "एन",
    "o": "ओ", "p": "पी", "q": "क्यू", "r": "आर", "s": "एस", "t": "टी", "u": "यू",
    "v": "भी", "w": "डब्लु", "x": "एक्स", "y": "वाई", "z": "जेड",
}

# The 85 terms true_gap.json says appear in NO corpus, plus the ones that a
# letter-by-letter or rule-based reading would get audibly wrong.
LEXICON = {
    "ayush": "आयुष", "agni group": "अग्नि ग्रुप", "apf": "ए पी एफ",
    "bams": "बी ए एम एस", "bhms": "बी एच एम एस", "btech": "बी टेक",
    "baidyanath": "बैद्यनाथ", "big mart": "बिग मार्ट", "bouddhanath": "बौद्धनाथ",
    "braindigit": "ब्रेनडिजिट", "broadlink": "ब्रोडलिङ्क", "bungamati": "बुङ्गमती",
    "cfo": "सी एफ ओ", "cft": "सी एफ टी", "cg group": "सी जी ग्रुप",
    "cg net": "सी जी नेट", "century bank": "सेन्चुरी बैङ्क", "chapagaon": "चापागाउँ",
    "classic tech": "क्लासिक टेक", "cloudfactory": "क्लाउड फ्याक्ट्री",
    "cotiviti": "कोटिभिटी", "dudbc": "डी यू डी बी सी", "dwidp": "डी डब्लु आई डी पी",
    "deerwalk": "डियरवाक", "dish home": "डिस होम", "f1soft": "एफ वान सफ्ट",
    "foodmandu": "फुडमान्डु", "g20": "जी बीस", "genese": "जेनेसे",
    "goldstar": "गोल्डस्टार", "hdmi": "एच डी एम आई", "hamro bazar": "हाम्रो बजार",
    "hamro patro": "हाम्रो पात्रो", "hisense": "हाइसेन्स", "horlicks": "हर्लिक्स",
    "iaea": "आई ए ई ए", "json": "जेसन", "jyoti group": "ज्योति ग्रुप",
    "khokana": "खोकना", "kwiks": "क्विक्स", "leapfrog": "लिपफ्रग",
    "mclr": "एम सी एल आर", "msc": "एम एस सी", "mtech": "एम टेक",
    "machhapuchchhre bank": "मच्छापुच्छ्रे बैङ्क", "mayos": "मायोस",
    "mero lagani": "मेरो लगानी", "moco": "मोको", "moit": "एम ओ आई टी",
    "nimh": "एन आई एम एच", "nescafe": "नेस्काफे", "nimbus group": "निम्बस ग्रुप",
    "okcredit": "ओके क्रेडिट", "phcc": "पी एच सी सी", "php": "पी एच पी",
    "pathao": "पठाओ", "quixote": "क्विक्जोट", "roi": "आर ओ आई",
    "rtgs": "आर टी जी एस", "sdo": "एस डी ओ", "swift": "स्विफ्ट",
    "salesberry": "सेल्सबेरी", "sano paila": "सानो पाइला", "sastodeal": "सस्तोडिल",
    "shangrila group": "शाङ्ग्रिला ग्रुप", "smart cell": "स्मार्ट सेल",
    "smartdoko": "स्मार्टडोको", "subisu": "सुबिसु", "surf excel": "सर्फ एक्सेल",
    "surya group": "सूर्य ग्रुप", "tcl": "टी सी एल", "tootle": "टुटल",
    "verisk": "भेरिस्क", "wipo": "विपो", "wai wai": "वाइ वाइ",
    "worldlink": "वर्ल्डलिङ्क", "yatriapp": "यात्री एप",
    "younginnovations": "यङ इनोभेसन्स", "zandu": "जन्डु", "indrive": "इनड्राइभ", "esewa": "इसेवा", "khalti": "खल्ती",
    "ncell": "एनसेल", "ntc": "एन टी सी", "nea": "एन ई ए",
    # high-frequency general vocabulary the rule engine would mangle
    "covid": "कोभिड", "email": "इमेल", "internet": "इन्टरनेट", "online": "अनलाइन",
    "mobile": "मोबाइल", "computer": "कम्प्युटर", "facebook": "फेसबुक",
    "youtube": "युट्युब", "google": "गुगल", "whatsapp": "ह्वाट्सएप",
    "app": "एप", "bank": "बैङ्क", "office": "अफिस", "police": "पुलिस",
}

# Multi-word lexicon keys, longest first, so "cg group" wins over "cg".
_PHRASES = sorted((k for k in LEXICON if " " in k), key=len, reverse=True)

# Digraphs before single letters; order within each tier matters.
_DIGRAPH = [
    ("chh", "छ"), ("cch", "च्छ"), ("sch", "स्क"), ("tch", "च"),
    ("ch", "च"), ("sh", "श"), ("th", "थ"), ("ph", "फ"), ("kh", "ख"),
    ("gh", "घ"), ("dh", "ध"), ("bh", "भ"), ("jh", "झ"), ("zh", "झ"),
    ("ck", "क"), ("ng", "ङ्ग"), ("qu", "क्व"), ("wh", "व"),
]
_VOWEL = {  # (independent, matra)
    "a": ("अ", ""), "aa": ("आ", "ा"), "i": ("इ", "ि"), "ee": ("ई", "ी"),
    "u": ("उ", "ु"), "oo": ("उ", "ु"), "e": ("ए", "े"), "ai": ("ऐ", "ै"),
    "o": ("ओ", "ो"), "au": ("औ", "ौ"), "ou": ("औ", "ौ"), "ea": ("ी", "ी"),
}
_CONS = {
    "b": "ब", "c": "क", "d": "ड", "f": "फ", "g": "ग", "h": "ह", "j": "ज",
    "k": "क", "l": "ल", "m": "म", "n": "न", "p": "प", "q": "क", "r": "र",
    "s": "स", "t": "ट", "v": "भ", "w": "व", "x": "क्स", "y": "य", "z": "ज",
}


def _translit(w):
    """Rule-based Latin -> Devanagari for words the lexicon does not carry.

    Approximate by design: it exists so an unseen brand is *spoken*, however
    imperfectly, instead of deleted. Anything whose pronunciation matters
    belongs in LEXICON.
    """
    w, out, i = w.lower(), [], 0
    while i < len(w):
        for src, dst in _DIGRAPH:
            if w.startswith(src, i):
                out.append(dst); i += len(src); break
        else:
            for vl in (2, 1):                      # longest vowel first: 'ai' before 'a'
                v = w[i:i + vl]
                if v in _VOWEL:
                    indep, matra = _VOWEL[v]
                    # A matra needs a consonant to attach to; otherwise the vowel
                    # stands alone. Note matra is "" for short 'a' -- that is the
                    # inherent vowel and must append nothing, not fall back to अ
                    # (which is what turned "eSewa" into एसेवअ).
                    attached = out and out[-1] and out[-1][-1] not in "ािीुूृेैोौअआइईउऊएऐओऔ"
                    out.append(matra if attached else indep)
                    i += vl; break
            else:
                c = w[i]
                if c in _CONS:
                    out.append(_CONS[c])
                    # no vowel follows -> close the syllable with a halant
                    nxt = w[i + 1:i + 2]
                    if not nxt or nxt not in "aeiou":
                        out.append("्")
                i += 1
    s = "".join(out)
    return re.sub(r"्$", "", s)                    # drop a word-final halant


def _is_acronym(w):
    """All-caps and either short or vowel-free -> read it letter by letter."""
    if not w.isupper() or not w.isalpha():
        return False
    return len(w) <= 5 or not set(w.lower()) & set("aeiou")


def _spell(w):
    return " ".join(LETTER[c] for c in w.lower() if c in LETTER)


def _num(tok):
    """Devanagari or ASCII digit run -> spoken Nepali."""
    a = to_ascii(tok)
    if not a.isdigit():
        return None
    if len(a) >= 7:
        return say_digits(a)                       # phone numbers are read digit by digit
    return say_int(int(a))


def _word(w, keep_latin=False):
    # v3: the grafted tokenizer (ne_en_9682) encodes Latin with zero <unk>, so a
    # Latin word no longer has to be transliterated to survive. Keeping the
    # surface is what puts code-switched Nepali ("... नाइकीको Airforce ...") into
    # the training distribution at all; transliterating it away is why v2 never
    # saw a single Latin character inside a Devanagari sentence.
    if keep_latin and re.fullmatch(r"[A-Za-z]+", w):
        return w
    low = w.lower()
    if low in LEXICON:
        return LEXICON[low]
    if w.isdigit() or to_ascii(w).isdigit():
        return _num(w)
    if re.fullmatch(r"[A-Za-z]+", w):
        return _spell(w) if _is_acronym(w) else _translit(w)
    # mixed alphanumeric such as F1Soft, G20, COVID19 -> split and recurse
    if re.fullmatch(r"[A-Za-z0-9]+", w):
        return " ".join(_word(p) for p in re.findall(r"[A-Za-z]+|[0-9]+", w))
    return w


def normalize(text, keep_latin=False):
    """Return text the tokenizer can encode with zero <unk>.

    `keep_latin=True` targets the v3 tokenizer (`tokenizer_v3/ne_en_9682`), which
    inherits Kyutai's byte_fallback and therefore encodes Latin cleanly. It keeps
    pure-Latin words as-is instead of transliterating them; digits are still
    verbalized, glosses still dropped, and the output is still restricted to a
    representable character set (SAFE plus ASCII letters). Default stays False so
    the v2 model and its `nepali_bpe4000` tokenizer are unaffected.
    """
    safe = SAFE_LATIN if keep_latin else SAFE
    t = unicodedata.normalize("NFC", text or "")
    # A Latin gloss in brackets -- "ट्याक्सी (Taxi)" -- is a transcription
    # convention, not something anyone says. Reading it back doubles the word,
    # so drop the whole bracket rather than transliterate inside it.
    t = re.sub(r"[(\[{][A-Za-z0-9 .,'&-]*[)\]}]", " ", t)
    t = t.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u0965", "\u0964").replace(";", ",").replace("\u200d", "")
    t = t.replace("-", " ").replace("\u2014", " ").replace("\u2013", " ").replace("/", " ")
    t = t.replace("%", " प्रतिशत ").replace("₹", " रुपैयाँ ").replace("$", " डलर ")
    t = t.replace("&", " एन्ड ").replace("@", " एट ").replace("+", " प्लस ")

    if not keep_latin:
        # Multi-word lexicon entries are transliterations; under keep_latin the
        # Latin surface is what we want in the text, so skip them.
        for p in _PHRASES:                         # multi-word entries first
            t = re.sub(re.escape(p), LEXICON[p], t, flags=re.IGNORECASE)

    # decimals, which say_int cannot take whole
    t = re.sub(r"([0-9०-९]+)[.]([0-9०-९]+)",
               lambda m: f"{_num(m.group(1))} दशमलव {say_digits(to_ascii(m.group(2)))}", t)

    t = re.sub(r"[A-Za-z0-9]+|[०-९]+",
               lambda m: _word(m.group(0), keep_latin) or "", t)

    # Final guard: anything still outside the tokenizer's charset is dropped
    # rather than allowed through to become <unk>.
    t = "".join(c for c in t if c in safe)
    return re.sub(r"\s+", " ", t).strip()


_SP = None


def _default_model():
    """Find nepali_bpe4000.model without assuming this file's checkout layout."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.environ.get("NE_BPE_MODEL"),
              os.path.join(here, "nepali_bpe4000.model"),
              os.path.join(here, "tokenizer", "nepali_bpe4000.model"),
              os.path.join(here, os.pardir, "tokenizer", "nepali_bpe4000.model"),
              "/root/tts/TTS_training/pocket_TTS/tokenizer/nepali_bpe4000.model"):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "nepali_bpe4000.model not found; set NE_BPE_MODEL or pass model=")


def assert_clean(text, model=None):
    """Encode and raise if any piece is <unk>. This is the acceptance test."""
    global _SP
    if _SP is None:
        import sentencepiece as spm
        _SP = spm.SentencePieceProcessor(model_file=model or _default_model())
    ids = _SP.encode(text)
    if 0 in ids:
        bad = [p for p, i in zip(_SP.encode(text, out_type=str), ids) if i == 0]
        raise ValueError(f"unencodable after normalize(): {bad} in {text!r}")
    return ids


if __name__ == "__main__":
    for line in (sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]):
        out = normalize(line)
        assert_clean(out)
        print(json.dumps({"in": line, "out": out}, ensure_ascii=False))

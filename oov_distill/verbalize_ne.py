"""Devanagari number verbalizer for Nepali.

The OOV list is 4,193 entries but only 695 are lexical; the other 3,498 are
digit strings -- 2,446 short numbers/years/decimals and 1,052 phone numbers.
Those are not a vocabulary problem. 1,052 distinct phone numbers teach a model
nothing that the ten digits plus a reading grammar do not, so they are handled
here in text rather than by hunting for audio of each one.

Nepali uses the South Asian scale: हजार (10^3), लाख (10^5), करोड (10^7).
Below 100 Nepali has a distinct word for every value, so 0-99 is a table, not a
composition rule -- this is the part naive verbalizers get wrong.

    python verbalize_ne.py --selftest
    python verbalize_ne.py --oov /root/tts/TTS_training/oov_words.txt --out pairs.jsonl
"""
import argparse, json, re

DIG = '०१२३४५६७८९'
D2A = {d: str(i) for i, d in enumerate(DIG)}
ONES = ['शून्य','एक','दुई','तीन','चार','पाँच','छ','सात','आठ','नौ']

# 0-99 is irregular in Nepali; spell it out rather than compose it.
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
SCALE = [(10**7,'करोड'), (10**5,'लाख'), (1000,'हजार'), (100,'सय')]

def to_ascii(s): return ''.join(D2A.get(c, c) for c in s)

def say_int(n):
    if n < 10:  return ONES[n]
    if n < 100: return TENS[n]
    for v, name in SCALE:
        if n >= v:
            head, rest = divmod(n, v)
            out = f"{say_int(head)} {name}"
            return out if not rest else f"{out} {say_int(rest)}"
    return ONES[n]

def say_digits(s):
    """Digit-by-digit -- how phone numbers and long IDs are actually read."""
    return ' '.join(ONES[int(c)] for c in s if c.isdigit())

def verbalize(tok):
    """-> (reading, kind). Phone-like and years get their own conventions."""
    a = to_ascii(str(tok).strip())
    if re.fullmatch(r'\d+\.\d+', a):
        i, f = a.split('.')
        return f"{say_int(int(i))} दशमलव {say_digits(f)}", 'decimal'
    if not re.fullmatch(r'\d+', a):
        return None, 'not-a-number'
    n = int(a)
    if len(a) >= 7:
        return say_digits(a), 'phone/id'          # read digit-by-digit
    if len(a) == 4 and (1900 <= n <= 2100 or 2000 <= n <= 2100):
        return say_int(n), 'year-ad'
    if len(a) == 4 and 2000 <= n <= 2199:
        return say_int(n), 'year-bs'
    return say_int(n), 'cardinal'

SELFTEST = [('२१','एक्काइस'), ('९२','बयानब्बे'), ('२०','बीस'), ('८','आठ'),
            ('१६','सोह्र'), ('१२','बाह्र'), ('३','तीन'), ('१००','एक सय'),
            ('२०७५','दुई हजार पचहत्तर'), ('४५','पैँतालिस'), ('११','एघार')]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--oov'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.selftest:
        # Expected strings are taken from what the flex ASR actually heard in the
        # Premal audio, so this checks the verbalizer against real speech.
        bad = 0
        for src, want in SELFTEST:
            got, kind = verbalize(src)
            ok = (got == want)
            bad += (not ok)
            print(f"  {'ok ' if ok else 'BAD'} {src:>6} -> {got!r:<34} ({kind})" + ('' if ok else f"  want {want!r}"))
        print(f"\n{len(SELFTEST)-bad}/{len(SELFTEST)} pass")
        raise SystemExit(1 if bad else 0)
    toks = [l.strip() for l in open(a.oov, encoding='utf-8') if l.strip()]
    n = 0; kinds = {}
    with open(a.out, 'w', encoding='utf-8') as fh:
        for t in toks:
            r, k = verbalize(t)
            kinds[k] = kinds.get(k, 0) + 1
            if r is None: continue
            fh.write(json.dumps({'written': t, 'spoken': r, 'kind': k}, ensure_ascii=False) + '\n')
            n += 1
    print(f"verbalized {n}/{len(toks)} -> {a.out}")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]): print(f"  {v:>5}  {k}")

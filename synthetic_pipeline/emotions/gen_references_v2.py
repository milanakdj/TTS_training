"""
Reference-clip generator v2 -- register-constrained Gemini TTS.

Why v2 exists: the v1 reference set (emotions/dataset_gemini_*) is unusable.
An audit found the angry and happy prompts made Gemini SHOUT an octave above
Amrita's natural register (F0 407/420 Hz vs neutral 206 Hz; <1-2% of spectral
energy below 250 Hz). That broke speaker identity -- CAM++ cross-emotion
similarity 0.529 against a same-speaker threshold of ~0.562, i.e. the four
folders were effectively four different speakers -- and made angry and happy
acoustically identical (MFCC centroid distance 14.0 vs 57.7 to neutral).

The fix, verified empirically before writing this: an explicit register
constraint in the prompt. A constrained "happy" probe measured F0 206.2 Hz with
19.6% of energy below 250 Hz, matching the neutral anchor (206.1 Hz / 24.9%).

So: emotion must come from tone, tempo, intensity and emphasis -- never from
pitch height or volume. Every prompt below carries REGISTER verbatim.

Quota: gemini-3.1-flash-tts-preview free tier is ~3 RPM per key and an unknown
daily cap. Three keys rotate with per-key cooldown tracking and 429 backoff.
Emotions are generated in priority order (angry and happy first -- they have
zero usable clips today) so that running out of quota still leaves the most
valuable references on disk. Fully resumable: existing non-trivial wavs skip.

Keys come from GEMINI_KEYS (comma-separated) in the environment. Never hardcode
them -- they must not land in this file or in any log.

Run: GEMINI_KEYS="k1,k2,k3" python3 gen_references_v2.py [--target-per-emotion N]
"""
import argparse
import json
import os
import random
import sys
import time
import wave

from google import genai
from google.genai import types

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Which TTS model. gemini-3.1-flash-tts-preview hit a hard daily cap on all
# three keys after 20 clips; gemini-2.5-flash-preview-tts has a SEPARATE quota
# and was still serving. Model choice is a CLI flag because the two must not be
# mixed inside one reference set -- the same voice name can render with a
# different character across model families, which would give the dataset two
# speaker identities. Each model writes to its own directory; screening then
# picks whichever set is internally consistent.
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
# Rebound in main() once the model/output dir are known; defined here so the
# module-level helpers have something valid if imported.
V2_DIR = os.path.join(OUT_ROOT, "v2_25flash")
STATE_PATH = os.path.join(V2_DIR, "gen_state.json")
LOG_PATH = os.path.join(V2_DIR, "gen_references_v2.log")
VOICE_NAME = "Zephyr"          # same prebuilt voice as v1, so the anchor identity carries over
SR = 24000

MIN_SPACING_PER_KEY = 22.0     # seconds; free tier is ~3 RPM per key
BACKOFF_429 = 65.0             # a 429 means that key's minute window is spent
MAX_ATTEMPTS_PER_CLIP = 4

# The constraint that makes this work. Do not soften it.
REGISTER = (
    "CRITICAL VOICE CONSTRAINT: keep her natural speaking register throughout — "
    "median pitch near 205 Hz, never above 260 Hz. She must NOT shout, shriek, "
    "raise her voice into a high register, or go falsetto. The emotion comes from "
    "vocal tone, intensity, tempo and emphasis, NOT from pitch height or volume. "
    "Chest voice only. Clean studio recording, no background noise, no music."
)

# Emotion styling. Each must separate from the others on tone/tempo/voice
# quality while sharing one register. angry vs happy collapsed in v1 because
# both prompts just said "loud and intense" -- these deliberately pull apart:
# angry is tense/clipped/dark, happy is smiling/bright/light.
STYLES = {
    "angry": (
        "A Nepali woman named Amrita speaking in cold, controlled, seething anger. "
        "Tense and clipped, jaw tight, consonants hard and precise, each word bitten off. "
        "Restrained fury held just under the surface — dangerous rather than loud. "
        "Slightly faster than normal, with hard stressed onsets and curt pauses."
    ),
    "happy": (
        "A Nepali woman named Amrita speaking with bright, warm happiness and an audible "
        "smile in her voice. Light, buoyant and gently animated, lifting at the ends of "
        "phrases. Relaxed and open-throated with a bright timbre, a little quicker than "
        "normal, genuinely pleased and affectionate."
    ),
    # v1 of this prompt produced 281-372 Hz clips (gate window is 162-257 Hz):
    # "breathless with anticipation" and "urgency" both read as pitch lift. The
    # excitement is now pinned to tempo and stress only, with the register named twice.
    "excited": (
        "A Nepali woman named Amrita speaking with eager, energised excitement. "
        "The excitement is carried ENTIRELY by tempo and emphasis — quick, forward-leaning "
        "pace, crisp stressed onsets, momentum and delighted urgency. Her pitch stays "
        "exactly where it sits in her calm speech: she does NOT lift, rise, squeal or turn "
        "shrill, and she is never breathless or airy. Grounded chest register throughout, "
        "as if thrilled but deliberately keeping her voice down because others are nearby."
    ),
    # v1 of this prompt said "soft and a little breathy" and got exactly that: the
    # clips measured +19.3 dB high/low spectral tilt and 8x the flatness of neutral
    # (whisper-like), which dropped CAM++ identity to 0.43-0.53 vs 0.85 for neutral.
    # Breathiness is now forbidden outright; the sadness rides on tempo and contour.
    "sad": (
        "A Nepali woman named Amrita speaking in dejected, resigned sorrow. "
        "Heavy, slow and weary, every phrase trailing downward, emotionally drained. "
        "She speaks in FULL VOICE at normal conversational loudness — modal chest voice, "
        "fully and firmly phonated. NO whispering, NO breathy or airy voice, NO hushed or "
        "half-voiced murmuring, and do not drop the volume. "
        "ABSOLUTELY NO crying, NO sobbing, NO sniffling, NO gasping and no other "
        "non-speech sounds — only sad, spoken words. The sadness comes from slow tempo, "
        "falling intonation and flat, heavy delivery, never from softening the voice."
    ),
    "neutral": (
        "A calm, objective, clear Nepali female narrator named Amrita recording a "
        "standard announcement in a quiet studio. Natural, even-paced and professional, "
        "steady speed, clear articulation, no emotional colouring at all."
    ),
}

# Emotion-congruent Nepali. Reference clips only supply timbre and voice quality,
# but congruent text makes Gemini act the emotion far more reliably than neutral
# prose does.
UTTERANCES = {
    "angry": [
        "मैले पटक पटक भनेँ, तर तिमीले मेरो कुरा कहिल्यै सुनेनौ।",
        "यो अन्याय अब मैले सहन सक्दिनँ, मलाई तुरुन्तै जवाफ देऊ।",
        "तिमीले मलाई कति पटक ढाँट्यौ, अब मेरो धैर्यको सीमा सकियो।",
        "मैले तिमीलाई विश्वास गरेकी थिएँ, तर तिमीले त्यो विश्वास तोड्यौ।",
        "अब बहाना बनाउन छोड, र आफ्नो गल्ती स्वीकार गर।",
        "यति धेरै लापरवाही गरेपछि पनि तिमीलाई कुनै पछुतो छैन।",
        "मेरो समय र मेहनत तिमीले व्यर्थमा बर्बाद गरिदियौ।",
        "यो कुरा यहीँ सक्किँदैन, मैले जवाफ खोजेरै छाड्छु।",
        "तिमीले जे गरे पनि म अब चुप लागेर बस्ने छैन।",
        "मलाई अरू कसैको सल्लाह चाहिँदैन, मैले निर्णय गरिसकेँ।",
        "बारम्बार सम्झाउँदा पनि तिमीले किन बुझ्दैनौ।",
        "यो व्यवहार मैले अब कदापि सहन गर्ने छैन।",
    ],
    "happy": [
        "मेरो नतिजा आयो, मैले परीक्षामा सबैभन्दा राम्रो अङ्क ल्याएँ।",
        "धेरै वर्षपछि आज हामी सबै एकै ठाउँमा भेटियौं, कति रमाइलो भयो।",
        "तिमीले पठाएको उपहार मैले पाएँ, मलाई साँच्चै धेरै मन पर्यो।",
        "आज बिहानको मौसम यति सुन्दर छ कि मन आफै प्रसन्न भयो।",
        "मेरो सानो भान्जीले आज पहिलो पटक मलाई फुपू भनेर बोलाई।",
        "हामीले सोचेभन्दा धेरै राम्रो काम भयो, सबैलाई धन्यवाद।",
        "लामो समयपछि घर फर्किँदा आमाको हातको खाना खाएँ।",
        "तिम्रो सफलताको खबर सुनेर मेरो मन खुसीले भरियो।",
        "आजको दिन मेरो जीवनको सबैभन्दा राम्रो दिन बन्यो।",
        "सबै मिलेर काम गर्दा कति सजिलो र रमाइलो हुन्छ।",
        "यो सुन्दर ठाउँमा आउन पाउँदा म साँच्चै भाग्यमानी छु।",
        "तिमीलाई भेट्न पाउँदा सधैं मेरो दिन राम्रो हुन्छ।",
    ],
    "excited": [
        "हामीले जित्यौं, हामीले साँच्चै जित्यौं, मलाई विश्वास नै लागेको छैन।",
        "भोलि हामी हिमाल घुम्न जाँदैछौं, मैले सबै तयारी सकेँ।",
        "छिटो आऊ, तिमीले यो हेर्नै पर्छ, यो एकदमै अद्भुत छ।",
        "मेरो नाम छनौट भयो, मैले त्यो अवसर पाएँ।",
        "अब केही घण्टामा नतिजा आउँछ, मलाई पर्खिनै सकिनँ।",
        "यो खबर सुनेपछि मैले तुरुन्तै तिमीलाई फोन गरेँ।",
        "हामीले वर्षौंदेखि सोचेको योजना अन्ततः सुरु हुँदैछ।",
        "यत्रो भीडमा पनि मैले तिमीलाई देखेँ, कति राम्रो भयो।",
        "मैले नयाँ काम पाएँ, भोलिदेखि नै सुरु गर्दैछु।",
        "सबै जना आइपुगेका छन्, अब कार्यक्रम सुरु हुनै लाग्यो।",
    ],
    "sad": [
        "आज घर सुनसान छ, उहाँको आवाज सुन्ने आशा सधैंको लागि टुट्यो।",
        "मैले सबै गुमाएँ, अब मेरो साथमा एक्लोपन मात्र बाँकी छ।",
        "मलाई माफ गरिदेऊ, मैले चाहेर पनि केही राम्रो गर्न सकिनँ।",
        "सोचेकी थिएँ परिस्थिति अलि सुध्रिएला, तर झन् बिग्रँदै गयो।",
        "अब फेरि पहिले जस्तो केही पनि हुने छैन।",
        "म केवल अलिकति माया चाहन्थिएँ, त्यो पनि मैले पाइनँ।",
        "यो पीडा कसैलाई सुनाउन पनि मन लाग्दैन।",
        "बर्षौंको मेहनत एकै दिनमा सबै व्यर्थ भयो।",
        "उहाँ बित्नुभएको आज एक वर्ष पुग्यो, तर मन अझै मान्दैन।",
        "मैले धेरै प्रयास गरेँ, तर अन्तमा एक्लै रह्याछु।",
        "जीवन कहिलेकाहीँ यति निर्दयी हुन्छ कि शब्द नै भेटिँदैन।",
        "बिदाइको बेला केही भन्न खोजेँ, तर आवाज नै निस्किएन।",
    ],
    # Neutral is the anchor: the speaker centroid every other emotion is
    # screened against, so it gets as many clips as the emotional sets.
    "neutral": [
        "यो कार्यक्रम प्रत्येक बिहान सात बजे प्रसारण हुनेछ।",
        "सेवाग्राहीले आवश्यक कागजात लिएर कार्यालयमा सम्पर्क राख्नुहोला।",
        "तापक्रम आज दिनभरि सामान्य रहने अनुमान गरिएको छ।",
        "नयाँ नियम आउँदो महिनाको पहिलो सातादेखि लागू हुनेछ।",
        "यात्रुहरूलाई समयमै स्थानमा आइपुग्न अनुरोध गरिन्छ।",
        "यो पुस्तकालय बिहान नौ बजेदेखि साँझ पाँच बजेसम्म खुल्ला रहन्छ।",
        "नेपाल एउटा सुन्दर र विविध संस्कृतिले भरिएको बहुभाषिक देश हो।",
        "थप जानकारीको लागि हाम्रो आधिकारिक वेबसाइट हेर्न सक्नुहुनेछ।",
        "काम सुरु गर्नुअघि सबै नियम र निर्देशनहरू राम्ररी पढ्नुहोला।",
        "अहिलेको समयमा प्रविधिको प्रयोगले हाम्रो जीवनलाई निकै सहज बनाएको छ।",
        "बैठकको निर्णय सबै सदस्यलाई लिखित रूपमा जानकारी गराइनेछ।",
        "यो तथ्याङ्क गत वर्षको सर्वेक्षणमा आधारित रहेको छ।",
    ],
}

# angry and happy have zero usable v1 clips, so they go first; neutral already
# has 6 verified keepers and goes last.
PRIORITY = ["angry", "happy", "sad", "excited", "neutral"]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class KeyPool:
    """Round-robins keys, respecting a per-key minimum spacing and 429 cooldowns."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.next_ok = {k: 0.0 for k in self.keys}
        self.dead = set()
        self.clients = {k: genai.Client(api_key=k) for k in self.keys}

    def alive(self):
        return [k for k in self.keys if k not in self.dead]

    def acquire(self):
        """Blocks until some key is usable. Returns (key, client) or None if all dead."""
        while True:
            live = self.alive()
            if not live:
                return None
            now = time.time()
            ready = sorted(live, key=lambda k: self.next_ok[k])
            k = ready[0]
            wait = self.next_ok[k] - now
            if wait > 0:
                time.sleep(min(wait, 30.0))
                continue
            return k, self.clients[k]

    def penalise(self, k, seconds):
        self.next_ok[k] = time.time() + seconds

    def kill(self, k, why):
        self.dead.add(k)
        log(f"  key ...{k[-6:]} retired: {why}")


def synth(client, prompt, model_id):
    r = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
                )
            ),
        ),
    )
    part = r.candidates[0].content.parts[0]
    return part.inline_data.data


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def build_prompt(emotion, text):
    return (
        f"# AUDIO PROFILE: Amrita\n"
        f"## THE SCENE\n{STYLES[emotion]}\n"
        f"### DIRECTOR'S NOTES\n{REGISTER}\n"
        f"#### TRANSCRIPT\nSpeak only these Nepali words, nothing else:\n"
        f'Amrita: "{text}"\n'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-emotion", type=int, default=12)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--emotions", default=None,
                    help="comma-separated subset to generate (default: all). Use when only "
                         "some emotions failed screening and their prompts were revised.")
    ap.add_argument("--out-subdir", default=None,
                    help="directory under emotions/ to write into; defaults to a "
                         "name derived from the model so sets never get mixed")
    args = ap.parse_args()

    global V2_DIR, STATE_PATH, LOG_PATH
    sub = args.out_subdir or ("v2" if "3.1" in args.model else "v2_25flash")
    V2_DIR = os.path.join(OUT_ROOT, sub)
    os.makedirs(V2_DIR, exist_ok=True)
    STATE_PATH = os.path.join(V2_DIR, "gen_state.json")
    LOG_PATH = os.path.join(V2_DIR, "gen_references_v2.log")

    keys = [k.strip() for k in os.environ.get("GEMINI_KEYS", "").split(",") if k.strip()]
    if not keys:
        sys.exit("GEMINI_KEYS not set")
    log(f"starting: model={args.model} dir={sub} keys={len(keys)} "
        f"target={args.target_per_emotion}/emotion")

    pool = KeyPool(keys)
    state = {"done": [], "failed": []}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)

    # Interleave emotions in priority order so quota exhaustion degrades evenly
    # within the priority ranking rather than starving whole later emotions.
    wanted = PRIORITY
    if args.emotions:
        wanted = [e.strip() for e in args.emotions.split(",") if e.strip()]
        unknown = [e for e in wanted if e not in UTTERANCES]
        if unknown:
            sys.exit("unknown emotion(s): " + ", ".join(unknown))
        wanted = [e for e in PRIORITY if e in wanted]
        log("generating only: " + ", ".join(wanted))

    jobs = []
    for i in range(max(len(v) for v in UTTERANCES.values())):
        for emotion in wanted:
            texts = UTTERANCES[emotion]
            if i < min(len(texts), args.target_per_emotion):
                jobs.append((emotion, i, texts[i]))

    made = 0
    for emotion, idx, text in jobs:
        d = os.path.join(V2_DIR, emotion)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"amrita_{emotion}_{idx + 1:03d}.wav")
        if os.path.exists(path) and os.path.getsize(path) > 20000:
            continue

        prompt = build_prompt(emotion, text)
        for attempt in range(1, MAX_ATTEMPTS_PER_CLIP + 1):
            got = pool.acquire()
            if got is None:
                log("ALL KEYS EXHAUSTED -- stopping early")
                json.dump(state, open(STATE_PATH, "w"), indent=1)
                log(f"generated {made} new clips this session")
                return
            k, client = got
            try:
                pcm = synth(client, prompt, args.model)
                pool.penalise(k, MIN_SPACING_PER_KEY)
                if not pcm or len(pcm) < 20000:
                    log(f"  {emotion}/{idx+1:03d} short payload ({len(pcm) if pcm else 0}B), retry")
                    continue
                write_wav(path, pcm)
                dur = len(pcm) / 2 / SR
                state["done"].append({"emotion": emotion, "idx": idx, "text": text,
                                      "path": path, "duration_s": round(dur, 2)})
                json.dump(state, open(STATE_PATH, "w"), ensure_ascii=False, indent=1)
                made += 1
                log(f"  {emotion}/{idx+1:03d} OK {dur:5.2f}s  (key ...{k[-6:]}, {made} made)")
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    # Could be the per-minute window or a daily cap; back off and
                    # let repeated failures retire the key naturally.
                    pool.penalise(k, BACKOFF_429)
                    log(f"  {emotion}/{idx+1:03d} 429 on key ...{k[-6:]}, cooling {BACKOFF_429:.0f}s")
                elif "400" in msg or "safety" in msg.lower():
                    pool.penalise(k, 5)
                    log(f"  {emotion}/{idx+1:03d} 400/safety, retry {attempt}")
                elif "401" in msg or "403" in msg or "API_KEY" in msg.upper():
                    pool.kill(k, "auth rejected")
                else:
                    pool.penalise(k, 10)
                    log(f"  {emotion}/{idx+1:03d} {type(e).__name__}: {msg[:160]}")
        else:
            state["failed"].append({"emotion": emotion, "idx": idx, "text": text})
            json.dump(state, open(STATE_PATH, "w"), ensure_ascii=False, indent=1)
            log(f"  {emotion}/{idx+1:03d} GAVE UP after {MAX_ATTEMPTS_PER_CLIP} attempts")

    log(f"COMPLETE -- generated {made} new clips")


if __name__ == "__main__":
    main()
